import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#import another classifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)


logging.basicConfig(level=logging.INFO)

def read_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"{file_path} not found.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    return None

def pre_process_data(df):
    mapping = {'Y': 1, 'N': 0}
    df['playoff'] = df['playoff'].map(mapping)
    df.fillna(0, inplace=True)
    return df

def get_teams_data():
    teams_file_path = "../basketballPlayoffs/teams.csv"
    teams_df = read_data(teams_file_path)
    if teams_df is not None:
        return pre_process_data(teams_df)
    return None

def get_sample_weights(train):
    max_year = train['year'].max()
    weights = train['year'].apply(lambda x: 2 if x == max_year else 1)
    return weights

def merge_with_team_data(df, teams_df):
    merged_df = df.merge(teams_df[['tmID', 'year', 'playoff']], on=['tmID', 'year'], how='left')
    return merged_df

def get_columns_to_remove():
    return ['lgID', 'tmID', 'playerID','playoff']

def train_and_evaluate(df, years, i, classifier):
    train_years = years[:i]  # Training on data from start till year i-1
    test_year = years[i]    # Testing on year i
    logging.info(f"Training years: {train_years}")
    logging.info(f"Testing year: {test_year}")

    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]
    remove_columns = get_columns_to_remove()

    #print("First few rows of training data:\n", train.head()) # Added
    #print("First few rows of testing data:\n", test.head()) # Added

    X_train = train.drop(remove_columns, axis=1)
    X_test = test.drop(remove_columns, axis=1)
    y_train = train['playoff']
    y_test = test['playoff']

    # Get sample weights
    sample_weights = get_sample_weights(train)

    clf = classifier

    if isinstance(classifier, KNeighborsClassifier) or isinstance(classifier, MLPClassifier):
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train, sample_weight=sample_weights)  # Use the sample weights here
    
    proba = clf.predict_proba(X_test)[:, 1]

    team_avg_predictions = {}
    for tmID, prob in zip(test['tmID'], proba):
        if tmID not in team_avg_predictions:
            team_avg_predictions[tmID] = []
        team_avg_predictions[tmID].append(prob)

    team_results = []
    for tmID, probs in team_avg_predictions.items():
        avg_prob = sum(probs) / len(probs)
        prediction = 1 if avg_prob >= 0.5 else 0
        actual_value = test[test['tmID'] == tmID]['playoff'].iloc[0]
        team_results.append({
            'tmID': tmID,
            'Year': test_year,
            'Classifier': classifier.__class__.__name__,
            'Predicted': prediction,
            'Probability': avg_prob,
            'Actual': actual_value,
        })

    accuracy = sum([1 for result in team_results if result['Predicted'] == result['Actual']]) / len(team_results)

    return team_results, accuracy

def plot_results(years, results_dict):
    for classifier, results in results_dict.items():
        print(classifier, results)
        plt.plot(years, results, label=classifier)

    plt.xlabel('Year Predicted')
    plt.ylabel('Accuracy')
    plt.title('Rolling Window Results for Various Classifiers')
    plt.legend()
    plt.show()

def plot_heatmaps(predictions_df):
    classifiers = predictions_df['Classifier'].unique()
    teams = predictions_df['tmID'].unique()
    years = predictions_df['Year'].unique()

    print("Heatmap data:")
    print(classifiers)
    print(teams)
    print(years)


    # Create a single figure to plot all heatmaps
    fig, axes = plt.subplots(nrows=len(classifiers), figsize=(15, 5*len(classifiers)))


    for idx, classifier in enumerate(classifiers):
        # Create an empty dataframe to store probabilities
        heatmap_data = pd.DataFrame(index=teams, columns=years)
        heatmap_data.fillna(0, inplace=True)

        # Fill in the dataframe with probabilities
        subset = predictions_df[predictions_df['Classifier'] == classifier]
        for _, row in subset.iterrows():
            heatmap_data.at[row['tmID'], row['Year']] = float(row['Probability'])

        # Plot the heatmap
        ax = axes[idx] if len(classifiers) > 1 else axes
        sns.heatmap(heatmap_data, cmap="PuBu", ax=ax, annot=True)
        ax.set_title(classifier)
        ax.set_xlabel("Year")
        ax.set_ylabel("Team ID")

    plt.tight_layout()
    plt.show()


def train_model():
    data_file_path = "../basketballPlayoffs/players_teams.csv"
    df = read_data(data_file_path)

    teams_df = get_teams_data()
    if df is not None and teams_df is not None:
        df = merge_with_team_data(df, teams_df)
        
        years = df['year'].unique()
        years.sort()

        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            #"LogisticRegression": LogisticRegression(max_iter=10000), # Increased max_iter for convergence
            "SVM": SVC(probability=True), # Enable probability estimates
            #"MLP": MLPClassifier(hidden_layer_sizes=(, 50, 10), max_iter=10000) # Increased max_iter for convergence
        }

        results_dict = {}
        prediction_data = []

        for classifier_name, classifier in classifiers.items():
            results = []
            for i in range(2, len(years)):  # Starting the loop from index 2 (which corresponds to 3rd year)
                team_results, accuracy = train_and_evaluate(df, years, i, classifier)
                prediction_data.extend(team_results)
                results.append(accuracy)
                results_dict[classifier_name] = results

        predictions_df = pd.DataFrame(prediction_data)
        predictions_df.to_csv('predictions_results-playersTeams.csv', index=False)

        plot_heatmaps(predictions_df)

if __name__ == "__main__":
    train_model()