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
#import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from model_plot import plot_heatmaps, plot_bar_chart, plot_line_chart

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
    train_years = years[:i+1]  # Use all years up to year i for training
    test_year = years[i+1]     # Testing on year i+1 

    if i+2 >= len(years): 
        logging.info("Reached the last available year for testing. Stopping the process.")
        return None, None

    predict_year = years[i+2]  # Predicting for year i+2 

    logging.info(f"Training years: {train_years}")
    logging.info(f"Testing year: {test_year}, Predicting for: {predict_year}")

    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]
    actual = df[df['year'] == predict_year]  # Changed actual_year to predict_year
    
    remove_columns = get_columns_to_remove()

    X_train = train.drop(remove_columns, axis=1)
    X_test = test.drop(remove_columns, axis=1)
    y_train = train['playoff']

    # Get sample weights
    sample_weights = get_sample_weights(train)

    clf = classifier

    if isinstance(classifier, KNeighborsClassifier) or isinstance(classifier, MLPClassifier):
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    
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
        if tmID in actual['tmID'].values:
            actual_value = actual[actual['tmID'] == tmID]['playoff'].iloc[0]  # Fetching actual data for year i+2
        else:
            continue  # Skip this team if there's no data for year i+2
        team_results.append({
            'tmID': tmID,
            'Year': predict_year,  # Changed actual_year to predict_year
            'Classifier': classifier.__class__.__name__,
            'Predicted': prediction,
            'Probability': avg_prob,
            'Actual': actual_value,
        })

    accuracy = sum([1 for result in team_results if result['Predicted'] == result['Actual']]) / len(team_results)

    return team_results, accuracy


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
            "SVM": SVC(probability=True), # Enable probability estimates
            "DecisionTree": DecisionTreeClassifier(),
            
        }

        results_dict = {}
        prediction_data = []

        for classifier_name, classifier in classifiers.items():
            results = []
            for i in range(1, len(years) - 2):  # Adjust to ensure there's always a year to predict after the test year
                team_results, accuracy = train_and_evaluate(df, years, i, classifier)
                if team_results:  # Check if team_results is not None
                    prediction_data.extend(team_results)
                    results.append(accuracy)
                    results_dict[classifier_name] = results

        predictions_df = pd.DataFrame(prediction_data)
        predictions_df.to_csv('predictions_results-playersTeams.csv', index=False)

        #make a selector to print or the barchart or the heatmap
        selector = input("Do you want to see the bar chart or the heatmap or line graph? (b/h/l): ")
        if selector == 'b':
            plot_bar_chart(predictions_df)
        elif selector == 'h':
            plot_heatmaps(predictions_df)
        elif selector == 'l':
            plot_line_chart(predictions_df)
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    train_model()