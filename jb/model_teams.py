import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import logging
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
    # Ensure no division by zero and handle potential errors
    try:
        df['winning_percentage'] = df['won'] / (df['won'] + df['lost']).replace(0, 1)
    except ZeroDivisionError:
        logging.error("Division by zero when calculating winning percentage.")
    return df


def get_columns_to_remove():
    return ['playoff', 'rank', 'seeded', 'firstRound', 'semis', 'finals', 'lgID', 'franchID', 'confID', 'divID', 'name', 'arena']

def train_and_evaluate(df, years, i, classifier):
    # Ensure we have enough years ahead for the split
    if i + 2 >= len(years): 
        logging.info("Insufficient years ahead for training, testing, and prediction. Stopping the process.")
        return None, None, None, None, None

    train_years = years[:i]
    test_year = years[i]
    predict_year = years[i+1]

    logging.info(f"Training years: {train_years}")
    logging.info(f"Testing year: {test_year}")
    logging.info(f"Predicting for year: {predict_year}")

    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]

    # Filter teams that don't exist in the prediction year
    teams_next_year = df[df['year'] == predict_year]['tmID'].unique()
    test = test[test['tmID'].isin(teams_next_year)]

    remove_columns = get_columns_to_remove()

    X_train = train.drop(remove_columns + ['tmID'], axis=1)
    X_test = test.drop(remove_columns + ['tmID'], axis=1)
    y_train = train['playoff']
    y_test = test['playoff']

    # Simplified winning percentage calculation
    X_train['winning_percentage'] = X_train['won'] / (X_train['won'] + X_train['lost'])
    X_test['winning_percentage'] = X_test['won'] / (X_test['won'] + X_test['lost'])

    clf = classifier
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)

    return predictions, proba, y_test.to_numpy(), test['tmID'], accuracy


def plot_results(years, results_dict):
    for classifier, results in results_dict.items():
        # Trim the years list to match the length of results for each classifier
        plt.plot(years[:len(results)], results, label=classifier)

    plt.xlabel('Year Predicted')
    plt.ylabel('Accuracy')
    plt.title('Expanding Window Results for Various Classifiers')
    plt.legend()
    plt.show()

def plot_teams_comparison(years, prediction_data):
    # Let's first organize the data by year, and then by actual vs predicted
    organized_data = {}

    for year in years:
        year_data = [entry for entry in prediction_data if entry['Year'] == year and entry['Actual'] == 1]
        if not year_data:
            continue

        actual_teams = [entry['tmID'] for entry in year_data]

        # Since we're already only looking at teams that made the playoffs, there's no need to sort and slice the year_data list.
        predicted_teams = [entry['tmID'] for entry in year_data]

        correct_predictions = len(set(actual_teams) & set(predicted_teams))
        accuracy = correct_predictions / len(actual_teams)

        organized_data[year] = {
            'Actual': actual_teams,
            'Predicted': predicted_teams,
            'Accuracy': accuracy
        }

    # Increase the size of the figure
    fig, ax = plt.subplots(figsize=(14, 8))  # Adjust the figsize values as necessary

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    table_data = [['Year', 'Actual Teams', 'Predicted Teams', 'Accuracy']]
    for year, data in organized_data.items():
        table_data.append([year, ', '.join(data['Actual']), ', '.join(data['Predicted']), f"{data['Accuracy']:.2f}"])

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')

    # Adjust font size for all cells
    table.auto_set_font_size(False)
    table.set_fontsize(5)  # Adjust the fontsize value as necessary

    plt.title('Comparison of Actual vs Predicted Playoff Teams', fontsize=14)  # Adjust title fontsize if needed
    plt.tight_layout()  # Ensure layout is tight and no content is clipped
    plt.show()


def train_model():
    data_file_path = "../basketballPlayoffs/teams.csv"
    df = read_data(data_file_path)

    if df is not None:
        df = pre_process_data(df)
        years = df['year'].unique()
        years.sort()

        classifiers = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=3),
        }

        results_dict = {}
        prediction_data = []

        for classifier_name, classifier in classifiers.items():
            results = []
            # Adjust the loop to account for the expanding window
            for i in range(1, len(years) - 1): 
                predicted, predicted_proba, actual, tmIDs, accuracy = train_and_evaluate(df, years, i, classifier)

                if predicted is None:  # Check if the function returned None due to insufficient years
                    break

                for p, pp, a, tmID in zip(predicted, predicted_proba, actual, tmIDs):
                    prediction_data.append({
                        'tmID': tmID,
                        'Year': years[i],
                        'Classifier': classifier_name,
                        'Predicted': p,
                        'Probability': pp,
                        'Actual': a
                    })

                results.append(accuracy)
                results_dict[classifier_name] = results
        
        predictions_df = pd.DataFrame(prediction_data)
        predictions_df.to_csv('predictions_results-teams.csv', index=False)

        plot_teams_comparison(df['year'].unique(), prediction_data)

        #plot_results(years[1:-1], results_dict)  # Adjust the years range for plotting

if __name__ == "__main__":
    train_model()