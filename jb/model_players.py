import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from model_plot import plot_heatmaps, plot_bar_chart, plot_line_chart
import warnings

warnings.filterwarnings("ignore")


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
    teams_file_path = "../ac/basketballPlayoffs/teams.csv"
    teams_df = read_data(teams_file_path)
    if teams_df is not None:
        return pre_process_data(teams_df)
    return None

def get_sample_weights(train):
    max_year = train['year'].max()
    weights = train['year'].apply(lambda x: 2 if x == max_year else 1)
    return weights

def compute_percentage(numerator, denominator):
    return round(numerator.divide(denominator).where(denominator != 0, 0.0)*100,2)

def merge_with_team_data(df, teams_df):
    player_teams = df.merge(teams_df[['tmID', 'year', 'playoff', 'confID']], on=['tmID', 'year'], how='left')
    #drop unneeded columns
    player_teams.drop('lgID', axis=1, inplace=True)

    # Assuming you have a column 'year' to sort by
    player_teams = player_teams.sort_values(by=['playerID', 'year'])
    player_teams['career_year'] = player_teams.groupby('playerID').cumcount() + 1

    # Regular Season Percentages
    player_teams['ft%'] = compute_percentage(player_teams['ftMade'], player_teams['ftAttempted'])
    player_teams['fg%'] = compute_percentage(player_teams['fgMade'], player_teams['fgAttempted'])
    player_teams['three%'] = compute_percentage(player_teams['threeMade'], player_teams['threeAttempted'])
    player_teams['gs%'] = compute_percentage(player_teams['GS'], player_teams['GP'])

    # Playoffs Percentages
    player_teams['Postft%'] = compute_percentage(player_teams['PostftMade'], player_teams['PostftAttempted'])
    player_teams['Postfg%'] = compute_percentage(player_teams['PostfgMade'], player_teams['PostfgAttempted'])
    player_teams['Postthree%'] = compute_percentage(player_teams['PostthreeMade'], player_teams['PostthreeAttempted'])
    player_teams['Postgs%'] = compute_percentage(player_teams['PostGS'], player_teams['PostGP'])

    return player_teams

def get_columns_to_remove():
    return ['tmID', 'playerID','playoff','confID']

def evaluate_predictions(test_proba_with_ids, actual, default_probability, classifier_name):
    team_avg_predictions = {tmID: {'probs': [], 'confID': confID} for tmID, confID in zip(actual['tmID'], actual['confID'])}

    for _, row in test_proba_with_ids.iterrows():
        player_id = row['playerID']
        if player_id in actual['playerID'].unique():
            if actual[actual['playerID'] == player_id]['stint'].max() > 1:
                continue  # Skip players that have been traded mid-season
            prob = row['probability']
            team_id = actual[actual['playerID'] == player_id]['tmID'].iloc[0]
            confID = actual[actual['playerID'] == player_id]['confID'].iloc[0]
            team_avg_predictions[team_id]['probs'].append(prob)
            team_avg_predictions[team_id]['confID'] = confID

    for tmID, data in team_avg_predictions.items():
        if data['probs']:
            data['avg_prob'] = sum(data['probs']) / len(data['probs'])
        else:
            data['avg_prob'] = default_probability

    east_teams = [(tmID, data['avg_prob']) for tmID, data in team_avg_predictions.items() if data['confID'] == 'EA']
    west_teams = [(tmID, data['avg_prob']) for tmID, data in team_avg_predictions.items() if data['confID'] == 'WE']

    top_east_teams = sorted(east_teams, key=lambda x: x[1], reverse=True)[:8]
    top_west_teams = sorted(west_teams, key=lambda x: x[1], reverse=True)[:8]

    top_teams = top_east_teams + top_west_teams

    team_results = []
    for tmID, avg_prob in top_teams:
        actual_playoff = actual[actual['tmID'] == tmID]['playoff'].iloc[0] if tmID in actual['tmID'].values else 0
        predicted_playoff = 1 if avg_prob > 0.5 else 0
        team_results.append({
            'tmID': tmID,
            'Predicted': predicted_playoff,
            'Probability': avg_prob,
            'Actual': actual_playoff,
            'Year': actual['year'].iloc[0],
            'Classifier': classifier_name
        })

    correct_predictions = sum(1 for result in team_results if result['Predicted'] == result['Actual'])
    total_predictions = len(team_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return team_results, accuracy

def train_and_evaluate(df, years, classifier):
    cumulative_train = pd.DataFrame()
    results = []

    for i in range(len(years) - 2):
        train_years = years[:i+1]
        test_year = years[i+1]
        predict_year = years[i+2]

        # Adding new year data to cumulative train data
        cumulative_train = cumulative_train._append(df[df['year'].isin(train_years)])

        test = df[df['year'] == test_year]
        actual = df[df['year'] == predict_year]

        remove_columns = get_columns_to_remove()
        X_train = cumulative_train.drop(remove_columns, axis=1)
        y_train = cumulative_train['playoff']
        X_test = test.drop(remove_columns, axis=1)

        sample_weights = get_sample_weights(cumulative_train)

        # Train classifier
        if isinstance(classifier, KNeighborsClassifier) or isinstance(classifier, MLPClassifier):
            classifier.fit(X_train, y_train)
        else:
            classifier.fit(X_train, y_train, sample_weight=sample_weights)

        # Predict probabilities for the test set
        proba = classifier.predict_proba(X_test)[:, 1]
        test_proba_with_ids = pd.DataFrame({'playerID': test['playerID'], 'probability': proba})

        # Define default probability
        default_probability = test_proba_with_ids['probability'].mean()

        # Evaluate predictions
        team_results, accuracy = evaluate_predictions(test_proba_with_ids, actual, default_probability, classifier.__class__.__name__)
        results.extend(team_results)

        # Here you might want to save or print the accuracy for this iteration
        print(f"Year: {predict_year}, Accuracy: {accuracy:.2f}")

    return results

def plot_teams_comparison(prediction_data):
    # Organize the data by years and classifiers, then by actual vs predicted
    organized_data = {}

    unique_years = sorted(set(entry['Year'] for entry in prediction_data))
    classifiers = sorted(set(entry['Classifier'] for entry in prediction_data))

    for classifier_name in classifiers:
        for year in unique_years:
            year_data = [entry for entry in prediction_data if entry['Year'] == year and entry['Classifier'] == classifier_name]
            if not year_data:
                logging.warning(f"No data available for year {year} for classifier {classifier_name}.")
                continue

            actual_teams = [entry['tmID'] for entry in year_data if entry['Actual'] == 1]
            predicted_teams = [entry['tmID'] for entry in year_data if entry['Predicted'] == 1]
            
            correct_predictions = sum(entry['Predicted'] == entry['Actual'] for entry in year_data)
            accuracy = correct_predictions / len(year_data) if year_data else 0

            organized_data[(classifier_name, year)] = {
                'Actual': ', '.join(actual_teams),
                'Predicted': ', '.join(predicted_teams),
                'Accuracy': f"{accuracy:.2%}"
            }

    if not organized_data:
        logging.error(f"No data to plot.")
        return

    # Create table data for each classifier
    for classifier_name in classifiers:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        ax.axis('tight')

        table_data = [['Year', 'Actual Teams', 'Predicted Teams', 'Accuracy']]
        for year in unique_years:
            key = (classifier_name, year)
            if key in organized_data:
                data = organized_data[key]
                table_data.append([year, data['Actual'], data['Predicted'], data['Accuracy']])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.1, 0.3, 0.3, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.set_title(f'Comparison of Actual vs Predicted Playoff Teams ({classifier_name})', fontsize=14, pad=20)

        plt.tight_layout()
        plt.show()


def train_model():
    data_file_path = "../ac/basketballPlayoffs/players_teams.csv"
    df = read_data(data_file_path)

    teams_df = get_teams_data()
    if df is not None and teams_df is not None:
        df = merge_with_team_data(df, teams_df)
        
        years = sorted(df['year'].unique())

        classifiers = {
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "DecisionTree": DecisionTreeClassifier(),
            "MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        }

        total_results = []

        for classifier_name, classifier in classifiers.items():
            logging.info(f"Training and evaluating with {classifier_name}")
            results = train_and_evaluate(df, years, classifier)
            total_results += results

            # Convert results to DataFrame and save or print as necessary
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'../predictions_results_{classifier_name}.csv', index=False)

            # Print accuracy for each classifier or save to log
            logging.info(f"Completed training and evaluation for {classifier_name}")

        plot_teams_comparison(total_results)

if __name__ == "__main__":
    train_model()