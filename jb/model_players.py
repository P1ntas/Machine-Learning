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

def train_and_evaluate(df, years, i, classifier):
    train_years = years[:i+1]  # Use all years up to year i for training
    test_year = years[i+1]     # Testing on year i+1 

    if i+2 >= len(years): 
        return None, None

    predict_year = years[i+2]  # Predicting for year i+2 


    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]
    actual = df[df['year'] == predict_year]  # Changed actual_year to predict_year

    #print the teams that make the playoffs (playoff=1) on actual year, dont repeat teams
    #print("Actual Playoff Teams:")
    #print(actual[actual['playoff'] == 1]['tmID'].unique())

     # Make a copy of the playerID before removing it for training
    test_player_ids = test['playerID'].copy()

    
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
    
    # Predict probabilities for the test set
    proba = clf.predict_proba(X_test)[:, 1]

    # Combine the predicted probabilities with the player IDs
    test_proba_with_ids = pd.DataFrame({'playerID': test_player_ids, 'probability': proba})

    # Filter this DataFrame to only include players who are also in the actual_year dataset
    actual_players = actual['playerID'].unique()
    test_proba_with_ids = test_proba_with_ids[test_proba_with_ids['playerID'].isin(actual_players)]

    # Calculate the average probability for each team based on the players' probabilities
    team_avg_predictions = {}
    for _, row in test_proba_with_ids.iterrows():
        player_id = row['playerID']
        prob = row['probability']
        # Get the team ID from the original test dataframe
        team_id = test.loc[test['playerID'] == player_id, 'tmID'].iloc[0]
        if team_id not in team_avg_predictions:
            team_avg_predictions[team_id] = {'probs': [], 'confID': test.loc[test['playerID'] == player_id, 'confID'].iloc[0]}
        team_avg_predictions[team_id]['probs'].append(prob)

    # Select top 4 teams from each conference
    east_teams = [(tmID, sum(data["probs"]) / len(data["probs"])) for tmID, data in team_avg_predictions.items() if data["confID"] == 'EA']
    west_teams = [(tmID, sum(data["probs"]) / len(data["probs"])) for tmID, data in team_avg_predictions.items() if data["confID"] == 'WE']

    top_east_teams = sorted(east_teams, key=lambda x: x[1], reverse=True)[:4]
    top_west_teams = sorted(west_teams, key=lambda x: x[1], reverse=True)[:4]

    top_teams = top_east_teams + top_west_teams

    team_results = []
    for tmID, avg_prob in top_teams:
        prediction = 1
        if tmID in actual['tmID'].values:
            actual_value = actual[actual['tmID'] == tmID]['playoff'].iloc[0]
        else:
            continue
        team_results.append({
            'tmID': tmID,
            'Year': predict_year,
            'Classifier': classifier.__class__.__name__,
            'Predicted': prediction,
            'Probability': avg_prob,
            'Actual': actual_value,
        })
        #logging.info(f"Data added for team {tmID} in year {predict_year} using classifier {classifier.__class__.__name__}")

    # Calculate accuracy
    accuracy = sum([1 for result in team_results if result['Predicted'] == result['Actual']]) / 8

    return team_results, accuracy

def plot_teams_comparison(prediction_data, classifier_name):
    # Organize the data by years, and then by actual vs predicted
    organized_data = {}

    unique_years = sorted(list(set([entry['Year'] for entry in prediction_data])))

    for year in unique_years:
        year_data = [entry for entry in prediction_data if entry['Year'] == year and entry['Classifier'] == classifier_name]
        if not year_data:  # If no data for the year, continue to the next
            logging.warning(f"No data available for year {year} for classifier {classifier_name}.")
            continue

        # Get actual playoff teams
        actual_teams = [entry['tmID'] for entry in year_data if entry['Actual'] == 1]
        if not actual_teams:
            logging.warning(f"No actual playoff teams found for year {year}.")

        # Get the top 8 predicted teams
        year_data_sorted = sorted(year_data, key=lambda x: x['Probability'], reverse=True)
        predicted_teams = [entry['tmID'] for entry in year_data_sorted[:8]]
        if not predicted_teams:
            logging.warning(f"No teams predicted for playoffs in year {year}.")

        # Calculate accuracy for this year
        correct_predictions = len(set(actual_teams) & set(predicted_teams))
        accuracy = correct_predictions / 8  # As 8 teams make the playoffs

        organized_data[year] = {
            'Actual': actual_teams,
            'Predicted': predicted_teams,
            'Accuracy': accuracy
        }

    if not organized_data:
        logging.error(f"No data to plot for classifier {classifier_name}.")
        return

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
    table.set_fontsize(8)  # Adjust the fontsize value as necessary

    plt.title(f'Comparison of Actual vs Predicted Playoff Teams ({classifier_name})', fontsize=14)  # Adjust title fontsize if needed
    plt.tight_layout()  # Ensure layout is tight and no content is clipped
    plt.show()

def train_model():
    data_file_path = "../ac/basketballPlayoffs/players_teams.csv"
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