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
from skopt import BayesSearchCV
from model_plot import plot_heatmaps, plot_bar_chart, plot_line_chart
import warnings
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', message="is_sparse is deprecated and will be removed in a future version.")


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
    teams_file_path = "basketballPlayoffs/teams.csv"
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

def aggregate_awards_counts(player_teams):
    #get from awards_players.csv
    awards_file_path = "../ac/basketballPlayoffs/awards_players.csv"
    awards_df = read_data(awards_file_path)
    #add a column prizeCount to player_teams on each year
    player_teams['prizeCount'] = 0
    #associate each player with each award and year (ex: player A won 2 awards in 2010, thus 2010 has 2, but 2011 is 0 (unless he wins again))
    for index, row in awards_df.iterrows():
        player_teams.loc[(player_teams['playerID'] == row['playerID']) & (player_teams['year'] == row['year']), 'prizeCount'] += 1
    
    return player_teams

def merge_with_team_data(df, teams_df):

    # temp_player_teams = df.merge(teams_df[['tmID', 'year', 'playoff', 'confID']], on=['tmID', 'year'], how='left')
    # temp_player_teams.drop('lgID', axis=1, inplace=True)

    # return temp_player_teams
    
    player_teams = df.merge(teams_df[['tmID', 'year', 'playoff', 'confID', 'firstRound','semis','finals']], on=['tmID', 'year'], how='left')
    #create column playoff_exit (0 if not in playoffs, 1 if lost in first round, 2 if lost in semis, 3 if lost in finals, 4 if won) get the info in the columns firstRound, semis, finals

    #drop unneeded columns
    player_teams.drop('lgID', axis=1, inplace=True)

    # Assuming you have a column 'year' to sort by
    player_teams = player_teams.sort_values(by=['playerID', 'year'])
    player_teams['career_year'] = player_teams.groupby('playerID').cumcount() + 1

    #swap all the W and L in firstRound, semis, finals columns for 1 and 0
    player_teams['firstRound'] = player_teams['firstRound'].replace(['W','L'], [1,0])
    player_teams['semis'] = player_teams['semis'].replace(['W','L'], [1,0])
    player_teams['finals'] = player_teams['finals'].replace(['W','L'], [1,0])

    
    player_teams['playoff_exit'] = 0
    # Iterate over the rows of the DataFrame
    for index, row in player_teams.iterrows():
        if row['playoff'] == 1:  # Proceed only if the team made the playoffs
            if row['firstRound'] == 0:
                player_teams.at[index, 'playoff_exit'] = 5
            elif row['semis'] == 0:  # Checks if 'playoff_exit' hasn't been set already
                player_teams.at[index, 'playoff_exit'] = 10
            elif row['finals'] == 0:
                player_teams.at[index, 'playoff_exit'] = 30
            elif row['finals'] == 1:
                player_teams.at[index, 'playoff_exit'] = 40
            # Add more conditions if there are more rounds or specific cases to handle

    #remove unneeded columns
    player_teams.drop(['firstRound','semis','finals'], axis=1, inplace=True)

    #add position column, get it from players.csv
    players_file_path = "../ac/basketballPlayoffs/players.csv"
    players_df = read_data(players_file_path)

    player_postion = players_df[['bioID', 'pos']]

    player_teams = player_teams.merge(player_postion, left_on='playerID', right_on='bioID', how='left')

    #remove bioID column
    player_teams.drop('bioID', axis=1, inplace=True)

    # Regular Season Percentages
    player_teams['ft%'] = compute_percentage(player_teams['ftMade'], player_teams['ftAttempted'])
    player_teams['fg%'] = compute_percentage(player_teams['fgMade'], player_teams['fgAttempted'])
    player_teams['three%'] = compute_percentage(player_teams['threeMade'], player_teams['threeAttempted'])
    player_teams['gs%'] = compute_percentage(player_teams['GS'], player_teams['GP'])*0.5

    # Playoffs Percentages
    player_teams['Postft%'] = compute_percentage(player_teams['PostftMade'], player_teams['PostftAttempted'])
    player_teams['Postfg%'] = compute_percentage(player_teams['PostfgMade'], player_teams['PostfgAttempted'])
    player_teams['Postthree%'] = compute_percentage(player_teams['PostthreeMade'], player_teams['PostthreeAttempted'])
    player_teams['Postgs%'] = compute_percentage(player_teams['PostGS'], player_teams['PostGP'])

    #effective field goal percentage
    player_teams['efg%'] = compute_percentage(player_teams['fgMade'] + 0.5 * player_teams['threeMade'], player_teams['fgAttempted']) * 2

    #true shooting percentage
    player_teams['ts%'] = compute_percentage(player_teams['points'], 2 * (player_teams['fgAttempted'] + 0.44 * player_teams['ftAttempted'])) * 2

    #per game stats
    #if pos contains G, then multiply by assists by 1.5, if contains C, then multiply by rebounds by 1.5, if contains F, then points by 1.5 (it may contain more than one letter ex: G-F )
    player_teams['ppg'] = compute_percentage(player_teams['points'], player_teams['GP'])
    player_teams['apg'] = compute_percentage(player_teams['assists'], player_teams['GP'])
    player_teams['rpg'] = compute_percentage(player_teams['rebounds'], player_teams['GP'])
    player_teams['spg'] = compute_percentage(player_teams['steals'], player_teams['GP'])
    player_teams['bpg'] = compute_percentage(player_teams['blocks'], player_teams['GP'])

    # For guards or players with mixed positions including guard
    player_teams.loc[player_teams['pos'].str.contains('G'), 'apg'] *= 1.5
    player_teams.loc[player_teams['pos'].str.contains('G'), 'spg'] *= 1.5

    # For centers or players with mixed positions including center
    player_teams.loc[player_teams['pos'].str.contains('C'), 'rpg'] *= 1.5
    # Assuming 'bpg' is blocks per game and exists in your DataFrame
    player_teams.loc[player_teams['pos'].str.contains('C'), 'bpg'] *= 1.5

    # For forwards or players with mixed positions including forward
    player_teams.loc[player_teams['pos'].str.contains('F'), 'ppg'] *= 1.5


    #per 36 minutes stats
    player_teams['pp36'] = compute_percentage(player_teams['points'], player_teams['minutes'])*36

    #defensive prowess: Defensive Prowess PCA: Use 'steals', 'blocks', and 'dRebounds' to create a 'Defensive Impact' principal component. Combine 'PF' (personal fouls) and 'turnovers' into a 'Defensive Discipline' component to represent careful play.
    player_teams['defensive_prowess'] = compute_percentage(player_teams['steals'] + player_teams['blocks'] + player_teams['dRebounds'], player_teams['GP'])*10
    player_teams['defensive_discipline'] = compute_percentage(player_teams['PF'] + player_teams['turnovers'], player_teams['GP'])*2
    
    #defensive rebounds percentage
    player_teams['drb%'] = compute_percentage(player_teams['dRebounds'], player_teams['rebounds'])

    #offensive rebounds percentage
    player_teams['orb%'] = compute_percentage(player_teams['oRebounds'], player_teams['rebounds'])

    #remove unneeded columns
    player_teams.drop(['pos','ftMade', 'ftAttempted', 'fgMade', 'fgAttempted', 'threeMade', 'threeAttempted', 'GS', 'GP', 'PostftMade', 'PostftAttempted', 'PostfgMade', 'PostfgAttempted', 'PostthreeMade', 'PostthreeAttempted', 'PostGS', 'PostGP', 'PF', 'turnovers', 'dRebounds', 'steals', 'blocks', 'PostPoints', 'PostRebounds', 'PostoRebounds','PostdRebounds','PostSteals','PostAssists', 'minutes', 'points', 'assists','dq','oRebounds','rebounds','PostMinutes','PostBlocks','PostTurnovers','PostPF','PostDQ'], axis=1, inplace=True)

    player_teams = aggregate_awards_counts(player_teams)

    #multiply by 1.5 the awards count
    player_teams['prizeCount']

    return player_teams

def get_columns_to_remove():
    return ['tmID', 'playerID','playoff','confID']

def players_awards(df):
    award_path = "basketballPlayoffs/awards_players.csv"
    players_awards = read_data(award_path)
    merged_df = df.merge(players_awards.groupby(['playerID', 'year'])['award'].count().reset_index(),
                         on=['playerID', 'year'], how='left')
    
    merged_df.rename(columns={'award': 'awards_count'}, inplace=True)
    merged_df.fillna(0, inplace=True)
    return merged_df 


def train_and_evaluate(df, years, i, classifier):
    param_dists = {
        RandomForestClassifier: {
            'n_estimators': [25, 50, 100, 150],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [3, 6, 9],
            'max_leaf_nodes': [3, 6, 9],
        },
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        SVC: {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.01, 0.1, 1, 10],
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        DecisionTreeClassifier: {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 6, 9],
            'max_leaf_nodes': [3, 6, 9],
            'min_samples_split': [2, 3, 4]
        },
        MLPClassifier: {
            'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        },
        BaggingClassifier: {
            'n_estimators': [10, 20, 50, 100],
            'max_samples': [0.5, 1.0],
            'max_features': [0.5, 1.0],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
        },
        SGDClassifier: {
            'loss': ['log'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [1000, 2000, 3000]
        },
    }
    #df = players_awards(df)
    train_years_before = years[:i]
    train_years = years[1:i+1]  # Use all years up to year i for training
    test_year = years[i]     # Testing on year i+1 

    if i + 1 > len(years): 
        return None, None

    predict_year = years[i+1]  # Predicting for year i+2 

    train_before = df[df['year'].isin(train_years_before)]
    train = df[df['year'].isin(train_years)]
    test = df[df['year'] == test_year]
    actual = df[df['year'] == predict_year]  # Changed actual_year to predict_year

    test_player_ids = test['playerID'].copy()

    train_before_indices_to_remove = []
    train_indices_to_remove = []
    #check if a player have a entry in the train df with the next year
    for index, row in train_before.iterrows():
        player_id = row['playerID']
        year = row['year'] + 1 
        if row['stint'] > 1:
            train_before_indices_to_remove.append(index)
        elif not (((train['playerID'] == player_id) & (train['year'] == year)).any()):
            data_to_add = {
                'playerID': player_id, 
                'year': year,        
            }
            new_row = pd.DataFrame(data_to_add, index=[0])
            train = pd.concat([train, new_row], ignore_index=True)
       

    train_before = train_before.drop(train_before_indices_to_remove)
    for index, row in train.iterrows():
        player_id = row['playerID']
        year = row['year'] - 1 
        if row['stint'] > 1:
            train_indices_to_remove.append(index)
        elif not (((train_before['playerID'] == player_id) & (train_before['year'] == year)).any()):
            data_to_add = {
                'playerID': player_id, 
                'year': year,        
            }
            new_row = pd.DataFrame(data_to_add, index=[0])
            train_before = pd.concat([train_before , new_row], ignore_index=True)
        
    train = train.drop(train_indices_to_remove)
    train.fillna(0, inplace=True)
    train.sort_values(by='playerID', inplace=True)
    train_before.fillna(0, inplace=True)
    train_before.sort_values(by='playerID', inplace=True)
    remove_columns = get_columns_to_remove()

    X_train = train_before.drop(remove_columns, axis=1)
    X_test = test.drop(remove_columns, axis=1)
    y_train = train['playoff']

    param_dist = param_dists.get(type(classifier), {})

    # Get sample weights
    sample_weights = get_sample_weights(train)

    #clf = GridSearchCV(classifier, param_dist, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    #clf = RandomizedSearchCV(classifier, param_dist, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    #clf = BayesSearchCV(classifier, param_dist, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, n_iter=10)

    clf = classifier

    if type(classifier) == KNeighborsClassifier or type(classifier) == SVC or type(classifier) == MLPClassifier or type(classifier) == LogisticRegression or type(classifier) == LGBMClassifier:
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train ,sample_weight=sample_weights)

    #classifier = clf.best_estimator_
    
    if type(classifier) == SGDClassifier:
        proba = classifier.predict(X_test)
    else:
        proba = classifier.predict_proba(X_test)[:, 1]

    test_proba_with_ids = pd.DataFrame({'playerID': test['playerID'], 'probability': proba})

    # Initialize team_avg_predictions with all teams from the actual_year
    team_avg_predictions = {tmID: {'probs': [], 'confID': confID} for tmID, confID in zip(actual['tmID'], actual['confID'])}

    # Get actual player IDs to filter the test set
    actual_players = actual['playerID'].unique()

    #define default probability as the average probability of all players in the test set
    default_probability = test_proba_with_ids['probability'].mean()

    # Update the probabilities for each team based on the actual year players
    for _, row in test_proba_with_ids.iterrows():
        player_id = row['playerID']
        if player_id in actual_players:
            #stint must be <= 1 to be considered
            if test[test['playerID'] == player_id]['stint'].max() > 1:
                continue    

            prob = row['probability']
            # Get team ID and confID from the actual DataFrame
            team_id = actual.loc[actual['playerID'] == player_id, 'tmID'].iloc[0]
            confID = actual.loc[actual['playerID'] == player_id, 'confID'].iloc[0]
            team_avg_predictions[team_id]['probs'].append(prob)
            # Ensure the confID is set for the team
            team_avg_predictions[team_id]['confID'] = confID

    # Calculate the average probability for teams that have players from the test set
    for tmID, data in team_avg_predictions.items():
        if data['probs']:  # If there are probabilities listed, calculate the average
            print(f"Team {tmID} has {len(data['probs'])} players in the test set.")
            data['avg_prob'] = sum(data['probs']) / len(data['probs'])
        else:  # For teams with no players from the test set, decide how to handle
            print(f"Team {tmID} has no players in the test set.")
            data['avg_prob'] = default_probability

    # Select top 4 teams from each conference
    east_teams = [(tmID, sum(data["probs"]) / len(data["probs"])) for tmID, data in team_avg_predictions.items() if data["confID"] == 'EA']
    west_teams = [(tmID, sum(data["probs"]) / len(data["probs"])) for tmID, data in team_avg_predictions.items() if data["confID"] == 'WE']

    all_teams = east_teams + west_teams

    top_east_teams = sorted(east_teams, key=lambda x: x[1], reverse=True)[:4]
    top_west_teams = sorted(west_teams, key=lambda x: x[1], reverse=True)[:4]

    top_teams = top_east_teams + top_west_teams

    team_results = []
    for tmID, avg_prob in all_teams:
        actual_playoff = actual[actual['tmID'] == tmID]['playoff'].iloc[0] if tmID in actual['tmID'].values else 0
        predicted_playoff = 1 if (tmID, avg_prob) in top_teams else 0
        team_results.append({
            'tmID': tmID,
            'Predicted': predicted_playoff,
            'Probability': avg_prob,
            'Actual': actual_playoff,
            'Year': actual['year'].iloc[0],
            'Classifier': classifier.__class__.__name__
        })

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

def plot_accuracy_over_time(prediction_data):
    unique_years = sorted(set(entry['Year'] for entry in prediction_data))
    classifiers = sorted(set(entry['Classifier'] for entry in prediction_data))

    # Prepare data for plotting
    accuracy_data = {classifier: [] for classifier in classifiers}

    # Collect accuracy for each classifier by year
    for year in unique_years:
        for classifier in classifiers:
            year_data = [entry for entry in prediction_data if entry['Year'] == year and entry['Classifier'] == classifier]
            correct_predictions = sum(entry['Predicted'] == entry['Actual'] for entry in year_data)
            accuracy = correct_predictions / len(year_data) if year_data else 0
            print(f"Year: {year}, Classifier: {classifier}, Accuracy: {accuracy:.2f}, Num Predictions: {len(year_data)}, Num Correct: {correct_predictions}")
            accuracy_data[classifier].append(accuracy)
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    for classifier, accuracies in accuracy_data.items():
        ax.plot(unique_years, accuracies, label=classifier, marker='x')

    ax.set_xlabel('Year')
    ax.set_ylabel('Accuracy')
    #make an x at each point
    ax.set_title('Classifier Accuracy Over Time')
    ax.legend()

    plt.tight_layout()
    plt.show()

def train_model():
    warnings.filterwarnings('ignore', message="is_sparse is deprecated and will be removed in a future version.")

    data_file_path = "basketballPlayoffs/players_teams.csv"
    df = read_data(data_file_path)

    teams_df = get_teams_data()
    if df is not None and teams_df is not None:
        df = merge_with_team_data(df, teams_df)
        
        years = df['year'].unique()
        years.sort()

        classifiers = {
            #"RandomForest": RandomForestClassifier(),
            #"KNN": KNeighborsClassifier(n_neighbors=3),
            #"SVM": SVC(probability=True), # Enable probability estimates
            "Bagging": BaggingClassifier(),
            #"DecisionTree": DecisionTreeClassifier(),
            #"MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), # Multi-layer Perceptron classifier
            #"LGBM": LGBMClassifier()
            "SGDClassifier": SGDClassifier()
        }

        results_dict = {}
        prediction_data = []

        for classifier_name, classifier in classifiers.items():
            results = []
            for i in range(1, len(years) - 1):  # Adjust to ensure there's always a year to predict after the test year
                team_results, accuracy = train_and_evaluate(df, years, i, classifier)
                if team_results:  # Check if team_results is not None
                    prediction_data.extend(team_results)
                    results.append(accuracy)
                    results_dict[classifier_name] = results

        predictions_df = pd.DataFrame(prediction_data)
        predictions_df.to_csv('predictions_results-playersTeams.csv', index=False)

        # Plot the accuracy for each classifier over time
        plot_accuracy_over_time(prediction_data)

        # selector = input("Do you want to see the bar chart or the heatmap or line graph? (b/h/l): ")
        # if selector == 'b':
        #     plot_bar_chart(predictions_df)
        # elif selector == 'h':
        #     plot_heatmaps(predictions_df)
        # elif selector == 'l':
        #     plot_line_chart(predictions_df)
        # else:
        #     print("Invalid input. Please try again.")

if __name__ == "__main__":
    train_model()