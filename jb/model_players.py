import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
#import bagging classifier
from sklearn.ensemble import BaggingClassifier
#import sgdc classifier
from sklearn.linear_model import SGDClassifier
import statistics



warnings.filterwarnings("ignore")

#ignore all possible warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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
    teams_file_path = "../ac/data/teams.csv"
    teams_df = read_data(teams_file_path)
    if teams_df is not None:
        return pre_process_data(teams_df)
    return None

def get_sample_weights(train):
    max_year = train['year'].max()
    weights = train['year'].apply(lambda x: 2 if x == max_year else 1)
    return weights

def compute_percentage(numerator, denominator):
    return round(numerator.divide(denominator).where(denominator != 0, 0.0),2)

def aggregate_awards_counts(player_teams):
    #get from awards_players.csv
    awards_file_path = "../ac/data/awards_players.csv"
    awards_df = read_data(awards_file_path)
    #add a column prizeCount to player_teams on each year
    player_teams['prizeCount'] = 0
    #associate each player with each award and year (ex: player A won 2 awards in 2010, thus 2010 has 2, but 2011 is 0 (unless he wins again))
    for index, row in awards_df.iterrows():
        player_teams.loc[(player_teams['playerID'] == row['playerID']) & (player_teams['year'] == row['year']), 'prizeCount'] += 1
    
    return player_teams

def merge_with_team_data(df, teams_df):
    player_teams = df.merge(teams_df[['tmID', 'year', 'playoff', 'confID', 'firstRound','semis','finals']], on=['tmID', 'year'], how='left')
    #drop unneeded columns
    player_teams.drop('lgID', axis=1, inplace=True)

    # Assuming you have a column 'year' to sort by
    player_teams = player_teams.sort_values(by=['playerID', 'year'])
    player_teams['career_year'] = player_teams.groupby('playerID').cumcount() + 1

    #add position column, get it from players.csv
    players_file_path = "../ac/data/players.csv"
    players_df = read_data(players_file_path)

    #create column playoff_exit (0 if not in playoffs, 1 if lost in first round, 2 if lost in semis, 3 if lost in finals, 4 if won) get the info in the columns firstRound, semis, finals

    #swap all the W and L in firstRound, semis, finals columns for 1 and 0
    player_teams['firstRound'] = player_teams['firstRound'].replace(['W','L'], [1,0])
    player_teams['semis'] = player_teams['semis'].replace(['W','L'], [1,0])
    player_teams['finals'] = player_teams['finals'].replace(['W','L'], [1,0])

    
    player_teams['playoff_exit'] = 0
    # Iterate over the rows of the DataFrame
    for index, row in player_teams.iterrows():
        if row['playoff'] == 1:  # Proceed only if the team made the playoffs
            if row['firstRound'] == 0:
                player_teams.at[index, 'playoff_exit'] = 0
            elif row['semis'] == 0:  # Checks if 'playoff_exit' hasn't been set already
                player_teams.at[index, 'playoff_exit'] = 1
            elif row['finals'] == 0:
                player_teams.at[index, 'playoff_exit'] = 5
            elif row['finals'] == 1:
                player_teams.at[index, 'playoff_exit'] = 7
            # Add more conditions if there are more rounds or specific cases to handle

    #remove unneeded columns
    player_teams.drop(['firstRound','semis','finals'], axis=1, inplace=True)

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

    # Calculate the average probability using only the top 5 players
    for tmID, data in team_avg_predictions.items():
        if data['probs']:
            top_probs = sorted(data['probs'], reverse=True)  # Sort and pick top 6
            data['avg_prob'] = sum(top_probs) / len(top_probs)
        else:
            data['avg_prob'] = default_probability
    
    print("Default probability: ", default_probability)


    east_teams = [(tmID, data['avg_prob']) for tmID, data in team_avg_predictions.items() if data['confID'] == 'EA']
    west_teams = [(tmID, data['avg_prob']) for tmID, data in team_avg_predictions.items() if data['confID'] == 'WE']

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
            'Classifier': classifier_name
        })

    correct_predictions = sum(1 for result in team_results if result['Predicted'] == result['Actual'])
    total_predictions = len(team_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    return team_results, accuracy


def train_and_evaluate(df, years, classifier):
    cumulative_train = pd.DataFrame()
    results = []

    #  # Define base classifiers for stacking
    # base_classifiers = [
    #     ('rf', RandomForestClassifier(n_estimators=10, max_depth=3)),
    #     ('knn', KNeighborsClassifier(n_neighbors=5)),
    #     ('svc', SVC(C=1, gamma=0.01, kernel='rbf', probability=True)),
    #     ('dt', DecisionTreeClassifier(max_depth=3)),
    #     ('mlp', MLPClassifier(hidden_layer_sizes=(50,))),
    #     ('sgd', SGDClassifier(loss='log', max_iter=1000))
    #     # Add other base classifiers as needed
    # ]
    
    # # Define meta-learner
    # meta_learner = LogisticRegression()

    # stacking_clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_learner, cv=5)

    # Define parameter distributions for each classifier type
    param_dists = {
        RandomForestClassifier: {
            'n_estimators': [150],
            'max_features': ['sqrt', 'log2', None],
            'max_depth': [3, 9],
            'max_leaf_nodes': [9],
        },
        KNeighborsClassifier: {
            'n_neighbors': [3, 9, 11],
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

    for i in range(len(years) - 2):
        train_years = years[:i+1]
        test_year = years[i+1]
        predict_year = years[i+2]

        # # Adding new year data to cumulative train data
        cumulative_train = cumulative_train._append(df[df['year'].isin(train_years)], ignore_index=True)

        test = df[df['year'] == test_year]
        actual = df[df['year'] == predict_year]

        remove_columns = get_columns_to_remove()
        X_train = cumulative_train.drop(remove_columns, axis=1)
        y_train = cumulative_train['playoff']
        X_test = test.drop(remove_columns, axis=1)

        sample_weights = get_sample_weights(cumulative_train)

        # Get the parameter distribution based on the classifier type
        param_dist = param_dists.get(type(classifier), {})
        
        # Define the classifier

        clf = GridSearchCV(classifier, param_dist, cv=5, scoring='accuracy')
        #clf = RandomizedSearchCV(classifier, param_dist, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        #clf = BayesSearchCV(classifier, param_dist, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, n_iter=10)
        #clf = classifier


        if type(classifier) == KNeighborsClassifier or type(classifier) == SVC or type(classifier) == MLPClassifier or type(classifier) == LogisticRegression or type(classifier) == LGBMClassifier:
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train ,sample_weight=sample_weights)
        
        #Update classifier to be the best estimator
        classifier = clf.best_estimator_

        print("Best parameters set found on development set:")
        print(classifier)

        # Predict probabilities for the test set using the best estimator
        if type(classifier) == SGDClassifier:
            proba = classifier.predict(X_test)
        else:
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

    # Determine the number of classifiers and create a grid layout accordingly
    num_classifiers = len(classifiers)
    fig, axes = plt.subplots(num_classifiers, 1, figsize=(12, 6 * num_classifiers))

    # In case there is only one classifier, axes will not be an array, so we wrap it in a list
    if num_classifiers == 1:
        axes = [axes]

    for idx, classifier_name in enumerate(classifiers):
        ax = axes[idx]
        ax.axis('off')
        ax.axis('tight')

        table_data = [['Year', 'Actual Teams', 'Predicted Teams', 'Accuracy']]
        for year in unique_years:
            key = (classifier_name, year)
            if key in organized_data:
                data = organized_data[key]
                table_data.append([year, data['Actual'], data['Predicted'], data['Accuracy']])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.1, 0.3, 0.3, 0.1])
        #add more with in the rows
        table.auto_set_column_width(col=list(range(len(table_data[0]))))
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        ax.set_title(f'Comparison of Actual vs Predicted Playoff Teams ({classifier_name})', fontsize=12)

    # Adjust layout to prevent overlapping of tables
    plt.tight_layout()
    plt.show()


def train_model():
    data_file_path = "../ac/data/players_teams.csv"
    df = read_data(data_file_path)

    teams_df = get_teams_data()
    if df is not None and teams_df is not None:
        df = merge_with_team_data(df, teams_df)
        
        years = sorted(df['year'].unique())

        classifiers = {
            "RandomForest": RandomForestClassifier(),
            #"Bagging": BaggingClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            #"SVM": SVC(probability=True), # Enable probability estimates
            #"DecisionTree": DecisionTreeClassifier(),
            #"MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), # Multi-layer Perceptron classifier
            #"LGBM": LGBMClassifier(random_state=42),
            #"LogisticRegression": LogisticRegression(),
            #"SGDClassifier": SGDClassifier()
        }

        total_results = []

        for classifier_name, classifier in classifiers.items():
            logging.info(f"Training and evaluating with {classifier_name}")
            results = train_and_evaluate(df, years, classifier)

            total_results += results

            # Convert results to DataFrame and save or print as necessary
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'./jb/results/predictions_results_{classifier_name}.csv', index=False)

            # Print accuracy for each classifier or save to log
            logging.info(f"Completed training and evaluation for {classifier_name}")


    #plot_teams_comparison(total_results)
    #print(total_results)
    plot_accuracy_over_time(total_results)

if __name__ == "__main__":
    train_model()