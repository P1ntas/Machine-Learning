from time import sleep
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
#joblib is used to save the model
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#add selectkbest
from sklearn.feature_selection import SelectKBest, f_classif

previous_year_predictions = {}

param_dists = {
        RandomForestClassifier: { #for accuracy, low computational cost
            'n_estimators': [20,40],
            'criterion': ['gini', 'entropy'],
            'random_state': [16, 256],
            'max_depth': [6],
            'max_leaf_nodes': [9],
            'min_samples_split': [4],
            'min_samples_leaf': [2],
        },
        KNeighborsClassifier: { #for accuracy
            'n_neighbors': [30,50],
            'weights': ['uniform', 'distance'],
            'leaf_size': [12,28,40],
            'metric': ['chebyshev', 'hamming'],
        },
        SVC: {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['rbf'],
            'shrinking': [True, False],
            'decision_function_shape': ['ovo', 'ovr'],
        },
        DecisionTreeClassifier: { #for accuracy (the choice is binary, but the criterion is entropy)
            'criterion': ['gini', 'entropy'],
            'max_depth': [ 6, 9],
            'max_leaf_nodes': [9],
            'min_samples_split': [4]
        },
        MLPClassifier: { #for accuracy
            'hidden_layer_sizes': [(50,100,50)],
            'activation': ['tanh', 'relu'],
        },
        BaggingClassifier: {
            'n_estimators': [10, 20],
            'max_samples': [0.5, 1.0],
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'warm_start': [True, False],
        },
        SGDClassifier: {    #for accuracy
            'loss': ['hinge', 'log', 'perceptron'],
            'alpha': [0.0001, 0.1],
            'learning_rate': ['constant', 'adaptive'],
            'eta0': [0.01, 10],
            'power_t': [0.1, 5],
        },
        LGBMClassifier: {  #for accuracy, low computational cost
            'num_leaves': [10, 100],
            'max_depth': [3, 10],
            'silent': [True, False],
            'importance_type': ['split'],
        },
        LogisticRegression: {  # for accuracy
            'penalty': ['l2'],  # 'l1' and 'elasticnet' might require the 'saga' solver, which is slower
            'C': [0.1, 1, 10],  # fewer values, spread across orders of magnitude
            'solver': ['lbfgs'],  # only include solvers that work well with 'l2'
            'max_iter': [500]  # reduced max iterations
        }
    }


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
    # Initialize all weights to 1
    weights = pd.Series(1, index=train.index)
    
    # Increase weight for players that reached at least the semis the previous year
    
    # # Increase weight for players with avgAttend > 10000, but not if they already have increased weight
    # weights[(train['avgAttend'] > 10000) & (weights == 1)] = 2

    weights[(train['playoff_exit'] > 5) & (weights == 1)] = 2
    
    # Increase weight if pos = C or F-C, and rpg > 8, but not if they already have increased weight
    weights[(train['pos'].isin([3, 4])) & (train['rpg'] > 8) & (weights == 1)] = 2

    # Increase weight if pos = G or G-F, and apg > 8, but not if they already have increased weight
    weights[(train['pos'].isin([1, 5])) & (train['apg'] > 7) & (weights == 1)] = 2
    
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
    
    player_teams = df.merge(teams_df[['tmID', 'year', 'playoff', 'confID', 'firstRound','semis','finals']], on=['tmID', 'year'], how='left')
    #create column playoff_exit (0 if not in playoffs, 1 if lost in first round, 2 if lost in semis, 3 if lost in finals, 4 if won) get the info in the columns firstRound, semis, finals

    # #for each player, get the avg attendance of their tmID in that year, get the column attend in teams.csv
    # player_teams = player_teams.merge(teams_df[['tmID', 'year', 'attend']], on=['tmID', 'year'], how='left')

    # player_teams['homeW'] = player_teams.merge(teams_df[['tmID', 'year', 'homeW']], on=['tmID', 'year'], how='left')['homeW']
    # player_teams['homeL'] = player_teams.merge(teams_df[['tmID', 'year', 'homeL']], on=['tmID', 'year'], how='left')['homeL']

    # player_teams['homeGames'] = player_teams['homeW'] + player_teams['homeL']

    #player_teams['avgAttend'] = player_teams['attend']/player_teams['homeGames']

    # #drop unneeded columns
    # player_teams.drop(['attend','homeW','homeL','homeGames'], axis=1, inplace=True)

    # print(player_teams.head(10)

    #get the players height and weight from players.csv
    players_file_path = "basketballPlayoffs/players.csv"
    players_df = read_data(players_file_path)
    print(players_df['bioID'].head(10))

    #merge with bioID and playerID
    player_bio = players_df[['bioID', 'height', 'weight']]
    player_teams = player_teams.merge(player_bio, left_on='playerID', right_on='bioID', how='left')

    #remove players with weight < 50 and height < 50
    player_teams = player_teams[player_teams['weight'] > 50]
    player_teams = player_teams[player_teams['height'] > 50]

    #drop weight and height columns
    player_teams.drop(['weight','height'], axis=1, inplace=True)

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
                player_teams.at[index, 'playoff_exit'] = 1
            elif row['semis'] == 0:  # Checks if 'playoff_exit' hasn't been set already
                player_teams.at[index, 'playoff_exit'] = 2
            elif row['finals'] == 0:
                player_teams.at[index, 'playoff_exit'] = 6
            elif row['finals'] == 1:
                player_teams.at[index, 'playoff_exit'] = 8
            # Add more conditions if there are more rounds or specific cases to handle

    #remove unneeded columns
    player_teams.drop(['firstRound','semis','finals'], axis=1, inplace=True)

    #add position column, get it from players.csv
    players_file_path = "../ac/basketballPlayoffs/players.csv"
    players_df = read_data(players_file_path)

    player_postion = players_df[['bioID', 'pos']]

    player_teams = player_teams.merge(player_postion, left_on='playerID', right_on='bioID', how='left')

    #remove bioID column
    #player_teams.drop('bioID', axis=1, inplace=True)

    #replace pos with numbers G=1, F=2, C=3, C-F/F-C=4, G-F/F-G=5
    player_teams['pos'] = player_teams['pos'].replace(['G','F','C','C-F','F-C','G-F','F-G'], [1,2,3,4,4,5,5])

    # # Regular Season Percentages
    #player_teams['ft%'] = compute_percentage(player_teams['ftMade'], player_teams['ftAttempted'])
    # player_teams['fg%'] = compute_percentage(player_teams['fgMade'], player_teams['fgAttempted'])
    #player_teams['three%'] = compute_percentage(player_teams['threeMade'], player_teams['threeAttempted'])*10
    # player_teams['gs%'] = compute_percentage(player_teams['GS'], player_teams['GP'])*0.5

    # # Playoffs Percentages
    #player_teams['Postft%'] = compute_percentage(player_teams['PostftMade'], player_teams['PostftAttempted'])
    # player_teams['Postfg%'] = compute_percentage(player_teams['PostfgMade'], player_teams['PostfgAttempted'])
    # player_teams['Postthree%'] = compute_percentage(player_teams['PostthreeMade'], player_teams['PostthreeAttempted'])
    # player_teams['Postgs%'] = compute_percentage(player_teams['PostGS'], player_teams['PostGP'])

    #effective field goal percentage
    player_teams['efg%'] = compute_percentage(player_teams['fgMade'] + 0.5 * player_teams['threeMade'], player_teams['fgAttempted'])

    #post effective field goal percentage
    #player_teams['Postefg%'] = compute_percentage(player_teams['PostfgMade'] + 0.5 * player_teams['PostthreeMade'], player_teams['PostfgAttempted'])

    #true shooting percentage
    player_teams['ts%'] = compute_percentage(player_teams['points'], 2 * (player_teams['fgAttempted'] + 0.44 * player_teams['ftAttempted'])) * 2

    max_efg_by_year = player_teams.groupby('year')['efg%'].transform('max')
    max_ts_by_year = player_teams.groupby('year')['ts%'].transform('max')
    min_efg_by_year = player_teams.groupby('year')['efg%'].transform('min')
    min_ts_by_year = player_teams.groupby('year')['ts%'].transform('min')

    # Apply the normalization within each year
    player_teams['efg%'] = (player_teams['efg%'] - min_efg_by_year) / (max_efg_by_year - min_efg_by_year)
    player_teams['ts%'] = (player_teams['ts%'] - min_ts_by_year) / (max_ts_by_year - min_ts_by_year)

    #drop efg and ts
    #player_teams.drop(['efg%','ts%'], axis=1, inplace=True)

    player_teams['ppg'] = round(player_teams['points']/player_teams['GP'],2)
    player_teams['rpg'] = round(player_teams['rebounds']/player_teams['GP'],2)
    player_teams['apg'] = round(player_teams['assists']/player_teams['GP'],2)
    player_teams['spg'] = round(player_teams['steals']/player_teams['GP'],2)
    player_teams['bpg'] = round(player_teams['blocks']/player_teams['GP'],2)

    #efficiency
    player_teams['eff'] = player_teams['ppg'] + player_teams['rpg'] + player_teams['apg'] + player_teams['spg'] + player_teams['bpg'] - (player_teams['fgAttempted'] - player_teams['fgMade']) - (player_teams['ftAttempted'] - player_teams['ftMade']) - player_teams['turnovers']

    # Calculate the min and max of `eff` for each year
    min_eff_by_year = player_teams.groupby('year')['eff'].transform('min')
    max_eff_by_year = player_teams.groupby('year')['eff'].transform('max')

    # Apply the normalization within each year
    player_teams['eff'] = (player_teams['eff'] - min_eff_by_year) / (max_eff_by_year - min_eff_by_year)


    #drop shooting percentages
    #player_teams.drop(['ft%','fg%','three%','gs%','Postft%','Postfg%','Postthree%','Postgs%'], axis=1, inplace=True)


    # # For guards or players with mixed positions including guard
    # player_teams.loc[player_teams['pos'].str.contains('G'), 'apg'] *= 1.5
    # player_teams.loc[player_teams['pos'].str.contains('G'), 'spg'] *= 1.5

    # # For centers or players with mixed positions including center
    # player_teams.loc[player_teams['pos'].str.contains('C'), 'rpg'] *= 1.5
    # # Assuming 'bpg' is blocks per game and exists in your DataFrame
    # player_teams.loc[player_teams['pos'].str.contains('C'), 'bpg'] *= 1.5

    # # For forwards or players with mixed positions including forward
    # player_teams.loc[player_teams['pos'].str.contains('F'), 'ppg'] *= 1.5


    #per 36 minutes stats
    #player_teams['pp36'] = compute_percentage(player_teams['points'], player_teams['minutes'])*36

    #defensive prowess: Defensive Prowess PCA: Use 'steals', 'blocks', and 'dRebounds' to create a 'Defensive Impact' principal component. Combine 'PF' (personal fouls) and 'turnovers' into a 'Defensive Discipline' component to represent careful play.
    player_teams['defensive_prowess'] = compute_percentage(player_teams['steals'] + player_teams['blocks'] + player_teams['dRebounds'], player_teams['GP'])*1
    player_teams['defensive_discipline'] = compute_percentage(player_teams['PF'] + player_teams['turnovers'], player_teams['GP'])*1
    
    #normalize defensive prowess and defensive discipline
    max_defensive_prowess_by_year = player_teams.groupby('year')['defensive_prowess'].transform('max')
    max_defensive_discipline_by_year = player_teams.groupby('year')['defensive_discipline'].transform('max')
    min_defensive_prowess_by_year = player_teams.groupby('year')['defensive_prowess'].transform('min')
    min_defensive_discipline_by_year = player_teams.groupby('year')['defensive_discipline'].transform('min')

    # Apply the normalization within each year
    player_teams['defensive_prowess'] = (player_teams['defensive_prowess'] - min_defensive_prowess_by_year) / (max_defensive_prowess_by_year - min_defensive_prowess_by_year)
    player_teams['defensive_discipline'] = (player_teams['defensive_discipline'] - min_defensive_discipline_by_year) / (max_defensive_discipline_by_year - min_defensive_discipline_by_year)
    # player_teams['drb%'] = compute_percentage(player_teams['dRebounds'], player_teams['rebounds'])

    # #offensive rebounds percentage
    #player_teams['orb%'] = compute_percentage(player_teams['oRebounds'], player_teams['rebounds'])
    #drop defensive prowess and defensive discipline
    #player_teams.drop(['defensive_prowess','defensive_discipline'], axis=1, inplace=True)

    #remove unneeded columns
    #"player_teams.drop(['pos'], axis=1, inplace=True)
    player_teams.drop(['ftMade', 'ftAttempted', 'fgMade', 'fgAttempted', 'threeMade', 'threeAttempted', 'GS', 'GP', 'PostftMade', 'PostftAttempted', 'PostfgMade', 'PostfgAttempted', 'PostthreeMade', 'PostthreeAttempted', 'PostGS', 'PostGP', 'PF', 'turnovers', 'dRebounds', 'steals', 'blocks', 'PostPoints', 'PostRebounds', 'PostoRebounds','PostdRebounds','PostSteals','PostAssists', 'minutes', 'points', 'assists','dq','oRebounds','rebounds','PostMinutes','PostBlocks','PostTurnovers','PostPF','PostDQ'], axis=1, inplace=True)

    player_teams.drop(['spg','bpg', 'ppg'], axis=1, inplace=True)

    player_teams = aggregate_awards_counts(player_teams)
    player_teams.drop(['bioID_x','bioID_y'], axis=1, inplace=True)
    #print(player_teams.head(10))
    #store in csv the player_teams dataframe
    player_teams.to_csv('new_players_teams.csv', index=False)

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

    # Create the pipeline
    numeric_features = ['career_year', 'efg%', 'ts%', 'rpg', 'apg', 'defensive_prowess', 'defensive_discipline', 'eff']
    categorical_features = ['pos']
    #do not consider the playerID column and stint

    # Create the preprocessing and feature selection pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Build the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k=5)),
        ('classifier', classifier)])
    
    # Adjust param_dist to include pipeline step names and feature selection hyperparameter 'k'
    param_dist = {**{
        'feature_selection__k': [5, 10],  # Example hyperparameters for 'k'
        'classifier__n_estimators': [20, 40],
        'classifier__max_depth': [6, None],
        # Add other parameters and respective ranges here
    }}
    
    # Initialize GridSearchCV with the pipeline
    clf = GridSearchCV(pipeline, param_dist, cv=5, scoring='accuracy', refit=True, verbose=2)

    param_dist = param_dists.get(type(classifier), {})

    # Get sample weights
    sample_weights = get_sample_weights(train)

    # Prepare training and testing data
    remove_columns = get_columns_to_remove()
    X_train = train_before.drop(remove_columns, axis=1)
    X_test = test.drop(remove_columns, axis=1)
    y_train = train['playoff']

    # Fit the model
    #if KNN dont use sample weights
    if classifier.__class__.__name__ == 'KNeighborsClassifier':
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train, classifier__sample_weight=get_sample_weights(train))  # Adjust for your sample weights

    print(f"Best parameters: {clf.best_params_}")
    print(f"Training accuracy: {clf.score(X_train, y_train)}")
    support_mask = clf.best_estimator_.named_steps['feature_selection'].get_support()
    # Retrieve feature names from the ColumnTransformer
    all_features = clf.best_estimator_.named_steps['preprocessor'].transformers_[0][2] + \
                   list(clf.best_estimator_.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())

    # Map support mask to feature names
    selected_features = [all_features[i] for i, val in enumerate(support_mask) if val]
    print(f"Selected features: {selected_features}")

    #print the best k features
    #print(f"Best features: {X_train.columns[clf.best_estimator_.named_steps['feature_selection'].get_support()]}")

    # Predict probabilities or class labels depending on classifier type
    if hasattr(clf.best_estimator_.named_steps['classifier'], 'predict_proba'):
        proba = clf.predict_proba(X_test)[:, 1]
    else:
        proba = clf.predict(X_test)

    test_proba_with_ids = pd.DataFrame({'playerID': test['playerID'], 'probability': proba})

    # Initialize team_avg_predictions with all teams from the actual_year
    team_avg_predictions = {tmID: {'probs': [], 'confID': confID, 'players':[]} for tmID, confID in zip(actual['tmID'], actual['confID'])}

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
            team_avg_predictions[team_id]['players'].append(player_id)
            team_avg_predictions[team_id]['confID'] = confID



    # Calculate the average probability for teams that have players from the test set
    for tmID, data in team_avg_predictions.items():
        if data['probs']:  # If there are probabilities listed, calculate the average
            top_3 = sorted(zip(data['players'], data['probs']), key=lambda x: x[1], reverse=True)
            data['avg_prob'] = sum([prob for _, prob in top_3]) / len(top_3)
        else:  # For teams with no players from the test set, decide how to handle
            print(f"Team {tmID} has no players in the test set.")
            data['avg_prob'] = default_probability

    #previous_year_predictions = team_avg_predictions.copy()

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
    
    player_results = []
    for index, row in test_proba_with_ids.iterrows():
        player_id = row['playerID']
        prob = row['probability']
        actual_playoff = actual[actual['playerID'] == player_id]['playoff'].iloc[0] if player_id in actual['playerID'].values else 0
        predicted_playoff = 1 if prob > 0.5 else 0
        player_results.append({
            'playerID': player_id,
            'Predicted': predicted_playoff,
            'Probability': prob,
            'Actual': actual_playoff,
            'Year': actual['year'].iloc[0],
            'Classifier': classifier.__class__.__name__
        })

    # Calculate accuracy
    accuracy = sum([1 for result in team_results if result['Predicted'] == result['Actual']]) / 8

    return team_results, accuracy, player_results


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

def plot_team(prediction_data):
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

def plot_player(prediction_data):
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
            "RandomForest": RandomForestClassifier(),
            #"KNN": KNeighborsClassifier(),
            #"SVM": SVC(probability=True), # Enable probability estimates
            #"Bagging": BaggingClassifier(),
            #"DecisionTree": DecisionTreeClassifier(random_state=15),
            #"MLP": MLPClassifier(), # Multi-layer Perceptron classifier
            "LGBM": LGBMClassifier(),
            #"SGDClassifier": SGDClassifier(),
            #"LogisticRegression": LogisticRegression(), # Logistic Regression (doesn't work with sparse data)

        }

        results_dict = {}
        team_data = []
        players_data = []

        for classifier_name, classifier in classifiers.items():
            results = []
            for i in range(1, len(years) - 1):  # Adjust to ensure there's always a year to predict after the test year
                team_results, accuracy, player_results = train_and_evaluate(df, years, i, classifier)
                if team_results:  # Check if team_results is not None
                    team_data.extend(team_results)
                    players_data.extend(player_results)
                    results.append(accuracy)
                    results_dict[classifier_name] = results

        predictions_df = pd.DataFrame(team_data)
        predictions_df.to_csv('predictions_results-playersTeams.csv', index=False)

        # Plot the accuracy for each classifier over time
        plot_team(team_data)
        plot_player(players_data)

if __name__ == "__main__":
    train_model()