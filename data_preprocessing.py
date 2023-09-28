import pandas as pd

def merge_data():
    merged_df = pd.merge(coaches, teams, on=['year', 'tmID'])
    merged_df = pd.merge(teams_post, merged_df, on=['year', 'tmID'])
    players_df = pd.merge(players, awards_players, left_on=['bioID'], right_on=['playerID'])
    players_df.pop('playerID')
    players_df = pd.merge(players, players_teams, left_on=['bioID'], right_on=['playerID'])
    merged_df = pd.merge(players_df, merged_df, on=['year', 'tmID'])

    return merged_df

def clean_teams():
    global teams
    teams.pop("lgID")
    teams.pop("divID")
    teams.pop("seeded")
    teams.pop("arena")
    teams.pop("o_reb")
    teams.pop("d_reb")
    return

def clean_teams_post():
    global teams_post
    teams_post.pop("lgID")
    return

def clean_series_post():
    global series_post
    series_post.pop("lgIDWinner")
    series_post.pop("lgIDLoser")
    return


def clean_coaches():
    global coaches
    # Handle missing or incomplete data
    coaches.pop("lgID")

    return 

def clean_players():
    global players
    players = players[(players['deathDate'] == '0000-00-00') & (players['birthDate'] != '0000-00-00')]
    players.pop('deathDate')

    players = players[players['bioID'].isin(players_teams['playerID'])]

    return

def clean_players_teams(players_teams):
    players_teams = players_teams[players_teams['playerID'].isin(players['bioID'])]
    players_teams.pop('lgID')
    return

def clean_awards_players(awards_players):
   
    # Handle missing or incomplete data
    awards_players.pop("lgID")

    awards_players = awards_players.groupby('playerID')['award'].count().reset_index()
    awards_players.columns = ['playerID', 'awards_count']
    return awards_players

def clean_data(awards_players):
    # Handle missing or incomplete data
    awards_players = clean_awards_players(awards_players)
    clean_coaches()
    clean_players()
    clean_teams()
    clean_teams_post()
    clean_series_post()
    return awards_players

def feature_engineering():
    # Generate new features that can improve the prediction
    return

if __name__ == "__main__":
    # Load Data

    awards_players = pd.read_csv("basketballPlayoffs/awards_players.csv")
    coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
    players = pd.read_csv("basketballPlayoffs/players.csv")
    players_teams = pd.read_csv("basketballPlayoffs/players_teams.csv")
    teams = pd.read_csv("basketballPlayoffs/teams.csv")
    series_post = pd.read_csv("basketballPlayoffs/series_post.csv")
    teams_post = pd.read_csv("basketballPlayoffs/teams_post.csv")

    # Data Cleaning
    awards_players = clean_data(awards_players)
    # Merge Data
    merged_df = merge_data()
    # Feature Engineering
    feature_engineering()

    merged_df.to_csv("preprocessed_data.csv", index=False)
    
    # Save the cleaned, merged, and engineered data to a new CSV
    # merged_and_cleaned_data.to_csv("preprocessed_data.csv", index=False)
