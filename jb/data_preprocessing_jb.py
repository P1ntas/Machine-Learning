import pandas as pd
def merge_data():
    # Merging players and teams data based on 'year' and 'tmID'
    players_df = pd.merge(players, awards_players, left_on=['bioID'], right_on=['playerID'], how='outer')
    players_df.pop('playerID')
    players_df = pd.merge(players, players_teams, left_on=['bioID'], right_on=['playerID'], how='outer')
    merged_df = pd.merge(players_df, teams, on=['year', 'tmID'], how='outer')
    merged_df.pop('playerID')
    merged_df.pop('lgID')
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

def clean_players_teams():
    global players_teams
    players_teams = players_teams[players_teams['playerID'].isin(players['bioID'])]
    players_teams.pop('lgID')
    return

def clean_awards_players():
    global awards_players
   
    # Handle missing or incomplete data
    awards_players.pop("lgID")

    awards_players = awards_players.groupby('playerID')['award'].count().reset_index()
    awards_players.columns = ['playerID', 'awards_count']
    return awards_players

def clean_data():
    clean_awards_players()
    clean_coaches()
    clean_players()
    clean_teams()
    return

def feature_engineering():
    # Generate New Features to be used in the model
    merged_df['winning_percentage'] = merged_df['won'] / (merged_df['won'] + merged_df['lost'] * -1)
    merged_df['winning_percentage'] = merged_df['winning_percentage'].fillna(0)
    return

if __name__ == "__main__":
    # Load Data
    awards_players = pd.read_csv("basketballPlayoffs/awards_players.csv")
    coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
    players = pd.read_csv("basketballPlayoffs/players.csv")
    players_teams = pd.read_csv("basketballPlayoffs/players_teams.csv")
    teams = pd.read_csv("basketballPlayoffs/teams.csv")
    
    # Data Cleaning
    clean_data()
    # Merge Data
    merged_df = merge_data()
    # Feature Engineering
    feature_engineering()

    # Save the cleaned, merged, and engineered data to a new CSV
    merged_df.to_csv("preprocessed_data_jb.csv", index=False)