import pandas as pd

def merge_data():
    return

def clean_teams():
    teams.pop("lgID")
    teams.pop("divID")
    teams.pop("seeded")
    teams.pop("arena")
    teams.pop("o_reb")
    teams.pop("d_reb")
    return

def clean_teams_post():
    teams_post.pop("lgID")
    return

def clean_series_post():
    series_post.pop("lgIDWinner")
    series_post.pop("lgIDLoser")
    return

def clean_players():
    players = players[(players['deathDate'] == '0000-00-00') & (players['birthDate'] != '0000-00-00')]
    players = players.drop(columns=['deathDate'])

    players = players[players['bioID'].isin(players_teams['playerID'])]
    players = players.drop_duplicates(subset=['bioID'])

    players.to_csv('basketballPlayoffs/players.csv', index=False)
    return

def clean_awards_players():
    # Handle missing or incomplete data
    df = pd.read_csv("basketballPlayoffs/awards_players.csv")
    df.pop("lgID")

    df = df.groupby('playerID')['award'].count().reset_index()
    df.columns = ['playerID', 'awards_count']
    print(df.head(15))
    return df

def clean_data():
    clean_players()
    clean_awards_players()
    clean_teams()
    clean_teams_post()
    clean_series_post()
    return

def feature_engineering():
    # Generate new features that can improve the prediction
    return

if __name__ == "__main__":
    # Load Data
    awards_players = pd.read_csv("basketballPlayoffs/awards_players.csv")
    coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
    players = pd.read_csv("basketballPlayoffs/players.csv")
    players_teams = pd.read_csv("basketballPlayoffs/teams.csv")
    teams = pd.read_csv("basketballPlayoffs/teams.csv")
    series_post = pd.read_csv("basketballPlayoffs/series_post.csv")
    teams_post = pd.read_csv("basketballPlayoffs/teams_post.csv")
    
    clean_players(players, players_teams)
    # Merge Data
    merge_data()
    
    # Data Cleaning
    clean_data()
    
    # Feature Engineering
    feature_engineering()
    
    # Save the cleaned, merged, and engineered data to a new CSV
    # merged_and_cleaned_data.to_csv("preprocessed_data.csv", index=False)
