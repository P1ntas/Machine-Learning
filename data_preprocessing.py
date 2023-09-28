import pandas as pd

def load_data():
    # Load all your CSV files into Pandas DataFrame from basketballPlayoffs folder
    awards_players = pd.read_csv("basketballPlayoffs/awards_players.csv")
    coaches = pd.read_csv("basketballPlayoffs/coaches.csv")
    players = pd.read_csv("basketballPlayoffs/players.csv")
    players_teams = pd.read_csv("basketballPlayoffs/teams.csv")
    teams = pd.read_csv("basketballPlayoffs/teams.csv")
    series_post = pd.read_csv("basketballPlayoffs/series_post.csv")
    teams_post = pd.read_csv("basketballPlayoffs/teams_post.csv")
    return

def merge_data():
    teams.pop("lgID")
    teams.pop("divID")
    teams.pop("seeded")
    teams.pop("arena")
    teams.pop("o_reb")
    teams.pop("d_reb")
    teams_post.pop("lgID")
    series_post.pop("lgIDWinner")
    series_post.pop("lgIDLoser")
    return

def clean_data():
    print()
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
    
    # Merge Data
    merge_data()
    
    # Data Cleaning
    clean_data()
    
    # Feature Engineering
    feature_engineering()
    
    # Save the cleaned, merged, and engineered data to a new CSV
    # merged_and_cleaned_data.to_csv("preprocessed_data.csv", index=False)
