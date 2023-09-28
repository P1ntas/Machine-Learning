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
    # Merge the datasets into a single DataFrame using keys like 'year', 'tmID', etc.
    return

def clean_coaches():
    # Handle missing or incomplete data
    df = pd.read_csv("basketballPlayoffs/coaches.csv")
    df.pop("lgID")


    print(df.head(15))
    return df

def clean_awards_players():
    # Handle missing or incomplete data
    df = pd.read_csv("basketballPlayoffs/awards_players.csv")
    df.pop("lgID")

    df = df.groupby('playerID')['award'].count().reset_index()
    df.columns = ['playerID', 'awards_count']
    print(df.head(15))
    return df

def clean_data():
    # Handle missing or incomplete data
    #clean_awards_players()
    clean_coaches()
    return

def feature_engineering():
    # Generate new features that can improve the prediction
    return

if __name__ == "__main__":
    # Load Data
    load_data()
    
    # Merge Data
    merge_data()
    
    # Data Cleaning
    clean_data()
    
    # Feature Engineering
    feature_engineering()
    
    # Save the cleaned, merged, and engineered data to a new CSV
    # merged_and_cleaned_data.to_csv("preprocessed_data.csv", index=False)
