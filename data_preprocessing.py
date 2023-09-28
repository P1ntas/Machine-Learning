import pandas as pd

def merge_data():
    merged_df = pd.merge(coaches, teams, on=['year', 'tmID'])
    merged_df = pd.merge(teams_post, merged_df, on=['year', 'tmID'])
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


def clean_coaches(coaches):
    # Handle missing or incomplete data
    coaches.pop("lgID")

    return 

def clean_players(players, players_teams):
    players = players[(players['deathDate'] == '0000-00-00') & (players['birthDate'] != '0000-00-00')]
    players = players.drop(columns=['deathDate'])

    players = players[players['bioID'].isin(players_teams['playerID'])]
    players = players.drop_duplicates(subset=['bioID'])

    players_teams = players_teams[players_teams['playerID'].isin(players['bioID'])]
    #remove lgID column
    players_teams = players_teams.drop(columns=['lgID'])
    #make a new df for player_season where players with stint > 0 are joined into one row if year is the same, the stats are summed, the rest of the players remain the same and the colomn stint is removed
    # Filter rows where stint > 0
    player_season = players_teams[players_teams['stint'] > 0]

    # Define columns that need to be grouped by and columns to be summed
    groupby_cols = ['playerID', 'year']
    sum_cols = ["GP", "GS", "minutes", "points", "oRebounds", "dRebounds", "rebounds", 
                "assists", "steals", "blocks", "turnovers", "PF", "fgAttempted", "fgMade", 
                "ftAttempted", "ftMade", "threeAttempted", "threeMade", "dq", "PostGP", 
                "PostGS", "PostMinutes", "PostPoints", "PostoRebounds", "PostdRebounds", 
                "PostRebounds", "PostAssists", "PostSteals", "PostBlocks", "PostTurnovers", 
                "PostPF", "PostfgAttempted", "PostfgMade", "PostftAttempted", "PostftMade", 
                "PostthreeAttempted", "PostthreeMade", "PostDQ"]

    # Group by 'playerID' and 'year', then sum the specific columns
    player_season = player_season.groupby(groupby_cols)[sum_cols].sum().reset_index()

    # Drop the 'stint' column
    player_season = player_season.drop(columns=['stint'])


    return

def clean_awards_players(award_players):
    # Handle missing or incomplete data
    award_players.pop("lgID")

    award_players = award_players.groupby('playerID')['award'].count().reset_index()
    award_players.columns = ['playerID', 'awards_count']
    return 

def clean_data():
    # Handle missing or incomplete data
    clean_awards_players(awards_players)
    clean_coaches(coaches)
    clean_players(players, players_teams)
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
    players_teams = pd.read_csv("basketballPlayoffs/players_teams.csv")
    teams = pd.read_csv("basketballPlayoffs/teams.csv")
    series_post = pd.read_csv("basketballPlayoffs/series_post.csv")
    teams_post = pd.read_csv("basketballPlayoffs/teams_post.csv")

    # Data Cleaning
    clean_data()
    
    # Merge Data
    merge_data()
    
    # Feature Engineering
    feature_engineering()
    
    # Save the cleaned, merged, and engineered data to a new CSV
    # merged_and_cleaned_data.to_csv("preprocessed_data.csv", index=False)
