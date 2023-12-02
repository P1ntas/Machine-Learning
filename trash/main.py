import pandas as pd

file_names = ["awards_players.csv", "coaches.csv", "players_teams.csv", "players.csv", "series_post.csv", "teams_post.csv", "teams.csv"]

for file_name in file_names:
    try:
        df = pd.read_csv("data/" + file_name)
        print(f"First 5 lines of {file_name}:\n")
        print(df.head(5))
        print("\n" + "=" * 40 + "\n")
    except FileNotFoundError:
        print(f"File {file_name} not found.")
    except pd.errors.EmptyDataError:
        print(f"File {file_name} is empty.")
    except pd.errors.ParserError:
        print(f"Could not parse file {file_name}. Check if it's a valid CSV file.")

