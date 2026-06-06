import os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

gamefinder = leaguegamefinder.LeagueGameFinder(league_id_nullable="00")
games = gamefinder.get_data_frames()[0]

games = games[
    [
        "SEASON_ID",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "PTS",
    ]
]

output_path = os.path.join(RAW_DIR, "games.csv")
games.to_csv(output_path, index=False)

print(f"Saved {len(games)} rows to {output_path}")
print(games.head())