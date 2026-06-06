import os
import pandas as pd

RAW_GAMES_PATH = "data/raw/games.csv"
PROCESSED_DIR = "data/processed"
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

games = pd.read_csv(RAW_GAMES_PATH)

games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
games["WIN"] = games["WL"].map({"W": 1, "L": 0})
games["IS_HOME"] = games["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

games["OPPONENT"] = games["MATCHUP"].apply(
    lambda x: x.split("vs. ")[1] if "vs." in x else x.split("@ ")[1]
)

games = games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

games["OPP_PTS"] = games.groupby("GAME_ID")["PTS"].transform(
    lambda x: x.iloc[::-1].values if len(x) == 2 else [None] * len(x)
)

games["POINT_DIFF"] = games["PTS"] - games["OPP_PTS"]

games["AVG_POINTS_FOR"] = (
    games.groupby("TEAM_ABBREVIATION")["PTS"]
    .transform(lambda x: x.shift().expanding().mean())
)

games["WIN_RATE"] = (
    games.groupby("TEAM_ABBREVIATION")["WIN"]
    .transform(lambda x: x.shift().expanding().mean())
)

games["LAST_10_WIN_RATE"] = (
    games.groupby("TEAM_ABBREVIATION")["WIN"]
    .transform(lambda x: x.shift().rolling(10, min_periods=3).mean())
)

games["LAST_10_AVG_POINTS"] = (
    games.groupby("TEAM_ABBREVIATION")["PTS"]
    .transform(lambda x: x.shift().rolling(10, min_periods=3).mean())
)

games["AVG_POINT_DIFF"] = (
    games.groupby("TEAM_ABBREVIATION")["POINT_DIFF"]
    .transform(lambda x: x.shift().expanding().mean())
)

features = games.dropna(
    subset=[
        "AVG_POINTS_FOR",
        "WIN_RATE",
        "LAST_10_WIN_RATE",
        "LAST_10_AVG_POINTS",
        "AVG_POINT_DIFF",
    ]
)

features.to_csv(FEATURES_PATH, index=False)

print(f"Saved features to {FEATURES_PATH}")
print(features.head())