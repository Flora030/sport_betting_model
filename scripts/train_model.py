import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURES_PATH = "data/processed/features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "nba_win_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(FEATURES_PATH)
data = data.dropna()

# Keep only rows where team is home
home_games = data[data["IS_HOME"] == 1].copy()
away_games = data[data["IS_HOME"] == 0].copy()

# Merge home row with away row from same game
matchups = home_games.merge(
    away_games,
    on="GAME_ID",
    suffixes=("_HOME", "_AWAY")
)

matchups["WIN"] = matchups["WIN_HOME"]

matchups["WIN_RATE_DIFF"] = matchups["WIN_RATE_HOME"] - matchups["WIN_RATE_AWAY"]
matchups["LAST_10_WIN_RATE_DIFF"] = matchups["LAST_10_WIN_RATE_HOME"] - matchups["LAST_10_WIN_RATE_AWAY"]
matchups["AVG_POINT_DIFF_DIFF"] = matchups["AVG_POINT_DIFF_HOME"] - matchups["AVG_POINT_DIFF_AWAY"]
matchups["AVG_POINTS_FOR_DIFF"] = matchups["AVG_POINTS_FOR_HOME"] - matchups["AVG_POINTS_FOR_AWAY"]
matchups["LAST_10_AVG_POINTS_DIFF"] = matchups["LAST_10_AVG_POINTS_HOME"] - matchups["LAST_10_AVG_POINTS_AWAY"]

feature_cols = [
    "WIN_RATE_DIFF",
    "LAST_10_WIN_RATE_DIFF",
    "AVG_POINT_DIFF_DIFF",
    "AVG_POINTS_FOR_DIFF",
    "LAST_10_AVG_POINTS_DIFF",
]

matchups = matchups.dropna(subset=feature_cols + ["WIN"])

X = matchups[feature_cols]
y = matchups["WIN"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

team_latest_features = (
    data.sort_values("GAME_DATE")
    .groupby("TEAM_ABBREVIATION")
    .tail(1)
    .reset_index(drop=True)
)

joblib.dump(
    {
        "model": model,
        "team_latest_features": team_latest_features,
        "feature_cols": feature_cols,
        "accuracy": accuracy,
        "model_type": "matchup_difference",
    },
    MODEL_PATH,
)

print(f"Rows used for training: {len(matchups)}")
print(f"Model saved to {MODEL_PATH}")
print(f"Accuracy: {accuracy:.3f}")