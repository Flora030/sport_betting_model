import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

GAMES_PATH = "data/raw/games.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "nba_win_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(GAMES_PATH):
    raise FileNotFoundError("Run scripts/fetch_games.py first.")

games = pd.read_csv(GAMES_PATH)
games = games.dropna(subset=["WL", "PTS"])

games["WIN"] = games["WL"].map({"W": 1, "L": 0})

team_stats = (
    games.groupby("TEAM_ABBREVIATION")
    .agg(
        avg_points=("PTS", "mean"),
        win_rate=("WIN", "mean"),
        games_played=("GAME_ID", "count"),
    )
    .reset_index()
)

training_data = games.merge(team_stats, on="TEAM_ABBREVIATION", how="left")

X = training_data[["avg_points", "win_rate", "games_played"]]
y = training_data["WIN"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump({"model": model, "team_stats": team_stats}, MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")
print(f"Accuracy: {accuracy:.3f}")