import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="NBA Prediction API",
    description="Predict NBA win probability using a baseline ML model.",
    version="0.1.0",
)

MODEL_PATH = "models/nba_win_model.joblib"


def load_model_package():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Run scripts/train_model.py first.")
    return joblib.load(MODEL_PATH)


model_package = load_model_package()
model = model_package["model"]
team_stats = model_package["team_stats"]


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/teams")
def get_teams():
    teams = team_stats["TEAM_ABBREVIATION"].sort_values().tolist()
    return {"teams": teams}


@app.get("/predict")
def predict(home: str, away: str):
    home = home.upper()
    away = away.upper()

    home_row = team_stats[team_stats["TEAM_ABBREVIATION"] == home]
    away_row = team_stats[team_stats["TEAM_ABBREVIATION"] == away]

    if home_row.empty:
        raise HTTPException(status_code=404, detail=f"Home team '{home}' not found.")

    if away_row.empty:
        raise HTTPException(status_code=404, detail=f"Away team '{away}' not found.")

    home_features = pd.DataFrame(
        [
            {
                "avg_points": home_row.iloc[0]["avg_points"],
                "win_rate": home_row.iloc[0]["win_rate"],
                "games_played": home_row.iloc[0]["games_played"],
            }
        ]
    )

    away_features = pd.DataFrame(
        [
            {
                "avg_points": away_row.iloc[0]["avg_points"],
                "win_rate": away_row.iloc[0]["win_rate"],
                "games_played": away_row.iloc[0]["games_played"],
            }
        ]
    )

    home_win_probability = model.predict_proba(home_features)[0][1]
    away_win_probability = model.predict_proba(away_features)[0][1]

    total = home_win_probability + away_win_probability

    normalized_home_probability = home_win_probability / total
    normalized_away_probability = away_win_probability / total

    return {
        "home_team": home,
        "away_team": away,
        "home_win_probability": round(normalized_home_probability, 4),
        "away_win_probability": round(normalized_away_probability, 4),
        "predicted_winner": home if normalized_home_probability > normalized_away_probability else away,
        "note": "Baseline model only. This does not include injuries, odds, rest days, or matchup-specific features.",
    }