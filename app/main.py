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
team_latest_features = model_package["team_latest_features"]
feature_cols = model_package["feature_cols"]
accuracy = model_package["accuracy"]


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/teams")
def get_teams():
    teams = (
        team_latest_features["TEAM_ABBREVIATION"]
        .sort_values()
        .tolist()
    )
    return {"teams": teams}


@app.get("/predict")
def predict(home: str, away: str):
    home = home.upper()
    away = away.upper()

    home_row = team_latest_features[
        team_latest_features["TEAM_ABBREVIATION"] == home
    ]

    away_row = team_latest_features[
        team_latest_features["TEAM_ABBREVIATION"] == away
    ]

    if home_row.empty:
        raise HTTPException(status_code=404, detail=f"Home team '{home}' not found.")

    if away_row.empty:
        raise HTTPException(status_code=404, detail=f"Away team '{away}' not found.")

    home_features = home_row.iloc[0][feature_cols].copy()
    away_features = away_row.iloc[0][feature_cols].copy()

    home_features["IS_HOME"] = 1
    away_features["IS_HOME"] = 0

    home_df = pd.DataFrame([home_features])
    away_df = pd.DataFrame([away_features])

    home_score = model.predict_proba(home_df)[0][1]
    away_score = model.predict_proba(away_df)[0][1]

    total = home_score + away_score

    home_prob = home_score / total
    away_prob = away_score / total

    return {
        "home_team": home,
        "away_team": away,
        "home_win_probability": round(home_prob, 4),
        "away_win_probability": round(away_prob, 4),
        "predicted_winner": home if home_prob > away_prob else away,
        "model_accuracy": round(accuracy, 4),
        "note": "Improved baseline with recent form, home/away, point differential, and team features.",
    }