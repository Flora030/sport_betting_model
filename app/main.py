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

def american_odds_to_implied_probability(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def calculate_expected_value(model_prob: float, american_odds: int) -> float:
    if american_odds > 0:
        profit_if_win = american_odds / 100
    else:
        profit_if_win = 100 / abs(american_odds)

    probability_losing = 1 - model_prob

    ev = (model_prob * profit_if_win) - probability_losing
    return ev

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

@app.get("/bet-analysis")
def bet_analysis(
    home: str,
    away: str,
    home_odds: int,
    away_odds: int
):
    prediction = predict(home, away)

    home_model_prob = prediction["home_win_probability"]
    away_model_prob = prediction["away_win_probability"]

    home_implied_prob = american_odds_to_implied_probability(home_odds)
    away_implied_prob = american_odds_to_implied_probability(away_odds)

    home_edge = home_model_prob - home_implied_prob
    away_edge = away_model_prob - away_implied_prob

    home_ev = calculate_expected_value(home_model_prob, home_odds)
    away_ev = calculate_expected_value(away_model_prob, away_odds)

    best_bet = None

    if home_ev > 0 and home_ev > away_ev:
        best_bet = home.upper()
    elif away_ev > 0 and away_ev > home_ev:
        best_bet = away.upper()

    return {
        "matchup": f"{away.upper()} @ {home.upper()}",
        "home_team": home.upper(),
        "away_team": away.upper(),
        "home": {
            "american_odds": home_odds,
            "model_probability": round(home_model_prob, 4),
            "implied_probability": round(home_implied_prob, 4),
            "edge": round(home_edge, 4),
            "expected_value_per_1_dollar": round(home_ev, 4),
        },
        "away": {
            "american_odds": away_odds,
            "model_probability": round(away_model_prob, 4),
            "implied_probability": round(away_implied_prob, 4),
            "edge": round(away_edge, 4),
            "expected_value_per_1_dollar": round(away_ev, 4),
        },
        "best_bet": best_bet,
        "note": "Positive EV means the model probability is better than the sportsbook implied probability. This is not betting advice.",
    }