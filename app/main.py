import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
import requests
from dotenv import load_dotenv

FEATURES_PATH = "data/processed/features.csv"

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

def fetch_nba_moneyline_odds():
    if not ODDS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ODDS_API_KEY in .env")

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    return response.json()

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
precision = model_package.get("precision")
log_loss = model_package.get("log_loss")
model_type = model_package.get("model_type", "unknown")

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

def build_matchup_dataset():
    features_path = "data/processed/features.csv"

    if not os.path.exists(features_path):
        raise HTTPException(
            status_code=404,
            detail="features.csv not found. Run build_features.py first."
        )

    data = pd.read_csv(FEATURES_PATH)

    home_games = data[data["IS_HOME"] == 1].copy()
    away_games = data[data["IS_HOME"] == 0].copy()

    matchups = home_games.merge(
        away_games,
        on="GAME_ID",
        suffixes=("_HOME", "_AWAY")
    )

    matchups["HOME_WIN"] = matchups["WIN_HOME"]

    matchups["WIN_RATE_DIFF"] = matchups["WIN_RATE_HOME"] - matchups["WIN_RATE_AWAY"]
    matchups["LAST_10_WIN_RATE_DIFF"] = matchups["LAST_10_WIN_RATE_HOME"] - matchups["LAST_10_WIN_RATE_AWAY"]
    matchups["AVG_POINT_DIFF_DIFF"] = matchups["AVG_POINT_DIFF_HOME"] - matchups["AVG_POINT_DIFF_AWAY"]
    matchups["AVG_POINTS_FOR_DIFF"] = matchups["AVG_POINTS_FOR_HOME"] - matchups["AVG_POINTS_FOR_AWAY"]
    matchups["LAST_10_AVG_POINTS_DIFF"] = matchups["LAST_10_AVG_POINTS_HOME"] - matchups["LAST_10_AVG_POINTS_AWAY"]
    matchups["AVG_POINTS_ALLOWED_DIFF"] = matchups["AVG_POINTS_ALLOWED_HOME"] - matchups["AVG_POINTS_ALLOWED_AWAY"]
    matchups["LAST_10_POINTS_ALLOWED_DIFF"] = matchups["LAST_10_POINTS_ALLOWED_HOME"] - matchups["LAST_10_POINTS_ALLOWED_AWAY"]
    matchups["REST_DAYS_DIFF"] = matchups["REST_DAYS_HOME"] - matchups["REST_DAYS_AWAY"]
    matchups["REST_DAYS_DIFF"] = matchups["REST_DAYS_HOME"] - matchups["REST_DAYS_AWAY"]
    matchups["LAST_5_WIN_RATE_DIFF"] = (
        matchups["LAST_5_WIN_RATE_HOME"] - matchups["LAST_5_WIN_RATE_AWAY"])
    matchups["LAST_5_AVG_POINTS_DIFF"] = (
        matchups["LAST_5_AVG_POINTS_HOME"] - matchups["LAST_5_AVG_POINTS_AWAY"])
    matchups["LAST_5_POINTS_ALLOWED_DIFF"] = (
        matchups["LAST_5_POINTS_ALLOWED_HOME"] - matchups["LAST_5_POINTS_ALLOWED_AWAY"])
    matchups["HOME_ADVANTAGE_DIFF"] = (
        matchups["HOME_WIN_RATE_HOME"] - matchups["AWAY_WIN_RATE_AWAY"])

    return matchups.dropna(subset=feature_cols + ["HOME_WIN"])

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/model-info")
def model_info():
    return {
        "model_type": model_type,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4) if precision else None,
        "log_loss": round(log_loss, 4) if log_loss else None,
        "features": feature_cols,
    }

@app.get("/teams")
def get_teams():
    teams = (
        team_latest_features["TEAM_ABBREVIATION"]
        .sort_values()
        .tolist()
    )
    return {"teams": teams}

@app.get("/team-stats/{team}")
def get_team_stats(team: str):
    team = team.upper()

    row = team_latest_features[
        team_latest_features["TEAM_ABBREVIATION"] == team
    ]

    if row.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team}' not found.")

    row = row.iloc[0]

    return {
        "team": team,
        "latest_game_date": str(row["GAME_DATE"]),
        "is_home_last_game": int(row["IS_HOME"]),
        "avg_points_for": round(float(row["AVG_POINTS_FOR"]), 2),
        "win_rate": round(float(row["WIN_RATE"]), 4),
        "last_10_win_rate": round(float(row["LAST_10_WIN_RATE"]), 4),
        "last_10_avg_points": round(float(row["LAST_10_AVG_POINTS"]), 2),
        "avg_point_diff": round(float(row["AVG_POINT_DIFF"]), 2),
    }

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

    home_row = home_row.iloc[0]
    away_row = away_row.iloc[0]

    matchup_features = pd.DataFrame([
        {
            "WIN_RATE_DIFF": home_row["WIN_RATE"] - away_row["WIN_RATE"],
            "LAST_10_WIN_RATE_DIFF": home_row["LAST_10_WIN_RATE"] - away_row["LAST_10_WIN_RATE"],
            "AVG_POINT_DIFF_DIFF": home_row["AVG_POINT_DIFF"] - away_row["AVG_POINT_DIFF"],
            "AVG_POINTS_FOR_DIFF": home_row["AVG_POINTS_FOR"] - away_row["AVG_POINTS_FOR"],
            "LAST_10_AVG_POINTS_DIFF": home_row["LAST_10_AVG_POINTS"] - away_row["LAST_10_AVG_POINTS"],
            "AVG_POINTS_ALLOWED_DIFF": home_row["AVG_POINTS_ALLOWED"] - away_row["AVG_POINTS_ALLOWED"],
            "LAST_10_POINTS_ALLOWED_DIFF": home_row["LAST_10_POINTS_ALLOWED"] - away_row["LAST_10_POINTS_ALLOWED"],
            "REST_DAYS_DIFF": home_row["REST_DAYS"] - away_row["REST_DAYS"], "LAST_5_WIN_RATE_DIFF": home_row["LAST_5_WIN_RATE"] - away_row["LAST_5_WIN_RATE"],
            "LAST_5_AVG_POINTS_DIFF": home_row["LAST_5_AVG_POINTS"] - away_row["LAST_5_AVG_POINTS"],
            "LAST_5_POINTS_ALLOWED_DIFF": home_row["LAST_5_POINTS_ALLOWED"] - away_row["LAST_5_POINTS_ALLOWED"],
            "HOME_ADVANTAGE_DIFF": home_row["HOME_WIN_RATE"] - away_row["AWAY_WIN_RATE"],
        }
    ])

    home_prob = model.predict_proba(matchup_features)[0][1]
    away_prob = 1 - home_prob

    return {
        "home_team": home,
        "away_team": away,
        "home_win_probability": round(home_prob, 4),
        "away_win_probability": round(away_prob, 4),
        "predicted_winner": home if home_prob > away_prob else away,
        "model_accuracy": round(accuracy, 4),
        "model_precision": round(precision, 4) if precision else None,
        "model_log_loss": round(log_loss, 4) if log_loss else None,
        "model_type": model_type,
        "note": "Matchup-difference model with rest days and defensive features.",
    }

@app.get("/backtest")
def backtest(min_edge: float = 0.03):
    features_path = "data/processed/features.csv"

    if not os.path.exists(features_path):
        raise HTTPException(status_code=404, detail="features.csv not found. Run build_features.py first.")

    data = pd.read_csv(FEATURES_PATH)

    home_games = data[data["IS_HOME"] == 1].copy()
    away_games = data[data["IS_HOME"] == 0].copy()

    matchups = home_games.merge(
        away_games,
        on="GAME_ID",
        suffixes=("_HOME", "_AWAY")
    )

    matchups["HOME_WIN"] = matchups["WIN_HOME"]

    matchups["WIN_RATE_DIFF"] = matchups["WIN_RATE_HOME"] - matchups["WIN_RATE_AWAY"]
    matchups["LAST_10_WIN_RATE_DIFF"] = matchups["LAST_10_WIN_RATE_HOME"] - matchups["LAST_10_WIN_RATE_AWAY"]
    matchups["AVG_POINT_DIFF_DIFF"] = matchups["AVG_POINT_DIFF_HOME"] - matchups["AVG_POINT_DIFF_AWAY"]
    matchups["AVG_POINTS_FOR_DIFF"] = matchups["AVG_POINTS_FOR_HOME"] - matchups["AVG_POINTS_FOR_AWAY"]
    matchups["LAST_10_AVG_POINTS_DIFF"] = matchups["LAST_10_AVG_POINTS_HOME"] - matchups["LAST_10_AVG_POINTS_AWAY"]
    matchups["AVG_POINTS_ALLOWED_DIFF"] = matchups["AVG_POINTS_ALLOWED_HOME"] - matchups["AVG_POINTS_ALLOWED_AWAY"]
    matchups["LAST_10_POINTS_ALLOWED_DIFF"] = matchups["LAST_10_POINTS_ALLOWED_HOME"] - matchups["LAST_10_POINTS_ALLOWED_AWAY"]
    matchups["REST_DAYS_DIFF"] = matchups["REST_DAYS_HOME"] - matchups["REST_DAYS_AWAY"]
    matchups["LAST_5_WIN_RATE_DIFF"] = (
    matchups["LAST_5_WIN_RATE_HOME"] - matchups["LAST_5_WIN_RATE_AWAY"])
    matchups["LAST_5_AVG_POINTS_DIFF"] = (
        matchups["LAST_5_AVG_POINTS_HOME"] - matchups["LAST_5_AVG_POINTS_AWAY"])
    matchups["LAST_5_POINTS_ALLOWED_DIFF"] = (
        matchups["LAST_5_POINTS_ALLOWED_HOME"] - matchups["LAST_5_POINTS_ALLOWED_AWAY"])
    matchups["HOME_ADVANTAGE_DIFF"] = (
        matchups["HOME_WIN_RATE_HOME"] - matchups["AWAY_WIN_RATE_AWAY"])
    matchups = matchups.dropna(subset=feature_cols + ["HOME_WIN"])

    X = matchups[feature_cols]
    matchups["HOME_WIN_PROB"] = model.predict_proba(X)[:, 1]

    # temporary fake even odds
    matchups["IMPLIED_PROB"] = 0.50
    matchups["EDGE"] = matchups["HOME_WIN_PROB"] - matchups["IMPLIED_PROB"]

    bets = matchups[matchups["EDGE"] >= min_edge].copy()
    bets["BET_WON"] = bets["HOME_WIN"] == 1
    bets["PROFIT"] = bets["BET_WON"].apply(lambda won: 1 if won else -1)

    total_bets = len(bets)
    wins = int(bets["BET_WON"].sum())
    profit = float(bets["PROFIT"].sum())
    roi = profit / total_bets if total_bets > 0 else 0

    return {
        "total_games": len(matchups),
        "minimum_edge": min_edge,
        "total_bets": total_bets,
        "wins": wins,
        "win_rate": round(wins / total_bets, 4) if total_bets else 0,
        "profit_per_1_dollar_bet": round(profit, 2),
        "roi_per_1_dollar_bet": round(roi, 4),
        "note": "Backtest uses fake even odds for now. Real sportsbook odds will make this more realistic."
    }

@app.get("/live-odds")
def live_odds():
    odds_data = fetch_nba_moneyline_odds()

    games = []

    for game in odds_data:
        bookmakers = game.get("bookmakers", [])

        if not bookmakers:
            continue

        bookmaker = bookmakers[0]
        markets = bookmaker.get("markets", [])

        if not markets:
            continue

        outcomes = markets[0].get("outcomes", [])

        games.append({
            "id": game.get("id"),
            "commence_time": game.get("commence_time"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "bookmaker": bookmaker.get("title"),
            "odds": outcomes,
        })

    return {"games": games}

def run_bet_analysis(home: str, away: str, home_odds: int, away_odds: int):
    prediction = predict(home, away)

    home_model_prob = prediction["home_win_probability"]
    away_model_prob = prediction["away_win_probability"]

    home_implied_prob = american_odds_to_implied_probability(home_odds)
    away_implied_prob = american_odds_to_implied_probability(away_odds)

    home_edge = home_model_prob - home_implied_prob
    away_edge = away_model_prob - away_implied_prob

    home_ev = calculate_expected_value(home_model_prob, home_odds)
    away_ev = calculate_expected_value(away_model_prob, away_odds)

    MIN_EDGE = 0.03
    best_bet = None
    confidence = "No Bet"

    if home_edge >= MIN_EDGE and home_ev > 0 and home_ev > away_ev:
        best_bet = home.upper()
        confidence = "High" if home_edge >= 0.08 else "Medium" if home_edge >= 0.05 else "Low"

    elif away_edge >= MIN_EDGE and away_ev > 0 and away_ev > home_ev:
        best_bet = away.upper()
        confidence = "High" if away_edge >= 0.08 else "Medium" if away_edge >= 0.05 else "Low"

    return {
        "matchup": f"{away.upper()} @ {home.upper()}",
        "home_team": home.upper(),
        "away_team": away.upper(),
        "minimum_edge_required": MIN_EDGE,
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
        "confidence": confidence,
        "note": "Positive EV means the model probability is better than the sportsbook implied probability. This is not betting advice.",
    }

@app.get("/bet-analysis")
def bet_analysis(home: str, away: str, home_odds: int, away_odds: int):
    return run_bet_analysis(home, away, home_odds, away_odds)

@app.get("/live-bet-analysis")
def live_bet_analysis():
    odds_data = fetch_nba_moneyline_odds()

    results = []

    name_to_abbr = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "LA Clippers": "LAC",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

    for game in odds_data:
        home_team_name = game.get("home_team")
        away_team_name = game.get("away_team")

        home_abbr = name_to_abbr.get(home_team_name)
        away_abbr = name_to_abbr.get(away_team_name)

        if not home_abbr or not away_abbr:
            continue

        best_home_odds = None
        best_away_odds = None
        best_home_bookmaker = None
        best_away_bookmaker = None

        all_bookmaker_odds = []

        for bookmaker in game.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title")
            markets = bookmaker.get("markets", [])

            if not markets:
                continue

            outcomes = markets[0].get("outcomes", [])

            current_home_odds = None
            current_away_odds = None

            for outcome in outcomes:
                if outcome["name"] == home_team_name:
                    current_home_odds = outcome["price"]
                elif outcome["name"] == away_team_name:
                    current_away_odds = outcome["price"]

            if current_home_odds is None or current_away_odds is None:
                continue

            all_bookmaker_odds.append({
                "bookmaker": bookmaker_name,
                "home_odds": current_home_odds,
                "away_odds": current_away_odds,
            })

            if best_home_odds is None or current_home_odds > best_home_odds:
                best_home_odds = current_home_odds
                best_home_bookmaker = bookmaker_name

            if best_away_odds is None or current_away_odds > best_away_odds:
                best_away_odds = current_away_odds
                best_away_bookmaker = bookmaker_name

        if best_home_odds is None or best_away_odds is None:
            continue

        analysis = run_bet_analysis(
            home=home_abbr,
            away=away_abbr,
            home_odds=best_home_odds,
            away_odds=best_away_odds,
        )

        analysis["commence_time"] = game.get("commence_time")
        analysis["home_team_full_name"] = home_team_name
        analysis["away_team_full_name"] = away_team_name
        analysis["best_home_bookmaker"] = best_home_bookmaker
        analysis["best_away_bookmaker"] = best_away_bookmaker

        if analysis["best_bet"] == home_abbr:
            analysis["recommended_bookmaker"] = best_home_bookmaker
        elif analysis["best_bet"] == away_abbr:
            analysis["recommended_bookmaker"] = best_away_bookmaker
        else:
            analysis["recommended_bookmaker"] = None

        analysis["all_bookmaker_odds"] = all_bookmaker_odds
        results.append(analysis)

    return {
        "games_analyzed": len(results),
        "games": results,
    }

@app.get("/calibration")
def calibration():
    matchups = build_matchup_dataset()

    X = matchups[feature_cols]
    matchups["PREDICTED_PROB"] = model.predict_proba(X)[:, 1]

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    matchups["PROB_BUCKET"] = pd.cut(
        matchups["PREDICTED_PROB"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    cal = (
        matchups.groupby("PROB_BUCKET", observed=False)
        .agg(
            games=("GAME_ID", "count"),
            avg_predicted_probability=("PREDICTED_PROB", "mean"),
            actual_win_rate=("HOME_WIN", "mean"),
        )
        .reset_index()
    )

    cal["calibration_error"] = (
        cal["avg_predicted_probability"] - cal["actual_win_rate"]
    ).abs()

    rows = []

    for _, row in cal.iterrows():
        rows.append({
            "probability_bucket": str(row["PROB_BUCKET"]),
            "games": int(row["games"]),
            "avg_predicted_probability": round(float(row["avg_predicted_probability"]), 4)
            if pd.notnull(row["avg_predicted_probability"]) else None,
            "actual_win_rate": round(float(row["actual_win_rate"]), 4)
            if pd.notnull(row["actual_win_rate"]) else None,
            "calibration_error": round(float(row["calibration_error"]), 4)
            if pd.notnull(row["calibration_error"]) else None,
        })

    avg_error = cal["calibration_error"].dropna().mean()

    return {
        "average_calibration_error": round(float(avg_error), 4),
        "buckets": rows,
        "note": "If predicted probability is well-calibrated, avg_predicted_probability should be close to actual_win_rate in each bucket.",
    }

@app.get("/save-paper-bets")
def save_paper_bets():
    analysis = live_bet_analysis()
    rows = []

    for game in analysis["games"]:
        if game["best_bet"] is None:
            continue

        best_bet = game["best_bet"]

        if best_bet == game["home_team"]:
            side = game["home"]
        else:
            side = game["away"]

        rows.append({
            "date": game["commence_time"],
            "matchup": game["matchup"],
            "best_bet": best_bet,
            "bookmaker": game["recommended_bookmaker"],
            "odds": side["american_odds"],
            "model_probability": side["model_probability"],
            "edge": side["edge"],
            "confidence": game["confidence"],
            "result": "",
            "profit": "",
        })

    path = "data/paper_bets.csv"

    new_df = pd.DataFrame(rows)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        old_df = pd.read_csv(path)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(
            subset=["date", "matchup", "best_bet"],
            keep="last"
        )
    else:
        final_df = new_df

    final_df.to_csv(path, index=False)

    return {
        "saved_bets": len(rows),
        "file": path,
        "note": "After games finish, manually update result as WIN or LOSS and profit."
    }