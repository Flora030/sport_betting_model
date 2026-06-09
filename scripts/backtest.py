import joblib
import pandas as pd

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/nba_win_model.joblib"

MIN_EDGE = 0.03

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
feature_cols = model_package["feature_cols"]

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

# fake sportsbook line for now: even odds = 50%
matchups["IMPLIED_PROB"] = 0.50
matchups["EDGE"] = matchups["HOME_WIN_PROB"] - matchups["IMPLIED_PROB"]

bets = matchups[matchups["EDGE"] >= MIN_EDGE].copy()
bets["BET_WON"] = bets["HOME_WIN"] == 1
bets["PROFIT"] = bets["BET_WON"].apply(lambda won: 1 if won else -1)

total_bets = len(bets)
wins = bets["BET_WON"].sum()
profit = bets["PROFIT"].sum()
roi = profit / total_bets if total_bets > 0 else 0

print("Backtest Results")
print("----------------")
print(f"Total games: {len(matchups)}")
print(f"Total bets: {total_bets}")
print(f"Wins: {wins}")
print(f"Win rate: {wins / total_bets if total_bets else 0:.3f}")
print(f"Profit: ${profit:.2f}")
print(f"ROI per $1 bet: {roi:.3f}")