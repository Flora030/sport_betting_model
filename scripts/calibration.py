import joblib
import pandas as pd

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/nba_win_model.joblib"

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
feature_cols = model_package["feature_cols"]

data = pd.read_csv(FEATURES_PATH).dropna()

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

matchups = matchups.dropna(subset=feature_cols + ["HOME_WIN"])

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

calibration = (
    matchups.groupby("PROB_BUCKET", observed=False)
    .agg(
        games=("GAME_ID", "count"),
        avg_predicted_probability=("PREDICTED_PROB", "mean"),
        actual_win_rate=("HOME_WIN", "mean"),
    )
    .reset_index()
)

calibration["calibration_error"] = (
    calibration["avg_predicted_probability"] - calibration["actual_win_rate"]
).abs()

print(calibration.to_string(index=False))

print("\nOverall Calibration")
print("-------------------")
print(f"Average calibration error: {calibration['calibration_error'].mean():.4f}")