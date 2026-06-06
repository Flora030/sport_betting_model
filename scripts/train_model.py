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

if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Run scripts/build_features.py first.")

data = pd.read_csv(FEATURES_PATH)

data["WIN"] = data["WIN"].map({"W": 1, "L": 0}) if data["WIN"].dtype == "object" else data["WIN"]

feature_cols = [
    "IS_HOME",
    "AVG_POINTS_FOR",
    "WIN_RATE",
    "LAST_10_WIN_RATE",
    "LAST_10_AVG_POINTS",
    "AVG_POINT_DIFF",
]

data = data.dropna(subset=feature_cols + ["WIN"])

X = data[feature_cols]
y = data["WIN"].astype(int)

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
    },
    MODEL_PATH,
)

print(f"Rows used for training: {len(data)}")
print(f"Model saved to {MODEL_PATH}")
print(f"Accuracy: {accuracy:.3f}")