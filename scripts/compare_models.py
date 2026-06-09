import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FEATURES_PATH = "data/processed/features.csv"

data = pd.read_csv(FEATURES_PATH)

home_games = data[data["IS_HOME"] == 1].copy()
away_games = data[data["IS_HOME"] == 0].copy()

matchups = home_games.merge(
    away_games,
    on="GAME_ID",
    suffixes=("_HOME", "_AWAY")
)

matchups["WIN"] = matchups["WIN_HOME"]

matchups["WIN_RATE_DIFF"] = (
    matchups["WIN_RATE_HOME"] - matchups["WIN_RATE_AWAY"]
)

matchups["LAST_10_WIN_RATE_DIFF"] = (
    matchups["LAST_10_WIN_RATE_HOME"] - matchups["LAST_10_WIN_RATE_AWAY"]
)

matchups["AVG_POINT_DIFF_DIFF"] = (
    matchups["AVG_POINT_DIFF_HOME"] - matchups["AVG_POINT_DIFF_AWAY"]
)

matchups["AVG_POINTS_FOR_DIFF"] = (
    matchups["AVG_POINTS_FOR_HOME"] - matchups["AVG_POINTS_FOR_AWAY"]
)

matchups["LAST_10_AVG_POINTS_DIFF"] = (
    matchups["LAST_10_AVG_POINTS_HOME"] - matchups["LAST_10_AVG_POINTS_AWAY"]
)

matchups["AVG_POINTS_ALLOWED_DIFF"] = (
    matchups["AVG_POINTS_ALLOWED_HOME"] - matchups["AVG_POINTS_ALLOWED_AWAY"]
)

matchups["LAST_10_POINTS_ALLOWED_DIFF"] = (
    matchups["LAST_10_POINTS_ALLOWED_HOME"] - matchups["LAST_10_POINTS_ALLOWED_AWAY"]
)

matchups["REST_DAYS_DIFF"] = (
    matchups["REST_DAYS_HOME"] - matchups["REST_DAYS_AWAY"]
)

matchups["LAST_5_WIN_RATE_DIFF"] = (
    matchups["LAST_5_WIN_RATE_HOME"] - matchups["LAST_5_WIN_RATE_AWAY"]
)

matchups["LAST_5_AVG_POINTS_DIFF"] = (
    matchups["LAST_5_AVG_POINTS_HOME"] - matchups["LAST_5_AVG_POINTS_AWAY"]
)

matchups["LAST_5_POINTS_ALLOWED_DIFF"] = (
    matchups["LAST_5_POINTS_ALLOWED_HOME"] - matchups["LAST_5_POINTS_ALLOWED_AWAY"]
)

matchups["HOME_ADVANTAGE_DIFF"] = (
    matchups["HOME_WIN_RATE_HOME"] - matchups["AWAY_WIN_RATE_AWAY"]
)

feature_cols = [
    "WIN_RATE_DIFF",
    "LAST_10_WIN_RATE_DIFF",
    "AVG_POINT_DIFF_DIFF",
    "AVG_POINTS_FOR_DIFF",
    "LAST_10_AVG_POINTS_DIFF",
    "AVG_POINTS_ALLOWED_DIFF",
    "LAST_10_POINTS_ALLOWED_DIFF",
    "REST_DAYS_DIFF",
    "LAST_5_WIN_RATE_DIFF",
    "LAST_5_AVG_POINTS_DIFF",
    "LAST_5_POINTS_ALLOWED_DIFF",
    "HOME_ADVANTAGE_DIFF",
]

matchups = matchups.dropna(subset=feature_cols + ["WIN"])

X = matchups[feature_cols]
y = matchups["WIN"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    ),
}

if HAS_XGBOOST:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "log_loss": log_loss(y_test, probs),
        }
    )

results_df = pd.DataFrame(results).sort_values("log_loss")

print(results_df)