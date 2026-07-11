import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from utils import load_data

from pathlib import Path
import warnings


warnings.filterwarnings("ignore")

TARGET = "is_mcu_canon"

# best params found via GridSearchCV during MT/experimentation (see notebook)
BEST_PARAMS = {
"learning_rate": 0.05,
    "max_depth": 4,
    "n_estimators": 100,
    "random_state": 42,
}

# calibrated via cross_val_predict during experimentation (see notebook) -
# default 0.5 undershoots recall for the canon class, 0.3 balances precision/recall better
THRESHOLD = 0.3


def predict_with_threshold(model, x, threshold=0.3):
    proba = model.predict_proba(x)[:, 1]
    return (proba >= threshold).astype(int)


def MT(data_path: str, output_path: str):
    df = load_data(data_path)

    # id is a row-order artifact from the raw data, not a real feature = leaks info
    x = df.drop(columns=[TARGET, "id"])
    y = df[TARGET]

    # honest hold-out check before committing to the final refit on all data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(**BEST_PARAMS)
    model.fit(x_train, y_train)

    y_pred = predict_with_threshold(model, x_test)
    print("==========HOLD-OUT REPORT==========")
    print(classification_report(y_test, y_pred))

    # refit on full dataset for the final artifact
    model.fit(x, y)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": THRESHOLD,
            "feature_names": x.columns.tolist(),
        },
        output_path,
    )
    return model


if __name__ == "__main__":
    try:
        model = MT(
            data_path="../data/processed-data/processed-data.csv",
            output_path="../models/model.pkl"
        )
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise