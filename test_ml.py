import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _prep_small_split(n_rows: int = 200):
    """Load a small slice of data and return processed train/test arrays."""
    df = pd.read_csv("data/census.csv").head(n_rows)
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )
    X_train, y_train, enc, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=enc,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_train_model_returns_logreg():
    X_train, y_train, _, _ = _prep_small_split()
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)


def test_inference_shape_and_dtype():
    X_train, y_train, X_test, _ = _prep_small_split()
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, (np.ndarray, list))
    assert len(preds) == X_test.shape[0]


def test_compute_model_metrics_known_values():
    # y and preds chosen so P=R=F1=0.5 (TP=1, FP=1, FN=1)
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1])
    p, r, f1 = compute_model_metrics(y_true, y_pred)
    assert abs(p - 0.5) < 1e-9
    assert abs(r - 0.5) < 1e-9
    assert abs(f1 - 0.5) < 1e-9
