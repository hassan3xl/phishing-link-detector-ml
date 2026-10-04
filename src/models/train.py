"""Phase 4: Model training and comparison.

Trains three models of increasing complexity on the same X_train/y_train
(from Phase 3's data/train_features.csv) and evaluates them on the same
held-out X_test/y_test for a fair, apples-to-apples comparison:

- Logistic regression: an interpretable linear baseline. Coefficients are
  directly readable (sign + magnitude = direction + strength of each
  feature's effect), which makes it a useful sanity check even though it
  usually won't win on raw performance.
- Random forest: an ensemble of decision trees that captures non-linear
  feature interactions (e.g. "high entropy AND a risky TLD" mattering more
  together than either alone) without much tuning, and needs no scaling
  since tree splits threshold one feature at a time regardless of its scale.
- XGBoost: gradient-boosted trees, typically the strongest raw performer on
  structured/tabular data like this because each new tree explicitly
  corrects the previous ensemble's errors - but it's the least directly
  interpretable of the three, which is exactly why Phase 6 (SHAP) matters.

Logistic regression is scale-sensitive (its coefficients and convergence
depend on feature ranges - a feature spanning 0..1800 like url_length would
dominate one spanning 0..1 like digit_ratio_domain purely from scale), so it
gets a StandardScaler fit on X_train only, wrapped in the same sklearn
Pipeline that's saved - this guarantees the fitted scaler travels with the
model artifact and is never accidentally refit on test/inference data.
Tree-based models (random forest, XGBoost) do not need scaling: a split
threshold on one feature is scale-invariant.

Run: python -m src.models.train
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.extract import FEATURE_NAMES

TRAIN_PATH = Path("data/train_features.csv")
TEST_PATH = Path("data/test_features.csv")
MODELS_DIR = Path("models_saved")
RANDOM_STATE = 42
CV_FOLDS = 5


def load_xy(path: Path):
    df = pd.read_csv(path)
    return df[FEATURE_NAMES], df["label"]


def cv_score(model, X, y, label: str) -> float:
    """5-fold stratified CV ROC-AUC on the training set only - used to pick
    a robust winner for the tuning pass without ever touching the test set
    (the same over-tuning-on-the-test-set pitfall the README's Appendix
    names). Tradeoff: 5x the training compute of a single fit, but a much
    more reliable estimate than one train/test split, which matters more as
    model complexity (and variance) increases from logistic regression to
    XGBoost.
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"[cv] {label}: ROC-AUC {scores.mean():.4f} +/- {scores.std():.4f} "
          f"({time.time() - t0:.1f}s)")
    return float(scores.mean())


def quick_test_metrics(model, X_test, y_test, label: str) -> dict:
    """Reports precision/recall/F1 with PHISHING (label=0) as the positive
    class, not sklearn's default label=1 (legitimate) - operationally, what
    matters is "did we catch the phishing URL," so recall must mean "of all
    actual phishing URLs, how many did we catch," not "of all legitimate
    URLs, how many did we correctly pass." ROC-AUC is threshold/label-
    orientation independent so it's unaffected by this choice.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    proba_legit = model.predict_proba(X_test)[:, 1]
    proba_phish = 1 - proba_legit
    y_phish = (y_test == 0).astype(int)
    pred_phish = (proba_phish >= 0.5).astype(int)
    metrics = {
        "precision": round(precision_score(y_phish, pred_phish), 4),
        "recall": round(recall_score(y_phish, pred_phish), 4),
        "f1": round(f1_score(y_phish, pred_phish), 4),
        "roc_auc": round(roc_auc_score(y_phish, proba_phish), 4),
    }
    print(f"[test] {label} (phishing=positive class): {metrics}")
    return metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    results = {}

    # --- Logistic regression baseline (scaled) ---
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    cv_auc = cv_score(logreg, X_train, y_train, "logistic_regression")
    logreg.fit(X_train, y_train)
    test_metrics = quick_test_metrics(logreg, X_test, y_test, "logistic_regression")
    joblib.dump(logreg, MODELS_DIR / "logistic_regression_v1.joblib")
    results["logistic_regression"] = {"cv_roc_auc": cv_auc, **test_metrics}

    # --- Random forest (no scaling) ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
    )
    cv_auc = cv_score(rf, X_train, y_train, "random_forest")
    rf.fit(X_train, y_train)
    test_metrics = quick_test_metrics(rf, X_test, y_test, "random_forest")
    joblib.dump(rf, MODELS_DIR / "random_forest_v1.joblib")
    results["random_forest"] = {"cv_roc_auc": cv_auc, **test_metrics}

    # --- XGBoost (no scaling) ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
    )
    cv_auc = cv_score(xgb, X_train, y_train, "xgboost")
    xgb.fit(X_train, y_train)
    test_metrics = quick_test_metrics(xgb, X_test, y_test, "xgboost")
    joblib.dump(xgb, MODELS_DIR / "xgboost_v1.joblib")
    results["xgboost"] = {"cv_roc_auc": cv_auc, **test_metrics}

    # --- Light hyperparameter tuning on the CV-selected best model ---
    best_name = max(results, key=lambda k: results[k]["cv_roc_auc"])
    print(f"\n[tune] best initial model by CV ROC-AUC: {best_name}")

    if best_name == "xgboost":
        param_dist = {
            "n_estimators": [200, 300, 400, 600],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.85, 1.0],
        }
        base = XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
    elif best_name == "random_forest":
        param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_leaf": [1, 2, 4],
        }
        base = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE)
    else:
        param_dist = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"]}
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=12, scoring="roc_auc", cv=skf,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"[tune] done in {time.time() - t0:.1f}s")
    print(f"[tune] best params: {search.best_params_}")
    print(f"[tune] best CV ROC-AUC: {search.best_score_:.4f} "
          f"(vs. untuned {results[best_name]['cv_roc_auc']:.4f})")

    tuned_model = search.best_estimator_
    tuned_metrics = quick_test_metrics(tuned_model, X_test, y_test, f"{best_name}_tuned")
    joblib.dump(tuned_model, MODELS_DIR / f"{best_name}_tuned_v1.joblib")
    results[f"{best_name}_tuned"] = {
        "cv_roc_auc": round(search.best_score_, 4),
        "best_params": search.best_params_,
        **tuned_metrics,
    }

    with open(MODELS_DIR / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[save] comparison table written to models_saved/comparison_results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
