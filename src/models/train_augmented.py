"""Trains the Augmented Random Forest Model (Phase 2 Roadmap).

Trains a 300-tree RandomForestClassifier on data/train_augmented_features.csv
(238,296 rows: 188,296 original PhiUSIIL + 50,000 augmented legitimate deep links)
and evaluates on data/test_features.csv (47,074 rows) + deep link stress tests.

Saves:
- models_saved/random_forest_v2_augmented.joblib
- models_saved/final_model.joblib (updated production artifact)
- models_saved/final_model_metadata.json

Run: python -m src.models.train_augmented
"""

import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from src.features.extract import FEATURE_NAMES, extract_features

AUGMENTED_TRAIN = Path("data/train_augmented_features.csv")
TEST_PATH = Path("data/test_features.csv")
MODELS_DIR = Path("models_saved")
MODEL_OUT = MODELS_DIR / "random_forest_v2_augmented.joblib"
FINAL_MODEL = MODELS_DIR / "final_model.joblib"
METADATA_OUT = MODELS_DIR / "final_model_metadata.json"

RANDOM_STATE = 42


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] loading augmented training set from {AUGMENTED_TRAIN}...")
    df_train = pd.read_csv(AUGMENTED_TRAIN)
    X_train = df_train[FEATURE_NAMES]
    y_train = df_train["label"]

    print(f"[load] loading held-out test set from {TEST_PATH}...")
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test[FEATURE_NAMES]
    y_test = df_test["label"]

    print(f"[info] train shape: {X_train.shape}, test shape: {X_test.shape}")

    # Configure Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("\n[train] fitting 300-tree Random Forest on augmented dataset...")
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_duration = time.time() - t0
    print(f"[train] completed training in {train_duration:.2f} seconds.")

    # Save artifact
    joblib.dump(rf, MODEL_OUT)
    print(f"[save] saved model to {MODEL_OUT}")

    # Evaluate on held-out test set (phishing = label 0, positive class for detection)
    proba_legit = rf.predict_proba(X_test)[:, 1]
    proba_phish = 1.0 - proba_legit
    y_test_phish = (y_test == 0).astype(int)

    pred_phish_05 = (proba_phish >= 0.5).astype(int)
    acc = accuracy_score(y_test_phish, pred_phish_05)
    prec = precision_score(y_test_phish, pred_phish_05)
    rec = recall_score(y_test_phish, pred_phish_05)
    f1 = f1_score(y_test_phish, pred_phish_05)
    auc = roc_auc_score(y_test_phish, proba_phish)
    cm = confusion_matrix(y_test_phish, pred_phish_05)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 65)
    print("HELD-OUT TEST SET METRICS (N = 47,074, Phishing = Positive Class)")
    print("=" * 65)
    print(f"Accuracy:         {acc:.5f} ({acc * 100:.2f}%)")
    print(f"Precision:        {prec:.5f} ({prec * 100:.2f}%)")
    print(f"Recall:           {rec:.5f} ({rec * 100:.2f}%)")
    print(f"F1-Score:         {f1:.5f} ({f1 * 100:.2f}%)")
    print(f"ROC-AUC:          {auc:.5f}")
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

    # Feature Importances
    importances = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("\nFEATURE IMPORTANCES (Gini Impurity / MDI):")
    for name, imp in importances.items():
        print(f"  {name:25s}: {imp:.5f} ({imp * 100:.2f}%)")

    # Real-World Deep-Link Stress Test
    stress_urls = [
        ("https://claude.ai/chat/", "Legitimate"),
        ("https://chatgpt.com/c/6abfdef0-2040-83ea-af6b-39608909ab10", "Legitimate"),
        ("https://en.wikipedia.org/wiki/Phishing", "Legitimate"),
        ("https://github.com/torvalds/linux", "Legitimate"),
        ("https://www.google.com/search?q=cybersecurity", "Legitimate"),
        ("http://paypal-secure-login.xyz/confirm?id=12345", "Phishing"),
        ("http://192.168.1.1/login.php?user=admin", "Phishing"),
        ("http://account-verification.top/auth", "Phishing"),
    ]

    print("\n" + "=" * 65)
    print("STRESS TEST: REAL-WORLD DEEP-LINK VERIFICATION (RAW TREE OUTPUT)")
    print("=" * 65)
    for raw_url, ground_truth in stress_urls:
        f = extract_features(raw_url)
        X_sample = pd.DataFrame([f])[FEATURE_NAMES]
        p_phish = float(1.0 - rf.predict_proba(X_sample)[:, 1][0])
        pred = "phishing" if p_phish >= 0.5 else "legitimate"
        is_correct = (pred.lower() == ground_truth.lower())
        status = "CORRECT" if is_correct else "FAILED"
        print(f"{raw_url[:55]:55s} | Expected: {ground_truth:10s} | Pred: {pred:10s} (p={p_phish:.4f}) -> {status}")

    # Set as active final model
    shutil.copyfile(MODEL_OUT, FINAL_MODEL)
    print(f"\n[deploy] copied {MODEL_OUT} to {FINAL_MODEL}")

    metadata = {
        "model_type": "random_forest_v2_augmented",
        "source_artifact": str(MODEL_OUT),
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 25,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
            "random_state": 42,
        },
        "training_data": {
            "samples": len(df_train),
            "original_samples": 188296,
            "augmented_samples": 50000,
        },
        "test_metrics": {
            "accuracy": round(acc, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1": round(f1, 5),
            "roc_auc": round(auc, 5),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "positive_class": "phishing (label=0)",
        "selection_rationale": (
            "Random Forest v2 trained on 238,296 samples with 50,000 legitimate deep-link "
            "augmentations, resolving the PhiUSIIL zero-path generalization failure while "
            f"maintaining {acc * 100:.2f}% accuracy and {prec * 100:.2f}% precision."
        ),
    }

    METADATA_OUT.write_text(json.dumps(metadata, indent=2))
    print(f"[metadata] wrote updated metadata to {METADATA_OUT}")


if __name__ == "__main__":
    main()
