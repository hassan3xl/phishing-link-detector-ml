"""Tests that the final selected model artifact loads and predicts without
error - a basic but important smoke test, since the API depends on it."""

from pathlib import Path

import joblib
import pandas as pd

from src.features.extract import FEATURE_NAMES, extract_features

MODEL_PATH = Path("models_saved/final_model.joblib")


def test_model_artifact_exists():
    assert MODEL_PATH.exists(), (
        f"{MODEL_PATH} missing - run `python -m src.models.train` and "
        "`python -m src.models.select_final`"
    )


def test_model_loads_without_error():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict_proba")


def test_model_predicts_expected_shape():
    model = joblib.load(MODEL_PATH)
    features = extract_features("https://www.google.com")
    X = pd.DataFrame([features])[FEATURE_NAMES]
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_model_predicts_reasonable_direction():
    model = joblib.load(MODEL_PATH)

    legit_features = extract_features("https://www.google.com")
    phish_features = extract_features("http://paypal-secure-login.xyz/confirm?id=12345")

    X = pd.DataFrame([legit_features, phish_features])[FEATURE_NAMES]
    proba_legit = model.predict_proba(X)[:, 1]  # P(label=1, legitimate)

    assert proba_legit[0] > proba_legit[1], (
        "model should score a plain, HTTPS, bare-domain URL as more likely "
        "legitimate than an obviously phishing-style URL"
    )
