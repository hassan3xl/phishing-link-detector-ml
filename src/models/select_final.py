"""Phase 5: locks in the final model selection as a dedicated artifact.

Selected in notebooks/03_evaluation.ipynb: untuned XGBoost
(n_estimators=300, max_depth=6, learning_rate=0.1), deployed at a
recall-favoring threshold of 0.25 (phishing = positive class) instead of the
default 0.5. See the README's Evaluation section for the full justification.

Writes:
- models_saved/final_model.joblib   (copy of models_saved/xgboost_v1.joblib)
- models_saved/final_model_metadata.json (threshold, feature order, notes)

Both are small/reproducible enough to commit, decoupling Phase 7 (the API)
from needing to know which of the 4 candidate models "xgboost_v1" was.

Run: python -m src.models.select_final
"""

import json
import shutil
from pathlib import Path

from src.features.extract import FEATURE_NAMES

MODELS_DIR = Path("models_saved")
SOURCE_MODEL = MODELS_DIR / "xgboost_v1.joblib"
FINAL_MODEL = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "final_model_metadata.json"

THRESHOLD = 0.25  # phishing (label=0) is the positive class; see Phase 5


def main():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"{SOURCE_MODEL} not found - run `python -m src.models.train` first.")

    shutil.copyfile(SOURCE_MODEL, FINAL_MODEL)

    metadata = {
        "model_type": "xgboost",
        "source_artifact": str(SOURCE_MODEL),
        "hyperparameters": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
        "feature_names": FEATURE_NAMES,
        "threshold": THRESHOLD,
        "positive_class": "phishing (label=0)",
        "prediction_rule": (
            "phishing_probability = 1 - model.predict_proba(X)[:, 1]; "
            "predict 'phishing' if phishing_probability >= threshold else 'legitimate'"
        ),
        "selection_rationale": (
            "Best phishing-recall (0.9941) and F1 (0.9969) of all 4 candidates at the "
            "default threshold, edging out the CV-ROC-AUC-tuned variant (0.9937 recall) "
            "which optimized a threshold-independent metric that didn't carry over to the "
            "deployed operating point. Threshold lowered from 0.5 to 0.25 via out-of-fold "
            "CV on train (F2-optimal region), trading 18 extra false positives for 12 fewer "
            "missed phishing URLs on test - justified by the asymmetric cost of a missed "
            "phishing URL vs. a user-friction false alarm. Full writeup: "
            "notebooks/03_evaluation.ipynb and the README's Evaluation section."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"[select_final] wrote {FINAL_MODEL} and {METADATA_PATH}")


if __name__ == "__main__":
    main()
