"""Prediction logging: on every request, append a JSON line recording the
input URL, extracted features, prediction, probability, and timestamp.

This is deliberately simple (a local JSONL file, not a database) but sets up
the real production concept it's standing in for: **drift monitoring**.
Logged predictions over time let you later check whether the *input*
distribution (e.g. average URL length, TLD mix of incoming traffic) or the
*prediction* distribution (e.g. rising phishing-flag rate) is drifting away
from what the model was trained on - which is exactly the kind of shift that
would justify retraining. This file is the foundation that a real drift-
detection job would read from; building that job itself is out of scope for
v1 (see README Future Work).

Path is overridable via PREDICTION_LOG_PATH (Phase 10: AWS Lambda's
filesystem is read-only except /tmp, so the Lambda image sets this to
/tmp/predictions.jsonl - written logs there are ephemeral per-execution-
environment/cold-start, not a durable store; see the README's Phase 10
section for how that's handled for real monitoring).
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

LOG_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))
_lock = Lock()  # append-only writes from concurrent requests shouldn't interleave


def log_prediction(url: str, features: dict, prediction: str, phishing_probability: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "features": features,
        "prediction": prediction,
        "phishing_probability": phishing_probability,
    }
    with _lock:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
