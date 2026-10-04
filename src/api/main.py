"""Phase 7: FastAPI prediction service. Also serves as the Phase 10 Lambda
entrypoint via the `handler` object at the bottom of this file.

Reuses `extract_features` from `src.features.extract` UNCHANGED - this is
the train/serve-skew safeguard named in the README: reimplementing feature
logic slightly differently inside an API is a real, common production ML
bug class, so this module imports the exact same function used to build
`data/train_features.csv` / `data/test_features.csv` in Phase 3, rather than
recomputing anything by hand.

Run locally: uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs

MODEL_ARTIFACT_PATH (env var, optional): overrides where the model artifact
is loaded from. Defaults to the relative path used by Phase 7's plain-Docker
image (models_saved/final_model.joblib, run from /app). Phase 10's Lambda
container image sets this to an absolute path under /var/task (where the
AWS Lambda Python base image places the deployment package) - see
Dockerfile.lambda. The metadata JSON is always read from the same directory
as the model file (final_model_metadata.json alongside final_model.joblib).
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.api.prediction_log import log_prediction
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse
from src.features.extract import FEATURE_NAMES, extract_features

STATIC_DIR = Path(__file__).parent / "static"

MODEL_PATH = Path(os.environ.get("MODEL_ARTIFACT_PATH", "models_saved/final_model.joblib"))
METADATA_PATH = MODEL_PATH.parent / "final_model_metadata.json"

_state = {"model": None, "threshold": None, "model_version": None}

KNOWN_LIMITATION_NOTE = (
    "KNOWN LIMITATION: this model was trained on PhiUSIIL, whose legitimate-URL "
    "class contains 0 URLs with any path component (all are bare homepage root "
    "URLs). The model has partly learned 'has a path -> phishing' as a shortcut "
    "and will misclassify many ordinary legitimate deep-linked URLs (articles, "
    "product pages, docs). See the README's Limitations section before relying "
    "on this endpoint for real traffic."
)


def _load_model_into_state() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found - run `python -m src.models.train` and "
            "`python -m src.models.select_final` first."
        )
    _state["model"] = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    _state["threshold"] = metadata["threshold"]
    _state["model_version"] = metadata.get("source_artifact", "final_model")


def _ensure_model_loaded() -> None:
    """Lazy-load guard used by the route handlers. A no-op when `lifespan`
    has already loaded the model (the normal uvicorn/Phase-7-Docker path);
    does the actual load on first call otherwise (Phase 10's Lambda path,
    where `lifespan` is disabled - see the `handler` setup at the bottom of
    this file - so the first request in a fresh execution environment loads
    the model once, and subsequent warm invocations reuse it via this
    module-level `_state` persisting across invocations in the same
    environment, the standard Lambda cold/warm-start pattern)."""
    if _state.get("model") is None:
        _load_model_into_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_into_state()
    yield
    # Reset values rather than _state.clear(): clear() removes the dict's
    # KEYS entirely, and _ensure_model_loaded()/health() read _state["model"]
    # by key - a bug caught by tests/test_lambda_handler.py running after
    # tests/test_api.py's TestClient exits its lifespan in the same pytest
    # process: a bare .clear() left a later direct handler() call hitting a
    # KeyError instead of gracefully re-loading.
    _state["model"] = None
    _state["threshold"] = None
    _state["model_version"] = None


app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "Classifies a URL as phishing or legitimate using lexical/structural "
        "features engineered from the raw URL string (no page content, no "
        "WHOIS/domain-age lookups - see README Scope). "
        f"\n\n{KNOWN_LIMITATION_NOTE}"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Phishing URL Detection System</h1><p>UI file not found.</p>")


@app.get("/health", response_model=HealthResponse)
def health():
    _ensure_model_loaded()
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
        model_version=str(_state.get("model_version")),
    )


from urllib.parse import urlparse


def normalize_url_for_inference(raw_url: str) -> str:
    """Normalizes user input URLs to bridge real-world input styles with the
    PhiUSIIL training distribution.

    1. Removes trailing root slash if path is bare ('https://google.com/' -> 'https://google.com').
       In PhiUSIIL, legitimate training URLs were collected as bare root domains without trailing slashes.
    2. Handles apex domains lacking a 'www.' prefix ('https://google.com' -> 'https://www.google.com').
       In PhiUSIIL, 100% of legitimate training domains were indexed with 'www.'; without this normalization,
       a bare apex domain spuriously trips the 'num_subdomains == 0' phishing decision rule.
    """
    url = raw_url.strip()
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)

    # 1. Normalize trailing root slash: https://domain.tld/ -> https://domain.tld
    if parsed.path == "/" and not parsed.query and not parsed.fragment:
        url = url[:-1]
        parsed = urlparse(url)

    # 2. Normalize apex domains without subdomains (e.g. google.com -> www.google.com)
    host = parsed.hostname or ""
    if host.count(".") == 1 and not host.replace(".", "").isdigit() and not host.startswith("www."):
        netloc = parsed.netloc
        if "@" in netloc:
            userinfo, rest = netloc.split("@", 1)
            new_netloc = f"{userinfo}@www.{rest}"
        else:
            new_netloc = f"www.{netloc}"
        url = parsed._replace(netloc=new_netloc).geturl()

    return url


VERIFIED_AUTHORITY_DOMAINS = {
    "claude.ai", "anthropic.com", "chatgpt.com", "openai.com", "perplexity.ai",
    "google.com", "youtube.com", "apple.com", "microsoft.com", "github.com",
    "gitlab.com", "stackoverflow.com", "wikipedia.org", "wikimedia.org", "amazon.com",
    "linkedin.com", "twitter.com", "x.com", "reddit.com", "facebook.com", "instagram.com",
    "discord.com", "slack.com", "dropbox.com", "notion.so", "figma.com", "canva.com",
    "spotify.com", "netflix.com", "medium.com", "quora.com", "substack.com",
    "arxiv.org", "nih.gov", "nature.com"
}


def get_apex_domain(host: str) -> str:
    host = (host or "").lower().strip()
    if not host or host.replace(".", "").isdigit():
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        two_part_tlds = {"co.uk", "com.au", "co.nz", "gov.uk", "ac.uk", "org.uk", "co.jp", "com.br"}
        if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in two_part_tlds:
            return f"{parts[-3]}.{parts[-2]}.{parts[-1]}"
        return f"{parts[-2]}.{parts[-1]}"
    return host


def check_domain_authority_guardrail(raw_url: str, features: dict) -> tuple[bool, str]:
    """Evaluates whether an input URL belongs to a verified clean authority domain.

    Protects legitimate deep links (e.g. https://claude.ai/chat/ or https://chatgpt.com/c/...)
    from the PhiUSIIL zero-path training dataset bias, while maintaining strict defense:
    - Never bypasses if host is an IP address
    - Never bypasses if URL has userinfo spoofing (@)
    - Never bypasses if TLD is in high-risk list
    - Only matches the verified registered apex domain (preventing subdomain spoofing like claude.ai.attacker.xyz)
    """
    if features.get("is_ip_address", 0) == 1:
        return False, ""
    if features.get("count_at", 0) > 0:
        return False, ""
    if features.get("tld_risk_flag", 0) == 1:
        return False, ""

    p = urlparse(raw_url if "://" in raw_url else "http://" + raw_url)
    host = p.hostname or ""
    apex = get_apex_domain(host)
    if apex in VERIFIED_AUTHORITY_DOMAINS:
        return True, apex
    return False, ""


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _ensure_model_loaded()
    model = _state.get("model")
    threshold = _state.get("threshold")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    normalized_url = normalize_url_for_inference(request.url)
    features = extract_features(normalized_url)

    # Check verified authority domain guardrail (protects deep links on verified domains from dataset bias)
    is_authority, authority_domain = check_domain_authority_guardrail(request.url, features)

    if is_authority:
        proba_phish = 0.0
        prediction = "legitimate"
        model_ver = f"{_state['model_version']} [Verified Authority: {authority_domain}]"
        note = f"Verified clean authority domain '{authority_domain}'. Deep-link protected from PhiUSIIL zero-path training bias."
    else:
        X = pd.DataFrame([features])[FEATURE_NAMES]
        proba_legit = model.predict_proba(X)[:, 1][0]
        proba_phish = float(1 - proba_legit)
        prediction = "phishing" if proba_phish >= threshold else "legitimate"
        model_ver = str(_state["model_version"])
        note = None

    log_prediction(request.url, features, prediction, proba_phish)

    return PredictResponse(
        url=request.url,
        prediction=prediction,
        phishing_probability=round(proba_phish, 6),
        threshold=threshold,
        model_version=model_ver,
        features=features,
        note=note,
    )


# Phase 10: AWS Lambda entrypoint (Dockerfile.lambda's CMD points at
# "src.api.main.handler"). Mangum adapts API Gateway HTTP API (v2 payload
# format) events into ASGI calls against the SAME `app` instance used by
# uvicorn locally/in Phase 7's Docker image - no route or handler logic is
# duplicated. `lifespan="off"` because Mangum invokes per-request rather
# than running a persistent event loop that would keep a `lifespan` context
# open across invocations; each route's `_ensure_model_loaded()` call
# handles loading instead (see above). Import is optional so Phase 7's
# plain-Docker/local/CI installs (which don't include mangum) can still
# import this module without it - only the Lambda image installs it.
try:
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None
