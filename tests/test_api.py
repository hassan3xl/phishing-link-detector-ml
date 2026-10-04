"""Tests for the FastAPI app: /health, /predict happy paths, and validation
failures. Uses starlette's TestClient as a context manager so the app's
`lifespan` (which loads the model) actually runs."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_index_page_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "Phishing URL Detection System" in resp.text


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_input_returns_expected_shape(client):
    resp = client.post("/predict", json={"url": "https://www.google.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in ("phishing", "legitimate")
    assert 0.0 <= body["phishing_probability"] <= 1.0
    assert body["url"] == "https://www.google.com"
    assert "threshold" in body and "model_version" in body


def test_predict_phishing_style_url_flagged_as_phishing(client):
    resp = client.post(
        "/predict", json={"url": "http://paypal-secure-login.xyz/confirm?id=12345"}
    )
    assert resp.status_code == 200
    assert resp.json()["prediction"] == "phishing"


def test_predict_bare_legitimate_url_flagged_as_legitimate(client):
    resp = client.post("/predict", json={"url": "https://www.google.com"})
    assert resp.status_code == 200
    assert resp.json()["prediction"] == "legitimate"


def test_predict_empty_url_is_422(client):
    resp = client.post("/predict", json={"url": ""})
    assert resp.status_code == 422


def test_predict_missing_url_field_is_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_too_long_url_is_422(client):
    resp = client.post("/predict", json={"url": "https://example.com/" + "a" * 3000})
    assert resp.status_code == 422


def test_predict_whitespace_only_url_is_422(client):
    resp = client.post("/predict", json={"url": "   "})
    assert resp.status_code == 422
