"""Tests for the Phase 10 Lambda entrypoint (src.api.main.handler).

Invokes the Mangum-wrapped handler directly with synthetic API Gateway HTTP
API (v2 payload format) events - the same event shape API Gateway sends in
production - rather than requiring a running container. Complements the
manual Docker + Runtime Interface Emulator verification described in the
README's Phase 10 section.
"""

import json

from src.api.main import handler


def _http_api_event(method: str, path: str, body: str | None = None) -> dict:
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"content-type": "application/json"},
        "requestContext": {
            "http": {
                "method": method, "path": path, "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1", "userAgent": "pytest",
            },
            "domainName": "test.execute-api.us-east-1.amazonaws.com",
            "requestId": "test-request-id",
            "routeKey": f"{method} {path}",
            "stage": "$default",
            "time": "09/Aug/2026:00:00:00 +0000",
            "timeEpoch": 0,
        },
        "body": body,
        "isBase64Encoded": False,
    }


def test_handler_exists():
    assert handler is not None, "mangum must be installed for the Lambda handler to build"


def test_handler_health():
    resp = handler(_http_api_event("GET", "/health"), None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_handler_predict_phishing_style_url():
    event = _http_api_event(
        "POST", "/predict", body=json.dumps({"url": "http://paypal-secure-login.xyz/confirm?id=12345"})
    )
    resp = handler(event, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["prediction"] == "phishing"


def test_handler_predict_legitimate_url():
    event = _http_api_event("POST", "/predict", body=json.dumps({"url": "https://www.google.com"}))
    resp = handler(event, None)
    assert resp["statusCode"] == 200
    body = json.loads(resp["body"])
    assert body["prediction"] == "legitimate"


def test_handler_predict_empty_url_is_422():
    event = _http_api_event("POST", "/predict", body=json.dumps({"url": ""}))
    resp = handler(event, None)
    assert resp["statusCode"] == 422
