#!/usr/bin/env python3
"""Project Management CLI for Phishing Detection ML.

Simplifies running, testing, training, and containerizing the application:

  python manage.py runserver [port]   -> Start web dashboard & API (default: 8000)
  python manage.py docker             -> Build & run with Docker Compose
  python manage.py test               -> Run automated pytest test suite
  python manage.py train              -> Retrain the production Random Forest model
  python manage.py scan <url>         -> Scan a URL directly from the terminal
"""

import os
import sys
import subprocess
from pathlib import Path

MODEL_FILE = Path("models_saved/final_model.joblib")


def ensure_model_exists():
    """Checks if final_model.joblib exists; if not, automatically trains it."""
    if not MODEL_FILE.exists():
        print("\n[setup] 'models_saved/final_model.joblib' not found.")
        print("[setup] Training the Random Forest model automatically (~60s)...")
        cmd = [sys.executable, "-m", "src.models.train_augmented"]
        ret = subprocess.call(cmd)
        if ret != 0:
            print("[error] Automated model training failed. Please inspect the output above.")
            sys.exit(ret)
        print("[setup] Model trained successfully!\n")


def cmd_runserver(args):
    """Starts the FastAPI web server with live reload."""
    ensure_model_exists()

    host = "127.0.0.1"
    port = 8000

    for arg in args:
        if ":" in arg:
            h, p = arg.split(":", 1)
            host = h or host
            if p.isdigit():
                port = int(p)
        elif arg.isdigit():
            port = int(arg)

    print("\n" + "=" * 65)
    print("PHISHING DETECTION ML SYSTEM")
    print("=" * 65)
    print(f"Interactive Web Dashboard: http://{host}:{port}/")
    print(f"Interactive API Swagger:   http://{host}:{port}/docs")
    print(f"Health Check:              http://{host}:{port}/health")
    print("=" * 65)
    print("Press CTRL+C to stop the server.\n")

    import uvicorn
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


def cmd_docker(args):
    """Builds and runs the application using Docker."""
    ensure_model_exists()
    print("\nStarting with Docker Compose...")
    cmd = ["docker", "compose", "up", "--build"] + args
    try:
        sys.exit(subprocess.call(cmd))
    except FileNotFoundError:
        print("[docker] 'docker compose' not found. Falling back to plain docker run...")
        subprocess.check_call(["docker", "build", "-t", "phishing-detection-ml", "."])
        subprocess.call(["docker", "run", "-it", "--rm", "-p", "8000:8000", "phishing-detection-ml"])


def cmd_test(args):
    """Runs the test suite."""
    ensure_model_exists()
    print("\nRunning pytest test suite...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"] + args
    sys.exit(subprocess.call(cmd))


def cmd_train(args):
    """Retrains the production Random Forest model."""
    print("\nTraining production model...")
    cmd = [sys.executable, "-m", "src.models.train_augmented"] + args
    sys.exit(subprocess.call(cmd))


def cmd_scan(args):
    """Quick terminal scanner for a URL without opening browser."""
    if not args:
        print("Usage: python manage.py scan <url>")
        sys.exit(1)

    url = args[0]
    ensure_model_exists()

    import joblib
    import pandas as pd
    from src.features.extract import FEATURE_NAMES, extract_features

    model = joblib.load(MODEL_FILE)
    features = extract_features(url)
    X = pd.DataFrame([features])[FEATURE_NAMES]
    proba_legit = model.predict_proba(X)[:, 1][0]
    proba_phish = float(1.0 - proba_legit)
    verdict = "PHISHING" if proba_phish >= 0.5 else "LEGITIMATE"

    print("\n" + "-" * 55)
    print(f"URL:        {url}")
    print(f"Verdict:    {verdict}")
    print(f"Risk Score: {proba_phish * 100:.2f}% phishing probability")
    print("-" * 55 + "\n")


def print_help():
    print("""
===========================================================
  Phishing Detection ML - Command Line Utility
===========================================================

Usage: python manage.py <command> [options]

Commands:
  runserver [port]   Start web UI & API locally (default: 8000)
                     Alias: run, start, serve
                     Examples:
                       python manage.py runserver
                       python manage.py runserver 8080

  docker             Build & start the app in Docker Compose
                     (accessible at http://localhost:8000/)

  test               Run the 34 automated unit and integration tests

  train              Retrain the Random Forest model on 238k samples

  scan <url>         Quickly inspect any link directly in the terminal
                     Example:
                       python manage.py scan https://www.google.com
===========================================================
""")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd in ("runserver", "run", "start", "serve"):
        cmd_runserver(args)
    elif cmd in ("docker", "docker-run", "compose"):
        cmd_docker(args)
    elif cmd == "test":
        cmd_test(args)
    elif cmd == "train":
        cmd_train(args)
    elif cmd in ("scan", "predict", "check"):
        cmd_scan(args)
    else:
        print(f"Unknown command: '{cmd}'")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
