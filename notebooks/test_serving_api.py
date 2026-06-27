import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


API_URL = "http://127.0.0.1:8000"
FEATURES_PATH = Path("../data/04_feature/features_inference.parquet")


def get_json(path: str) -> dict:
    with urlopen(f"{API_URL}{path}") as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> dict:
    request = Request(
        f"{API_URL}{path}",
        data=json.dumps(payload, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    print("Testing serving API...")
    print(f"API URL: {API_URL}")
    print()

    print("1. Checking /health")
    health = get_json("/health")
    print(json.dumps(health, indent=2))

    if health.get("status") != "ok":
        raise RuntimeError(f"API health is not ok: {health}")

    print()
    print("2. Checking /metadata")
    metadata = get_json("/metadata")
    print(json.dumps(metadata, indent=2))

    print()
    print("3. Loading sample features")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    features = pd.read_parquet(FEATURES_PATH).head(3)
    print(f"Loaded {len(features)} rows from {FEATURES_PATH}")

    print()
    print("4. Testing /predict-one")
    one_payload = {
        "features": features.iloc[0].to_dict()
    }

    one_prediction = post_json("/predict-one", one_payload)
    print(json.dumps(one_prediction, indent=2))

    required_single_keys = {
        "prediction",
        "probability_default",
        "model_uri",
        "model_status",
    }

    missing_keys = required_single_keys - set(one_prediction)
    if missing_keys:
        raise RuntimeError(f"/predict-one missing keys: {missing_keys}")

    print()
    print("5. Testing /predict batch")
    batch_payload = {
        "rows": features.to_dict(orient="records")
    }

    batch_prediction = post_json("/predict", batch_payload)
    print(json.dumps(batch_prediction, indent=2))

    if batch_prediction.get("n_rows") != len(features):
        raise RuntimeError(
            f"Expected {len(features)} predictions, got {batch_prediction.get('n_rows')}"
        )

    predictions = batch_prediction.get("predictions", [])
    if len(predictions) != len(features):
        raise RuntimeError(
            f"Expected {len(features)} prediction rows, got {len(predictions)}"
        )

    print()
    print("Serving API test passed.")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as exc:
        print(f"HTTP error: {exc.code}")
        print(exc.read().decode("utf-8"))
        raise
    except URLError as exc:
        print("Could not connect to the API.")
        print("Make sure the server is running:")
        print("uv run uvicorn mortgage_default.serving.app:app --reload --host 0.0.0.0 --port 8000")
        raise