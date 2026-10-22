from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)
HAS_MODEL = (Path(__file__).resolve().parents[1] / "models" / "detector.joblib").exists()


def test_health():
    assert client.get("/health").json() == {"ok": True}


def test_short_text():
    assert client.post("/predict", json={"text": "too short"}).json()["status"] == "too_short"


def test_metrics_is_dict():
    assert isinstance(client.get("/metrics").json(), dict)


@pytest.mark.skipif(not HAS_MODEL, reason="model not trained yet")
def test_long_text_scores():
    text = "This is a simple sentence about my day at college. " * 8
    result = client.post("/predict", json={"text": text}).json()
    assert result["status"] == "ok" and 0.0 <= result["p_ai"] <= 1.0
    assert result["verdict"] in {"human", "ai", "uncertain"}
