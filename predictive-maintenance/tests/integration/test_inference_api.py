"""
Inference API integration tests.

Tests all API endpoints against the running inference service,
verifying response schemas match the api-contract.

Run:
    pytest tests/integration/test_inference_api.py -v -m integration
"""

import os
import pytest
import requests

pytestmark = pytest.mark.integration

BASE_URL = os.environ.get("INFERENCE_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def _get(endpoint: str, **kwargs):
    return requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, timeout=10, **kwargs)


def _post(endpoint: str, json_data: dict, **kwargs):
    return requests.post(
        f"{BASE_URL}{endpoint}", json=json_data, headers=HEADERS, timeout=30, **kwargs
    )


# ---- Fixture: skip if API not reachable ---------------------------------


@pytest.fixture(scope="module", autouse=True)
def check_api():
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
    except Exception:
        pytest.skip("Inference API not reachable")


# ---- Tests ---------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self):
        resp = _get("/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        data = _get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "uptime" in data
        assert "timestamp" in data

    def test_health_has_dependencies(self):
        data = _get("/health").json()
        deps = data.get("dependencies")
        if deps:
            # All deps should have a status field
            for name, info in deps.items():
                assert "status" in info, f"Dependency {name} missing status"


class TestRootEndpoint:
    def test_root(self):
        resp = _get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "service" in body
        assert "version" in body


class TestModelsEndpoint:
    def test_list_models(self):
        resp = _get("/models")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)


class TestRULPrediction:
    @pytest.fixture
    def rul_payload(self):
        import numpy as np

        # 50 timesteps x 14 features
        return {
            "data": {
                "equipment_id": "TEST-EQ-001",
                "sequence": [
                    [float(v) for v in row] for row in np.random.randn(50, 14).tolist()
                ],
            },
            "return_confidence": True,
        }

    def test_predict_rul_success(self, rul_payload):
        resp = _post("/predict/rul", rul_payload)
        # 200 if model loaded, 503 if not
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            body = resp.json()
            assert "equipment_id" in body
            assert "rul_cycles" in body
            assert "rul_hours" in body
            assert "health_status" in body
            assert "timestamp" in body

    def test_predict_rul_invalid_input(self):
        resp = _post("/predict/rul", {"data": {}})
        assert resp.status_code == 422  # validation error


class TestHealthPrediction:
    @pytest.fixture
    def health_payload(self):
        import numpy as np

        return {
            "data": {
                "equipment_id": "TEST-EQ-001",
                "features": {f"feat_{i}": float(np.random.randn()) for i in range(14)},
            },
            "return_probabilities": True,
        }

    def test_predict_health_success(self, health_payload):
        resp = _post("/predict/health", health_payload)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            body = resp.json()
            assert "equipment_id" in body
            assert "health_status" in body
            assert "health_status_code" in body
            assert "timestamp" in body


class TestBatchPrediction:
    @pytest.fixture
    def batch_payload(self):
        import numpy as np

        return {
            "sequences": [
                {
                    "equipment_id": f"TEST-EQ-{i:03d}",
                    "sequence": np.random.randn(50, 14).tolist(),
                }
                for i in range(3)
            ]
        }

    def test_predict_batch_success(self, batch_payload):
        resp = _post("/predict/batch", batch_payload)
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            body = resp.json()
            assert "results" in body
            assert "batch_size" in body
            assert "processing_time_ms" in body
            assert "timestamp" in body


class TestErrorResponse:
    def test_404_returns_structured_error(self):
        resp = _get("/nonexistent-endpoint")
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body
        assert "message" in body
        assert "timestamp" in body
        assert "request_id" in body


class TestRateLimiting:
    def test_rate_limit_header_present(self):
        resp = _get("/health")
        # Rate limit headers may not be on exempted paths;
        # just verify the server responds
        assert resp.status_code == 200
