"""M5 tests: API contract over the real artifacts (skips if pipeline not run)."""

import pytest
from fastapi.testclient import TestClient

from src.config import load_params, processed_dir


@pytest.fixture(scope="module")
def client():
    params = load_params()
    if not (processed_dir(params) / "recommendations.parquet").exists():
        pytest.skip("run `make pipeline` first")
    from src.serve.api import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_skus"] > 0


def test_recommend_contract(client):
    sku = client.get("/skus").json()[0]
    r = client.get("/recommend", params={"item_id": sku["item_id"], "store_id": sku["store_id"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == load_params()["forecast"]["horizon"]
    assert body["policy"]["order_up_to_S"] > body["policy"]["reorder_point_s"]
    policies = {k["policy"] for k in body["simulated_kpis"]}
    assert {"naive_reorder", "sS_quantile", "milp"} <= policies
    for k in body["simulated_kpis"]:
        assert 0.0 <= k["fill_rate_mean"] <= 1.0
        assert k["total_cost_mean"] >= 0.0


def test_recommend_unknown_sku_404(client):
    r = client.get("/recommend", params={"item_id": "NOPE", "store_id": "CA_1"})
    assert r.status_code == 404
