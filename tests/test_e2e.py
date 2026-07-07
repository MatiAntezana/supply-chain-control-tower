"""E2E integration: the five layers agree with each other on the real artifacts.

Verifies the data contracts across layer boundaries: every SKU flows
forecast -> policy -> simulation -> recommendation without loss, and the
recommendation is exactly the cost-minimizing simulated policy.
"""

import pandas as pd
import pytest

from src.config import load_params, processed_dir


@pytest.fixture(scope="module")
def artifacts():
    params = load_params()
    d = processed_dir(params)
    needed = ["forecast.parquet", "policies.parquet", "simulation_kpis.parquet",
              "recommendations.parquet"]
    if not all((d / f).exists() for f in needed):
        pytest.skip("run `make pipeline` first")
    return {f.split(".")[0]: pd.read_parquet(d / f) for f in needed}


def _sku_set(df):
    return set(map(tuple, df[["item_id", "store_id"]].drop_duplicates().to_numpy()))


def test_no_sku_lost_between_layers(artifacts):
    skus = _sku_set(artifacts["forecast"])
    assert _sku_set(artifacts["policies"]) == skus
    assert _sku_set(artifacts["simulation_kpis"]) == skus
    assert _sku_set(artifacts["recommendations"]) == skus


def test_every_sku_simulated_under_all_policies(artifacts):
    counts = artifacts["simulation_kpis"].groupby(["item_id", "store_id"])["policy"].nunique()
    assert (counts == 3).all()


def test_recommendation_is_cost_minimizer(artifacts):
    kpis = artifacts["simulation_kpis"]
    best = kpis.loc[kpis.groupby(["item_id", "store_id"])["total_cost_mean"].idxmin()]
    best = best.set_index(["item_id", "store_id"])["policy"]
    recs = artifacts["recommendations"].set_index(["item_id", "store_id"])["recommended_policy"]
    assert (recs.sort_index() == best.sort_index()).all()


def test_policies_are_sane(artifacts):
    p = artifacts["policies"]
    assert (p["order_up_to_S"] > p["reorder_point_s"]).all()
    assert (p["safety_stock"] >= 0).all()
    assert (p["milp_status"] == "Optimal").all()
