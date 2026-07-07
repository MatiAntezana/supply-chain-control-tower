"""M3 tests: MILP optimality on a known tiny instance, policy monotonicity."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_params
from src.optimize.milp import solve_milp
from src.optimize.policy import newsvendor_policy, round_order


@pytest.fixture()
def params():
    return load_params()


def _tiny_forecast(mu=10.0, spread=4.0):
    dates = pd.date_range("2016-05-01", periods=28)
    return pd.DataFrame({
        "date": dates,
        "p50": mu, "p90": mu + spread * 0.78, "p95": mu + spread,
        "sell_price": 5.0,
    })


def _econ():
    return pd.Series({
        "price": 5.0, "unit_cost": 3.5, "holding_cost_day": 0.1,
        "stockout_penalty": 7.5, "ordering_cost": 50.0,
    })


def test_milp_known_optimum():
    """1 SKU, T=4, lead time 0, demand 5/day, I0=10: must order exactly once.

    Ordering costs 50; holding 0.1/unit-day. Covering days 2-3 needs 10 more
    units. Two orders would cost 100 > any holding saving, so the optimum is
    one order of 10. Optimal timing: order at t=2 (arrive same day, lt=0).
    Cost = 50 + holding on end-of-day inventories [5,0,5,0] = 50 + 1.0.
    """
    res = solve_milp(
        demand=np.full((1, 4), 5.0),
        safety_stock=np.zeros(1),
        initial_inventory=np.array([10.0]),
        holding_cost_day=np.array([0.1]),
        ordering_cost=np.array([50.0]),
        lead_time=0, min_order=0, multiple=1, capacity=None,
    )
    assert res["status"] == "Optimal"
    assert res["orders"].sum() == 10
    assert (res["orders"][0] > 0).sum() == 1
    assert res["objective"] == pytest.approx(51.0)


def test_milp_respects_min_order_and_multiple():
    res = solve_milp(
        demand=np.full((1, 6), 3.0),
        safety_stock=np.zeros(1),
        initial_inventory=np.array([3.0]),
        holding_cost_day=np.array([0.01]),
        ordering_cost=np.array([10.0]),
        lead_time=1, min_order=10, multiple=5, capacity=None,
    )
    assert res["status"] == "Optimal"
    placed = res["orders"][res["orders"] > 0]
    assert (placed >= 10).all()
    assert (placed % 5 == 0).all()


def test_reorder_point_increases_with_service_level(params):
    f = _tiny_forecast()
    lo = newsvendor_policy(f, _econ(), params, service_level=0.90)
    hi = newsvendor_policy(f, _econ(), params, service_level=0.99)
    assert hi["reorder_point_s"] > lo["reorder_point_s"]
    assert hi["safety_stock"] > lo["safety_stock"]


def test_round_order():
    assert round_order(0, 10, 5) == 0
    assert round_order(3, 10, 5) == 10
    assert round_order(12, 10, 5) == 15
