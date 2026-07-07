"""M4 tests: seeded determinism, dominance consistency, demand moment match."""

import numpy as np
import pandas as pd

from src.config import load_params
from src.simulate.demand import sample_demand_paths
from src.simulate.des import run_replication


def _fsku(mu=10.0, spread=5.0, days=28):
    return pd.DataFrame({
        "date": pd.date_range("2016-05-01", periods=days),
        "p50": mu, "p90": mu + 0.78 * spread, "p95": mu + spread,
    })


def _econ():
    return pd.Series({"holding_cost_day": 0.05, "ordering_cost": 50.0,
                      "stockout_penalty": 10.0})


def _policy(s=60.0, cap=120.0):
    return {"mu_day": 10.0, "sigma_day": 3.0, "safety_stock": s - 50.0,
            "reorder_point_s": s, "order_up_to_S": cap, "eoq": 60.0,
            "service_level": 0.95}


def test_demand_sampler_matches_mean():
    rng = np.random.default_rng(7)
    paths = sample_demand_paths(_fsku(), 4000, rng)
    assert abs(paths.mean() - 10.0) < 0.2
    assert (paths >= 0).all()


def test_replication_is_deterministic_given_seed():
    params = load_params()
    paths = sample_demand_paths(_fsku(), 1, np.random.default_rng(3))[0]
    kpis = [
        run_replication(paths, _policy(), _econ(), params["inventory"], 70.0,
                        np.random.default_rng(42))
        for _ in range(2)
    ]
    assert kpis[0] == kpis[1]


def test_more_safety_stock_gives_higher_fill_rate():
    """Obvious dominance: s=90 must beat s=20 on service, averaged over reps."""
    params = load_params()
    inv = params["inventory"]
    rng = np.random.default_rng(5)
    paths = sample_demand_paths(_fsku(), 100, rng)

    def avg_fill(policy):
        return np.mean([
            run_replication(paths[i], policy, _econ(), inv,
                            policy["reorder_point_s"] + 10,
                            np.random.default_rng([9, i]))["fill_rate"]
            for i in range(len(paths))
        ])

    assert avg_fill(_policy(s=90.0, cap=150.0)) > avg_fill(_policy(s=20.0, cap=80.0))
