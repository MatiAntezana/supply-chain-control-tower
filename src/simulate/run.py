"""M4 orchestrator: Monte-Carlo validation of every candidate policy.

For each SKU, three policies face the same stochastic demand paths and
lead-time streams (common random numbers -> fair, low-variance comparison):
  - naive_reorder : reorder point = mean lead-time demand, no safety stock
  - sS_quantile   : (s,S) from the forecast quantiles (M3 closed form)
  - milp          : the MILP order schedule (M3 exact)
Also sweeps service levels to trace the cost <-> service frontier.

Outputs: data/processed/simulation_kpis.parquet,
         data/processed/recommendations.parquet,
         reports/cost_service_curve.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_params, processed_dir
from src.forecast.train import FORECAST_FILE
from src.optimize.economics import sku_economics
from src.optimize.policy import initial_inventory, newsvendor_policy
from src.simulate.demand import sample_demand_paths
from src.simulate.des import run_replication

KPIS_FILE = "simulation_kpis.parquet"
RECS_FILE = "recommendations.parquet"


def naive_policy(pol: dict) -> dict:
    """Baseline: same EOQ, but reorder at mean lead-time demand (no tail)."""
    naive = dict(pol)
    naive["safety_stock"] = 0.0
    naive["reorder_point_s"] = pol["reorder_point_s"] - pol["safety_stock"]
    naive["order_up_to_S"] = pol["order_up_to_S"] - pol["safety_stock"]
    return naive


def _simulate_policy(demand_paths, pol, sched, econ, inv, i0, seed_child) -> pd.DataFrame:
    rows = []
    for rep in range(demand_paths.shape[0]):
        rng = np.random.default_rng([seed_child, rep])  # same LT stream per rep
        rows.append(run_replication(
            demand_paths[rep], pol, econ, inv, i0, rng, orders_schedule=sched
        ))
    return pd.DataFrame(rows)


def _aggregate(reps: pd.DataFrame) -> dict:
    return {
        "fill_rate_mean": reps["fill_rate"].mean(),
        "fill_rate_p10": reps["fill_rate"].quantile(0.10),
        "stockout_days_mean": reps["stockout_days"].mean(),
        "holding_cost_mean": reps["holding_cost"].mean(),
        "ordering_cost_mean": reps["ordering_cost"].mean(),
        "stockout_cost_mean": reps["stockout_cost"].mean(),
        "total_cost_mean": reps["total_cost"].mean(),
        "total_cost_p90": reps["total_cost"].quantile(0.90),
    }


def run_simulation(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = params["inventory"]
    sim = params["simulation"]
    seed = params["seed"]
    forecast = pd.read_parquet(processed_dir(params) / FORECAST_FILE)
    policies = pd.read_parquet(processed_dir(params) / "policies.parquet")
    econ_df = sku_economics(forecast, params).set_index(["item_id", "store_id"])

    kpi_rows, curve_rows = [], []
    for idx, prow in policies.reset_index(drop=True).iterrows():
        item, store = prow["item_id"], prow["store_id"]
        fsku = forecast[(forecast["item_id"] == item) & (forecast["store_id"] == store)].sort_values("date")
        econ = econ_df.loc[(item, store)]
        pol = {k: prow[k] for k in
               ["mu_day", "sigma_day", "safety_stock", "reorder_point_s",
                "order_up_to_S", "eoq", "service_level"]}
        i0 = initial_inventory(pol)

        rng_demand = np.random.default_rng([seed, idx])
        paths = sample_demand_paths(fsku, sim["n_replications"], rng_demand)

        candidates = {"naive_reorder": (naive_policy(pol), None), "sS_quantile": (pol, None)}
        sched = prow.get("milp_orders_json")
        if sched is not None:
            candidates["milp"] = (pol, np.asarray(json.loads(sched)))

        for name, (p, schedule) in candidates.items():
            reps = _simulate_policy(paths, p, schedule, econ, inv, i0, seed + idx)
            kpi_rows.append({"item_id": item, "store_id": store, "policy": name,
                             **_aggregate(reps)})

        for level in sim["service_levels_sweep"]:
            p = newsvendor_policy(fsku, econ, params, service_level=level)
            reps = _simulate_policy(paths, p, None, econ, inv, initial_inventory(p), seed + idx)
            curve_rows.append({"item_id": item, "store_id": store, "service_level": level,
                               **_aggregate(reps)})

    kpis = pd.DataFrame(kpi_rows)
    kpis.to_parquet(processed_dir(params) / KPIS_FILE, index=False)

    curve = (
        pd.DataFrame(curve_rows)
        .groupby("service_level")[["fill_rate_mean", "holding_cost_mean",
                                   "stockout_cost_mean", "total_cost_mean"]]
        .mean()
        .reset_index()
    )
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    curve.to_csv(REPO_ROOT / "reports" / "cost_service_curve.csv", index=False)

    best = (
        kpis.loc[kpis.groupby(["item_id", "store_id"])["total_cost_mean"].idxmin()]
        [["item_id", "store_id", "policy", "fill_rate_mean", "total_cost_mean"]]
        .rename(columns={"policy": "recommended_policy"})
    )
    recs = best.merge(policies, on=["item_id", "store_id"])
    recs.to_parquet(processed_dir(params) / RECS_FILE, index=False)
    return kpis, curve


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    params = load_params()
    kpis, curve = run_simulation(params)
    print(kpis.groupby("policy")[["fill_rate_mean", "total_cost_mean"]].mean().round(3).to_string())
    print("\ncost <-> service frontier:\n" + curve.round(3).to_string(index=False))
