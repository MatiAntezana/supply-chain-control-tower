"""M3 orchestrator: forecast -> (s,S) policy per SKU + MILP per store.

Writes data/processed/policies.parquet (one row per SKU with both policies'
parameters and deterministic costs) and reports/optimize_comparison.csv.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_params, processed_dir, set_seeds
from src.forecast.train import FORECAST_FILE
from src.optimize.economics import sku_economics
from src.optimize.milp import solve_milp
from src.optimize.policy import evaluate_deterministic, initial_inventory, newsvendor_policy

POLICIES_FILE = "policies.parquet"


def run_optimization(params: dict) -> pd.DataFrame:
    inv = params["inventory"]
    forecast = pd.read_parquet(processed_dir(params) / FORECAST_FILE)
    econ_df = sku_economics(forecast, params).set_index(["item_id", "store_id"])

    rows = []
    for store, fstore in forecast.groupby("store_id", observed=True):
        skus, policies, demand_rows = [], [], []
        for item, fsku in fstore.groupby("item_id", observed=True):
            fsku = fsku.sort_values("date")
            econ = econ_df.loc[(item, store)]
            pol = newsvendor_policy(fsku, econ, params)
            skus.append(item)
            policies.append(pol)
            demand_rows.append(fsku["p50"].to_numpy())

        demand = np.vstack(demand_rows)
        ss = np.array([p["safety_stock"] for p in policies])
        i0 = np.array([initial_inventory(p) for p in policies])
        h = np.array([econ_df.loc[(k, store), "holding_cost_day"] for k in skus])
        k_cost = np.array([econ_df.loc[(k, store), "ordering_cost"] for k in skus])

        milp = solve_milp(
            demand, ss, i0, h, k_cost,
            lead_time=int(round(inv["lead_time"]["mean_days"])),
            min_order=inv["min_order_qty"], multiple=inv["order_multiple"],
            capacity=inv["capacity_units"],
        )

        for j, (item, pol) in enumerate(zip(skus, policies)):
            econ = econ_df.loc[(item, store)]
            cost_ss = evaluate_deterministic(demand[j], pol, econ, params)
            cost_milp = evaluate_deterministic(
                demand[j], pol, econ, params, orders_schedule=milp["orders"][j]
            )
            rows.append({
                "item_id": item, "store_id": store, **pol,
                "milp_status": milp["status"],
                "milp_orders_json": json.dumps(milp["orders"][j].tolist()),
                "milp_orders_total": float(milp["orders"][j].sum()),
                "milp_n_orders": int((milp["orders"][j] > 0).sum()),
                **{f"ss_det_{k}": v for k, v in cost_ss.items()},
                **{f"milp_det_{k}": v for k, v in cost_milp.items()},
            })

    out = pd.DataFrame(rows)
    out.to_parquet(processed_dir(params) / POLICIES_FILE, index=False)

    comp = out.groupby("store_id")[
        ["ss_det_total_cost", "milp_det_total_cost", "ss_det_fill_rate", "milp_det_fill_rate"]
    ].agg(["mean", "sum"]).round(2)
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    comp.to_csv(reports / "optimize_comparison.csv")
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    params = load_params()
    set_seeds(params["seed"])
    out = run_optimization(params)
    print(out[["milp_status"]].value_counts().to_string())
    tot = out[["ss_det_total_cost", "milp_det_total_cost"]].sum()
    print(f"deterministic 28d cost — (s,S): {tot.iloc[0]:,.0f} | MILP: {tot.iloc[1]:,.0f}")
