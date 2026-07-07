"""Evidently drift report: training-window features (reference) vs the
served window (current). Writes reports/drift_report.html and a machine-
readable summary consumed by the retrain trigger.

--inject-drift shifts the current window's prices/demand to demonstrate a
full retrain cycle (clearly labeled as synthetic in the summary).
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import REPO_ROOT, load_contract, load_params, processed_dir
from src.ingest.features import FEATURES_FILE
from src.simulate.run import KPIS_FILE, RECS_FILE

SUMMARY_FILE = REPO_ROOT / "reports" / "monitoring_summary.json"

DRIFT_COLS = ["units", "sell_price", "price_rel", "lag_28", "rmean_28_7",
              "rmean_28_28", "snap", "is_event"]


def _drift_share(snapshot_dict: dict) -> tuple[float, int]:
    """Extract (share, count) of drifted columns from the Evidently snapshot."""
    for metric in snapshot_dict.get("metrics", []):
        if "DriftedColumnsCount" in str(metric.get("metric_name", "")):
            value = metric.get("value", {})
            return float(value.get("share", 0.0)), int(value.get("count", 0))
    raise KeyError("DriftedColumnsCount not found in Evidently snapshot")


def simulated_fill_rate(params: dict) -> float:
    """Mean simulated fill rate of the recommended policy across SKUs."""
    d = processed_dir(params)
    kpis = pd.read_parquet(d / KPIS_FILE)
    recs = pd.read_parquet(d / RECS_FILE)[["item_id", "store_id", "recommended_policy"]]
    chosen = kpis.merge(
        recs, left_on=["item_id", "store_id", "policy"],
        right_on=["item_id", "store_id", "recommended_policy"],
    )
    return float(chosen["fill_rate_mean"].mean())


def run_drift(params: dict, inject_drift: bool = False) -> dict:
    fc_horizon = params["forecast"]["horizon"]
    df = pd.read_parquet(processed_dir(params) / FEATURES_FILE)
    origin = df["date"].max() - pd.Timedelta(days=fc_horizon)
    # Reference = the 8 weeks right before the origin, not the full history:
    # a full-history reference makes K-S fire on seasonality alone (huge N,
    # different season mix) and the trigger would always be on.
    ref_start = origin - pd.Timedelta(days=56)
    reference = df[(df["date"] > ref_start) & (df["date"] <= origin)][DRIFT_COLS]
    current = df[df["date"] > origin][DRIFT_COLS].copy()

    if inject_drift:
        current["sell_price"] *= 1.5   # synthetic price shock
        current["units"] = (current["units"] * 2 + 3).astype(current["units"].dtype)
        current["price_rel"] *= 1.5

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    snapshot.save_html(str(REPO_ROOT / "reports" / "drift_report.html"))
    share, count = _drift_share(snapshot.dict())

    fill = simulated_fill_rate(params)
    mon = params["monitoring"]
    summary = {
        "feature_contract_version": load_contract()["version"],
        "drift_share": round(share, 4),
        "drifted_columns": count,
        "drift_threshold": mon["drift_share_threshold"],
        "simulated_fill_rate": round(fill, 4),
        "fill_rate_floor": mon["fill_rate_floor"],
        "synthetic_drift_injected": inject_drift,
        "retrain_needed": share > mon["drift_share_threshold"] or fill < mon["fill_rate_floor"],
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inject-drift", action="store_true",
                    help="synthetically shift the current window (demo of the retrain cycle)")
    args = ap.parse_args()
    summary = run_drift(load_params(), inject_drift=args.inject_drift)
    print(json.dumps(summary, indent=2))
