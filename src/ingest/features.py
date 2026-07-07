"""Feature engineering per the versioned contract in configs/feature_contract.yaml.

All autoregressive features use lags >= horizon (28d), so a single direct
model can forecast the full horizon with zero leakage and no recursion.
Output: data/processed/features.parquet.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_contract, load_params, processed_dir
from src.ingest.load_m5 import PANEL_FILE

FEATURES_FILE = "features.parquet"


def build_features(panel: pd.DataFrame, params: dict) -> pd.DataFrame:
    fc = params["forecast"]
    df = panel.sort_values(["item_id", "store_id", "date"]).copy()
    g = df.groupby(["item_id", "store_id"], observed=True)["units"]

    for lag in fc["lags"]:
        df[f"lag_{lag}"] = g.shift(lag).astype("float32")

    base_lag = min(fc["lags"])  # 28 = horizon
    shifted = g.shift(base_lag)
    grp = df.groupby(["item_id", "store_id"], observed=True)
    for w in fc["rolling_windows"]:
        df[f"rmean_{base_lag}_{w}"] = (
            shifted.groupby([df["item_id"], df["store_id"]], observed=True)
            .rolling(w, min_periods=w).mean().reset_index(level=[0, 1], drop=True)
        ).astype("float32")
        df[f"rstd_{base_lag}_{w}"] = (
            shifted.groupby([df["item_id"], df["store_id"]], observed=True)
            .rolling(w, min_periods=w).std().reset_index(level=[0, 1], drop=True)
        ).astype("float32")

    price_rm = grp["sell_price"].transform(lambda s: s.rolling(28, min_periods=1).mean())
    df["price_rel"] = (df["sell_price"] / price_rm).astype("float32")
    df["sell_price"] = df["sell_price"].astype("float32")

    # Drop warm-up rows where the longest feature window is not yet available.
    warmup_cols = [f"lag_{max(fc['lags'])}", f"rmean_{base_lag}_{max(fc['rolling_windows'])}"]
    df = df.dropna(subset=warmup_cols).reset_index(drop=True)
    return df


def validate_contract(df: pd.DataFrame, contract: dict) -> list[str]:
    """Return a list of violations (empty = contract satisfied)."""
    errors = []
    spec = {**contract["keys"], **contract["target"], **contract["features"]}
    for col, meta in spec.items():
        if col not in df.columns:
            errors.append(f"missing column: {col}")
            continue
        kind = df[col].dtype.kind
        if kind != meta["dtype"]:
            errors.append(f"{col}: dtype kind {kind!r} != contract {meta['dtype']!r}")
    for col in contract["categoricals"]:
        if col not in df.columns:
            errors.append(f"missing categorical: {col}")
    if not errors and df[list(contract["keys"])].duplicated().any():
        errors.append("duplicate (item_id, store_id, date) keys")
    return errors


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    params = load_params()
    panel = pd.read_parquet(processed_dir(params) / PANEL_FILE)
    feats = build_features(panel, params)
    violations = validate_contract(feats, load_contract())
    if violations:
        raise SystemExit("feature contract violated:\n- " + "\n- ".join(violations))
    out = processed_dir(params) / FEATURES_FILE
    feats.to_parquet(out, index=False)
    print(f"features: {len(feats):,} rows, contract v{load_contract()['version']} OK")
