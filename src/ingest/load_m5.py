"""M5 ingestion: raw Walmart CSVs -> tidy daily panel (item x store x day).

Reads the locally-provided `m5-forecasting-accuracy/` folder (no downloads,
no credentials). Output: data/processed/panel.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_params, processed_dir, raw_dir

PANEL_FILE = "panel.parquet"


def _read_raw(raw: Path, stores: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    sales = sales[sales["store_id"].isin(stores)].reset_index(drop=True)
    calendar = pd.read_csv(raw / "calendar.csv")
    prices = pd.read_csv(raw / "sell_prices.csv")
    prices = prices[prices["store_id"].isin(stores)].reset_index(drop=True)
    return sales, calendar, prices


def _select_top_items(sales: pd.DataFrame, n: int) -> list[str]:
    """Top-N items by total units sold across the selected stores."""
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    totals = (
        sales.assign(total=sales[day_cols].sum(axis=1))
        .groupby("item_id")["total"]
        .sum()
        .sort_values(ascending=False)
    )
    return totals.head(n).index.tolist()


def build_panel(params: dict) -> pd.DataFrame:
    """Build the tidy panel and write it to parquet. Fully deterministic."""
    cfg = params["data"]
    raw = raw_dir(params)
    sales, calendar, prices = _read_raw(raw, cfg["stores"])

    items = _select_top_items(sales, cfg["top_n_items"])
    sales = sales[sales["item_id"].isin(items)]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(
        id_vars=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="units",
    )

    cal = calendar[
        ["d", "date", "wm_yr_wk", "wday", "month", "year",
         "event_name_1", "snap_CA", "snap_TX", "snap_WI"]
    ].copy()
    long = long.merge(cal, on="d", how="left")
    long["date"] = pd.to_datetime(long["date"])

    # SNAP flag for the store's own state.
    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    long["snap"] = 0
    for state, col in snap_map.items():
        mask = long["state_id"] == state
        long.loc[mask, "snap"] = long.loc[mask, col].astype(int)
    long["is_event"] = long["event_name_1"].notna().astype(int)
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI", "event_name_1"])

    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # Price is missing before an item is first listed; back/forward-fill per series.
    long = long.sort_values(["item_id", "store_id", "date"])
    long["sell_price"] = long.groupby(["item_id", "store_id"])["sell_price"].transform(
        lambda s: s.ffill().bfill()
    )

    if cfg["drop_leading_zeros"]:
        # Drop the pre-launch stretch: days before the first recorded sale.
        first_sale = (
            long[long["units"] > 0].groupby(["item_id", "store_id"])["date"].min().rename("first_sale")
        )
        long = long.merge(first_sale, on=["item_id", "store_id"], how="left")
        long = long[long["date"] >= long["first_sale"]].drop(columns=["first_sale"])

    long["units"] = long["units"].astype("int32")
    long = long.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    out = processed_dir(params) / PANEL_FILE
    long.to_parquet(out, index=False)
    return long


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    params = load_params()
    panel = build_panel(params)
    print(f"panel: {len(panel):,} rows, {panel[['item_id', 'store_id']].drop_duplicates().shape[0]} series")
