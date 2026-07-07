"""Per-SKU economics derived from M5 prices + synthetic cost parameters.

M5 has no supplier costs, so unit cost / holding / stockout penalties are
SYNTHETIC fractions of the observed sell price (see configs/params.yaml),
seeded and documented — never real company data.
"""

from __future__ import annotations

import pandas as pd


def sku_economics(forecast: pd.DataFrame, params: dict) -> pd.DataFrame:
    """One row per (item_id, store_id): price, unit_cost, holding/day, penalties."""
    inv = params["inventory"]
    econ = (
        forecast.groupby(["item_id", "store_id"], observed=True)["sell_price"]
        .mean()
        .rename("price")
        .reset_index()
    )
    econ["unit_cost"] = econ["price"] * inv["unit_cost_frac_of_price"]
    econ["holding_cost_day"] = econ["unit_cost"] * inv["holding_rate_annual"] / 365.0
    econ["stockout_penalty"] = econ["price"] * inv["stockout_penalty_frac_of_price"]
    econ["ordering_cost"] = inv["ordering_cost"]
    return econ
