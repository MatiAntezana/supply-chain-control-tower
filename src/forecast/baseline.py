"""Seasonal-naive probabilistic baseline.

For each (series, weekday), the forecast quantiles are the empirical quantiles
of that weekday's units over the last 8 weeks before the forecast origin.
Simple, leak-free, and surprisingly hard to beat on intermittent retail demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_WEEKS = 8


def baseline_forecast(
    train: pd.DataFrame, test_index: pd.DataFrame, quantiles: list[float]
) -> dict[float, np.ndarray]:
    """Predict quantiles for `test_index` rows (item_id, store_id, date).

    `train` must contain item_id, store_id, date, wday, units up to the origin.
    """
    recent = train.sort_values("date").groupby(
        ["item_id", "store_id", "wday"], observed=True
    )["units"].apply(lambda s: s.tail(N_WEEKS).to_numpy())

    preds: dict[float, list[float]] = {q: [] for q in quantiles}
    global_fallback = train["units"].to_numpy()
    for _, row in test_index.iterrows():
        key = (row["item_id"], row["store_id"], row["wday"])
        hist = recent.get(key, global_fallback)
        for q in quantiles:
            preds[q].append(float(np.quantile(hist, q)))
    return {q: np.asarray(v) for q, v in preds.items()}
