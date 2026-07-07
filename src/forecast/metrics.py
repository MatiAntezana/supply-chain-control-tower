"""Forecast evaluation metrics: RMSSE (M5-style), pinball loss, quantile coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmsse(train_actuals: pd.Series, test_actuals: np.ndarray, test_preds: np.ndarray) -> float:
    """Root Mean Squared Scaled Error, scaled by the naive lag-1 error on train."""
    scale = np.mean(np.diff(train_actuals.to_numpy()) ** 2)
    if scale == 0:
        return np.nan
    return float(np.sqrt(np.mean((test_actuals - test_preds) ** 2) / scale))


def pinball_loss(actuals: np.ndarray, preds: np.ndarray, q: float) -> float:
    diff = actuals - preds
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(actuals: np.ndarray, preds: np.ndarray) -> float:
    """Empirical coverage: fraction of actuals at or below the predicted quantile."""
    return float(np.mean(actuals <= preds))


def evaluate_fold(
    test: pd.DataFrame, preds: dict[float, np.ndarray], train: pd.DataFrame, quantiles: list[float]
) -> dict:
    """Metrics for one backtest fold. `test`/`train` need item_id/store_id/units."""
    out: dict[str, float] = {}
    y = test["units"].to_numpy(dtype=float)
    for q in quantiles:
        out[f"pinball_p{int(q * 100)}"] = pinball_loss(y, preds[q], q)
        out[f"coverage_p{int(q * 100)}"] = coverage(y, preds[q])

    # RMSSE on the median, weighted by each series' total units (WRMSSE-like).
    p50 = pd.Series(preds[0.5], index=test.index)
    scores, weights = [], []
    for (item, store), te in test.groupby(["item_id", "store_id"], observed=True):
        tr = train[(train["item_id"] == item) & (train["store_id"] == store)]
        s = rmsse(tr["units"], te["units"].to_numpy(dtype=float), p50.loc[te.index].to_numpy())
        if not np.isnan(s):
            scores.append(s)
            weights.append(tr["units"].sum())
    weights_arr = np.asarray(weights, dtype=float)
    out["rmsse_mean"] = float(np.mean(scores))
    out["wrmsse"] = float(np.average(scores, weights=weights_arr / weights_arr.sum()))
    return out
