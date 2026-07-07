"""Stochastic demand generator: samples integer daily demand paths whose
per-day distribution is moment-matched to the forecast quantiles (p50, p95).

Overdispersed days (variance > mean, the retail norm) use a negative binomial;
otherwise Poisson. This is how forecast uncertainty enters the digital twin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

Z95 = norm.ppf(0.95)


def sample_demand_paths(fsku: pd.DataFrame, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (n_reps, T) integer demand matrix for one SKU.

    `fsku` must be date-sorted with columns p50, p95.
    """
    mean = fsku["p50"].to_numpy(dtype=float).clip(min=0.01)
    sigma = ((fsku["p95"] - fsku["p50"]).to_numpy(dtype=float) / Z95).clip(min=0.0)
    var = np.maximum(sigma**2, mean * 1.0001)  # at least Poisson dispersion

    t_len = len(mean)
    out = np.empty((n_reps, t_len), dtype=np.int64)
    for t in range(t_len):
        if var[t] > mean[t] * 1.001:  # negative binomial (overdispersed)
            r = mean[t] ** 2 / (var[t] - mean[t])
            p = r / (r + mean[t])
            out[:, t] = rng.negative_binomial(r, p, size=n_reps)
        else:
            out[:, t] = rng.poisson(mean[t], size=n_reps)
    return out
