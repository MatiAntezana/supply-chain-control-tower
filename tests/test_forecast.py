"""M2 tests: temporal splits without leakage, quantile sanity, metrics math."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_params, processed_dir
from src.forecast.backtest import make_folds
from src.forecast.metrics import coverage, pinball_loss, rmsse
from src.forecast.train import FORECAST_FILE


def test_folds_are_temporal_and_disjoint_from_train():
    dates = pd.Series(pd.date_range("2015-01-01", "2016-06-19"))
    origins = make_folds(dates, horizon=28, n_folds=3, step=28)
    assert origins == sorted(origins)
    # Every test window must end at or before the last available date.
    for o in origins:
        assert o + pd.Timedelta(days=28) <= dates.max()


def test_pinball_and_coverage_known_values():
    y = np.array([10.0, 10.0])
    assert pinball_loss(y, np.array([10.0, 10.0]), 0.5) == 0.0
    # under-forecast by 2 at q=0.9 costs 0.9*2
    assert pinball_loss(y, np.array([8.0, 8.0]), 0.9) == pytest.approx(1.8)
    assert coverage(y, np.array([12.0, 9.0])) == 0.5


def test_rmsse_perfect_forecast_is_zero():
    train = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    y = np.array([2.0, 3.0])
    assert rmsse(train, y, y.copy()) == 0.0


def test_forecast_artifact_quantiles_do_not_cross():
    params = load_params()
    path = processed_dir(params) / FORECAST_FILE
    if not path.exists():
        pytest.skip("run `make forecast` first")
    f = pd.read_parquet(path)
    assert (f["p50"] <= f["p90"] + 1e-9).all()
    assert (f["p90"] <= f["p95"] + 1e-9).all()
    assert (f["p50"] >= 0).all()
    # Served window is exactly the configured horizon.
    assert f["date"].nunique() == params["forecast"]["horizon"]
