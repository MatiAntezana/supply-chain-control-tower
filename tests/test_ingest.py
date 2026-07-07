"""M1 tests: panel integrity + feature contract + determinism of ingestion."""

import pandas as pd
import pytest

from src.config import load_contract, load_params, processed_dir
from src.ingest.features import FEATURES_FILE, validate_contract
from src.ingest.load_m5 import PANEL_FILE


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def panel(params):
    path = processed_dir(params) / PANEL_FILE
    if not path.exists():
        pytest.skip("run `make data` first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def features(params):
    path = processed_dir(params) / FEATURES_FILE
    if not path.exists():
        pytest.skip("run `make data` first")
    return pd.read_parquet(path)


def test_panel_shape_and_keys(panel, params):
    assert set(panel["store_id"].unique()) <= set(params["data"]["stores"])
    assert panel["item_id"].nunique() <= params["data"]["top_n_items"]
    assert not panel[["item_id", "store_id", "date"]].duplicated().any()
    assert (panel["units"] >= 0).all()


def test_panel_no_gaps(panel):
    """Each series must be a contiguous daily range (no missing days)."""
    span = panel.groupby(["item_id", "store_id"])["date"].agg(["min", "max", "size"])
    expected = (span["max"] - span["min"]).dt.days + 1
    assert (span["size"] == expected).all()


def test_feature_contract(features):
    assert validate_contract(features, load_contract()) == []


def test_lags_are_leak_free(features):
    """lag_28 today must equal units 28 days ago — never anything more recent."""
    s = features[
        (features["item_id"] == features["item_id"].iloc[0])
        & (features["store_id"] == features["store_id"].iloc[0])
    ].set_index("date").sort_index()
    joined = s[["lag_28"]].join(s["units"].shift(28).rename("units_28"))
    joined = joined.dropna()
    assert (joined["lag_28"] == joined["units_28"]).all()
