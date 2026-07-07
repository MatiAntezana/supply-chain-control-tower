"""Rolling-origin backtest: LightGBM quantile vs seasonal-naive baseline.

Folds are strictly temporal: train <= origin < test <= origin + horizon.
Writes reports/backtest_metrics.csv and logs each fold to MLflow.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd

from src.config import REPO_ROOT, load_params, mlflow_uri, processed_dir, set_seeds
from src.forecast.baseline import baseline_forecast
from src.forecast.metrics import evaluate_fold
from src.forecast.model import QuantileForecaster
from src.ingest.features import FEATURES_FILE

REPORTS_DIR = REPO_ROOT / "reports"


def make_folds(dates: pd.Series, horizon: int, n_folds: int, step: int) -> list[pd.Timestamp]:
    """Return fold origins, oldest first. Test window = (origin, origin+horizon]."""
    last = dates.max()
    return [last - pd.Timedelta(days=horizon + k * step) for k in range(n_folds - 1, -1, -1)]


def run_backtest(params: dict) -> pd.DataFrame:
    fc = params["forecast"]
    quantiles = fc["quantiles"]
    df = pd.read_parquet(processed_dir(params) / FEATURES_FILE)
    origins = make_folds(df["date"], fc["horizon"], fc["backtest"]["n_folds"], fc["backtest"]["step_days"])

    mlflow.set_tracking_uri(mlflow_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment"])

    rows = []
    for i, origin in enumerate(origins):
        train = df[df["date"] <= origin]
        test = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=fc["horizon"]))]
        assert train["date"].max() <= origin < test["date"].min(), "temporal leakage"

        model = QuantileForecaster(quantiles, fc["lgbm"], params["seed"]).fit(train)
        m_model = evaluate_fold(test, model.predict(test), train, quantiles)
        m_base = evaluate_fold(
            test, baseline_forecast(train, test[["item_id", "store_id", "wday"]], quantiles),
            train, quantiles,
        )
        rows.append({"fold": i, "origin": str(origin.date()), "model": "lgbm_quantile", **m_model})
        rows.append({"fold": i, "origin": str(origin.date()), "model": "seasonal_naive", **m_base})

        with mlflow.start_run(run_name=f"backtest_fold{i}"):
            mlflow.log_params({"origin": str(origin.date()), "horizon": fc["horizon"], **fc["lgbm"]})
            mlflow.log_metrics({f"lgbm_{k}": v for k, v in m_model.items()})
            mlflow.log_metrics({f"base_{k}": v for k, v in m_base.items()})

    result = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    result.to_csv(REPORTS_DIR / "backtest_metrics.csv", index=False)
    summary = result.groupby("model").mean(numeric_only=True).drop(columns="fold").round(4)
    (REPORTS_DIR / "backtest_summary.json").write_text(json.dumps(summary.to_dict("index"), indent=2))
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    params = load_params()
    set_seeds(params["seed"])
    result = run_backtest(params)
    print(result.groupby("model").mean(numeric_only=True).drop(columns="fold").round(3).to_string())
