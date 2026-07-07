"""Train final quantile models and produce the served demand forecast.

The forecast origin is (last_date - horizon): the model trains on everything
before it and predicts the final 28 observed days. Serving a window that has
actuals lets the monitoring layer compare forecast vs reality honestly.
Outputs: models/lgbm_final_p*.txt, data/processed/forecast.parquet.
"""

from __future__ import annotations

import argparse

import mlflow
import pandas as pd

from src.config import REPO_ROOT, load_params, mlflow_uri, processed_dir, set_seeds
from src.forecast.model import QuantileForecaster
from src.ingest.features import FEATURES_FILE

FORECAST_FILE = "forecast.parquet"
MODEL_TAG = "final"


def train_and_forecast(params: dict) -> pd.DataFrame:
    fc = params["forecast"]
    quantiles = fc["quantiles"]
    df = pd.read_parquet(processed_dir(params) / FEATURES_FILE)

    origin = df["date"].max() - pd.Timedelta(days=fc["horizon"])
    train = df[df["date"] <= origin]
    future = df[df["date"] > origin]

    model = QuantileForecaster(quantiles, fc["lgbm"], params["seed"]).fit(train)
    model.save(MODEL_TAG)
    preds = model.predict(future)

    out = future[["item_id", "store_id", "date", "units", "sell_price"]].copy()
    out = out.rename(columns={"units": "actual_units"})
    for q in quantiles:
        out[f"p{int(q * 100)}"] = preds[q]
    out["origin"] = origin
    out.to_parquet(processed_dir(params) / FORECAST_FILE, index=False)

    mlflow.set_tracking_uri(mlflow_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="train_final"):
        mlflow.log_params({"origin": str(origin.date()), "n_train_rows": len(train), **fc["lgbm"]})
        for path in (REPO_ROOT / "models").glob(f"lgbm_{MODEL_TAG}_*.txt"):
            mlflow.log_artifact(str(path))
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    params = load_params()
    set_seeds(params["seed"])
    out = train_and_forecast(params)
    print(f"forecast: {len(out):,} rows, origin={out['origin'].iloc[0].date()}")
