"""FastAPI service: the control tower's decision endpoint.

GET /recommend?item_id=&store_id= returns, for one SKU:
forecast quantiles + recommended replenishment policy + simulated KPIs.
Artifacts are precomputed by the pipeline (make pipeline) and loaded once.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import load_params, processed_dir
from src.forecast.train import FORECAST_FILE
from src.simulate.run import KPIS_FILE, RECS_FILE

app = FastAPI(title="Supply Chain Control Tower", version="1.0")


@lru_cache(maxsize=1)
def artifacts() -> dict[str, pd.DataFrame]:
    params = load_params()
    d = processed_dir(params)
    return {
        "forecast": pd.read_parquet(d / FORECAST_FILE),
        "kpis": pd.read_parquet(d / KPIS_FILE),
        "recs": pd.read_parquet(d / RECS_FILE),
    }


class ForecastPoint(BaseModel):
    date: str
    p50: float
    p90: float
    p95: float
    actual: float | None = None


class Policy(BaseModel):
    recommended_policy: str
    reorder_point_s: float
    order_up_to_S: float
    safety_stock: float
    service_level: float
    eoq: float


class PolicyKPIs(BaseModel):
    policy: str
    fill_rate_mean: float
    fill_rate_p10: float
    stockout_days_mean: float
    holding_cost_mean: float
    ordering_cost_mean: float
    stockout_cost_mean: float
    total_cost_mean: float
    total_cost_p90: float


class Recommendation(BaseModel):
    item_id: str
    store_id: str
    forecast: list[ForecastPoint]
    policy: Policy
    simulated_kpis: list[PolicyKPIs]


@app.get("/health")
def health() -> dict:
    a = artifacts()
    return {"status": "ok", "n_skus": len(a["recs"]), "forecast_rows": len(a["forecast"])}


@app.get("/skus")
def skus() -> list[dict]:
    return (
        artifacts()["recs"][["item_id", "store_id", "recommended_policy"]]
        .to_dict("records")
    )


@app.get("/recommend", response_model=Recommendation)
def recommend(item_id: str, store_id: str) -> Recommendation:
    a = artifacts()
    rec = a["recs"][(a["recs"]["item_id"] == item_id) & (a["recs"]["store_id"] == store_id)]
    if rec.empty:
        raise HTTPException(404, f"unknown SKU: {item_id} @ {store_id}")
    rec = rec.iloc[0]

    f = a["forecast"]
    fsku = f[(f["item_id"] == item_id) & (f["store_id"] == store_id)].sort_values("date")
    kpis = a["kpis"]
    ksku = kpis[(kpis["item_id"] == item_id) & (kpis["store_id"] == store_id)]

    return Recommendation(
        item_id=item_id,
        store_id=store_id,
        forecast=[
            ForecastPoint(date=str(r.date.date()), p50=r.p50, p90=r.p90, p95=r.p95,
                          actual=float(r.actual_units))
            for r in fsku.itertuples()
        ],
        policy=Policy(
            recommended_policy=rec["recommended_policy"],
            reorder_point_s=rec["reorder_point_s"],
            order_up_to_S=rec["order_up_to_S"],
            safety_stock=rec["safety_stock"],
            service_level=rec["service_level"],
            eoq=rec["eoq"],
        ),
        simulated_kpis=[
            PolicyKPIs(policy=r.policy, fill_rate_mean=r.fill_rate_mean,
                       fill_rate_p10=r.fill_rate_p10,
                       stockout_days_mean=r.stockout_days_mean,
                       holding_cost_mean=r.holding_cost_mean,
                       ordering_cost_mean=r.ordering_cost_mean,
                       stockout_cost_mean=r.stockout_cost_mean,
                       total_cost_mean=r.total_cost_mean,
                       total_cost_p90=r.total_cost_p90)
            for r in ksku.itertuples()
        ],
    )
