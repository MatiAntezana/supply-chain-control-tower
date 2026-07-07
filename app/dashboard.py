"""Streamlit dashboard: forecast, policy and simulated KPIs per SKU.

Consumes the FastAPI service (API_URL env var, default localhost:8000).
If the API is unreachable it falls back to reading the local artifacts, so
the same file works on Hugging Face Spaces and in local dev.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, load_params, processed_dir  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SC Control Tower", page_icon="📦", layout="wide")
st.title("📦 Supply Chain Control Tower")
st.caption("Forecast probabilístico → política óptima de reposición → validación por simulación (M5 Walmart)")


@st.cache_data(ttl=600)
def get_skus() -> pd.DataFrame:
    try:
        r = httpx.get(f"{API_URL}/skus", timeout=5)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception:
        params = load_params()
        recs = pd.read_parquet(processed_dir(params) / "recommendations.parquet")
        return recs[["item_id", "store_id", "recommended_policy"]]


@st.cache_data(ttl=600)
def get_recommendation(item_id: str, store_id: str) -> dict:
    try:
        r = httpx.get(f"{API_URL}/recommend", params={"item_id": item_id, "store_id": store_id}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        from src.serve.api import recommend  # same logic, no HTTP
        return recommend(item_id, store_id).model_dump()


skus = get_skus()
col1, col2 = st.sidebar.columns(2)
store = col1.selectbox("Tienda", sorted(skus["store_id"].unique()))
item = col2.selectbox("Producto", sorted(skus[skus["store_id"] == store]["item_id"].unique()))
rec = get_recommendation(item, store)

# --- policy summary -------------------------------------------------------
pol = rec["policy"]
st.subheader(f"Política recomendada: `{pol['recommended_policy']}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Punto de pedido (s)", f"{pol['reorder_point_s']:.0f} u")
c2.metric("Order-up-to (S)", f"{pol['order_up_to_S']:.0f} u")
c3.metric("Safety stock", f"{pol['safety_stock']:.0f} u")
c4.metric("Nivel de servicio objetivo", f"{pol['service_level']:.0%}")

# --- forecast fan chart ---------------------------------------------------
f = pd.DataFrame(rec["forecast"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=f["date"], y=f["p95"], name="p95", line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=f["date"], y=f["p50"], name="forecast p50", fill="tonexty",
                         fillcolor="rgba(99,110,250,0.2)", line=dict(color="#636efa")))
fig.add_trace(go.Scatter(x=f["date"], y=f["actual"], name="venta real",
                         line=dict(color="#00cc96", dash="dot")))
fig.update_layout(title="Demanda: forecast probabilístico (banda p50–p95) vs real",
                  height=380, margin=dict(t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# --- simulated KPI comparison --------------------------------------------
st.subheader("Validación por simulación (Monte-Carlo, 200 réplicas)")
k = pd.DataFrame(rec["simulated_kpis"]).set_index("policy")
c1, c2 = st.columns(2)
with c1:
    figc = go.Figure(go.Bar(x=k.index, y=k["total_cost_mean"],
                            error_y=dict(array=k["total_cost_p90"] - k["total_cost_mean"]),
                            marker_color=["#ef553b", "#636efa", "#00cc96"][: len(k)]))
    figc.update_layout(title="Costo total esperado (28 días) — barra de error: p90", height=330)
    st.plotly_chart(figc, use_container_width=True)
with c2:
    figf = go.Figure(go.Bar(x=k.index, y=k["fill_rate_mean"],
                            error_y=dict(array=k["fill_rate_mean"] - k["fill_rate_p10"]),
                            marker_color=["#ef553b", "#636efa", "#00cc96"][: len(k)]))
    figf.update_layout(title="Fill rate esperado — barra de error: p10", height=330,
                       yaxis=dict(range=[0.5, 1.02]))
    st.plotly_chart(figf, use_container_width=True)
st.dataframe(k.round(3), use_container_width=True)

# --- cost <-> service frontier -------------------------------------------
curve_path = REPO_ROOT / "reports" / "cost_service_curve.csv"
if curve_path.exists():
    st.subheader("Frontera costo ↔ nivel de servicio (promedio sobre 150 SKUs)")
    curve = pd.read_csv(curve_path)
    figs = go.Figure(go.Scatter(x=curve["fill_rate_mean"], y=curve["total_cost_mean"],
                                mode="lines+markers+text",
                                text=[f"{s:.0%}" for s in curve["service_level"]],
                                textposition="top center"))
    figs.update_layout(title="Cada punto = política (s,S) a un nivel de servicio objetivo",
                       xaxis_title="fill rate simulado", yaxis_title="costo total esperado ($ / 28d)",
                       height=380)
    st.plotly_chart(figs, use_container_width=True)
