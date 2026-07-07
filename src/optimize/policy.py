"""Closed-form (s,S) replenishment policy from probabilistic forecasts.

The reorder point s is the service-level quantile of lead-time demand:
    s = mu_L + z * sigma_L,   sigma_L^2 = L*sigma_d^2 + mu_d^2*sigma_LT^2
where sigma_d is implied by the forecast quantile spread (p95 vs p50) — this
is where probabilistic forecasting feeds inventory: the tail, not the mean.
S = s + EOQ. Near-instant per SKU; the price is normality + no coupling.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

Z95 = norm.ppf(0.95)


def implied_daily_stats(fsku: pd.DataFrame) -> tuple[float, float]:
    """Daily demand mean/std implied by the forecast quantiles of one SKU."""
    mu = float(fsku["p50"].mean())
    sigma = float(((fsku["p95"] - fsku["p50"]) / Z95).clip(lower=0).mean())
    return mu, max(sigma, 1e-6)


def round_order(qty: float, min_order: int, multiple: int) -> int:
    """Round up to the order multiple, respecting the minimum order size."""
    if qty <= 0:
        return 0
    qty = max(qty, min_order)
    return int(math.ceil(qty / multiple) * multiple)


def newsvendor_policy(fsku: pd.DataFrame, econ: pd.Series, params: dict,
                      service_level: float | None = None) -> dict:
    inv = params["inventory"]
    service = service_level if service_level is not None else inv["service_level_target"]
    mu_d, sigma_d = implied_daily_stats(fsku)
    lt_mean = inv["lead_time"]["mean_days"]
    lt_sigma = lt_mean * inv["lead_time"]["cv"]

    mu_l = mu_d * lt_mean
    sigma_l = math.sqrt(lt_mean * sigma_d**2 + (mu_d * lt_sigma) ** 2)
    safety_stock = norm.ppf(service) * sigma_l
    s = mu_l + safety_stock

    # EOQ on daily rates: sqrt(2*K*mu / h_day)
    eoq = math.sqrt(2 * econ["ordering_cost"] * mu_d / econ["holding_cost_day"])
    order_up_to = s + round_order(eoq, inv["min_order_qty"], inv["order_multiple"])
    return {
        "mu_day": mu_d, "sigma_day": sigma_d, "safety_stock": safety_stock,
        "reorder_point_s": s, "order_up_to_S": order_up_to, "eoq": eoq,
        "service_level": service,
    }


def initial_inventory(policy: dict) -> float:
    """Common starting stock for every approach: s plus one day of demand."""
    return policy["reorder_point_s"] + policy["mu_day"]


def evaluate_deterministic(
    demand: np.ndarray, policy: dict, econ: pd.Series, params: dict,
    orders_schedule: np.ndarray | None = None,
) -> dict:
    """Deterministic day-by-day cost of a policy against a demand path.

    If `orders_schedule` is given (MILP output), it is executed as-is;
    otherwise the (s,S) rule decides orders. Used for the M3 comparison —
    the *stochastic* verdict belongs to the M4 simulator.
    """
    inv = params["inventory"]
    lt = int(round(inv["lead_time"]["mean_days"]))
    t_len = len(demand)
    on_hand = initial_inventory(policy)
    pipeline = np.zeros(t_len + lt + 1)
    holding = ordering = stockout_units = filled = 0.0

    for t in range(t_len):
        on_hand += pipeline[t]
        if orders_schedule is not None:
            q = float(orders_schedule[t])
            if q > 0:
                pipeline[t + lt] += q
                ordering += econ["ordering_cost"]
        else:
            inv_position = on_hand + pipeline[t + 1 : t + lt + 1].sum()
            if inv_position <= policy["reorder_point_s"]:
                q = round_order(
                    policy["order_up_to_S"] - inv_position,
                    inv["min_order_qty"], inv["order_multiple"],
                )
                if q > 0:
                    pipeline[t + lt] += q
                    ordering += econ["ordering_cost"]
        sold = min(on_hand, demand[t])
        stockout_units += demand[t] - sold
        filled += sold
        on_hand -= sold
        holding += on_hand * econ["holding_cost_day"]

    total_demand = float(demand.sum())
    return {
        "holding_cost": holding,
        "ordering_cost": ordering,
        "stockout_cost": stockout_units * econ["stockout_penalty"],
        "total_cost": holding + ordering + stockout_units * econ["stockout_penalty"],
        "fill_rate": filled / total_demand if total_demand > 0 else 1.0,
    }
