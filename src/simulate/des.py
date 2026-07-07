"""Single-echelon discrete-event inventory simulator (SimPy).

One replication = one demand path + stochastic supplier lead times (gamma).
Orders are placed at daily reviews; deliveries arrive as *events* at
continuous times (t + lead_time), which is exactly what a closed-form policy
cannot see: crossing orders, late arrivals, demand spikes during exposure.
"""

from __future__ import annotations

import numpy as np
import simpy

from src.optimize.policy import round_order


def lead_time_sampler(inv_params: dict, rng: np.random.Generator):
    """Gamma-distributed lead time with configured mean and CV, >= 0.5 days."""
    mean = inv_params["lead_time"]["mean_days"]
    cv = inv_params["lead_time"]["cv"]
    shape = 1.0 / cv**2
    scale = mean * cv**2

    def sample() -> float:
        return max(0.5, float(rng.gamma(shape, scale)))

    return sample


class _State:
    __slots__ = ("on_hand", "on_order", "filled", "demanded", "stockout_days",
                 "holding", "n_orders")

    def __init__(self, initial: float):
        self.on_hand = initial
        self.on_order = 0.0
        self.filled = 0.0
        self.demanded = 0.0
        self.stockout_days = 0
        self.holding = 0.0
        self.n_orders = 0


def run_replication(
    demand: np.ndarray,            # (T,) integer demand path
    policy: dict,                  # needs reorder_point_s / order_up_to_S
    econ,                          # price/cost Series
    inv_params: dict,
    initial_on_hand: float,
    rng: np.random.Generator,
    orders_schedule: np.ndarray | None = None,  # (T,) MILP mode if given
) -> dict:
    env = simpy.Environment()
    state = _State(initial_on_hand)
    sample_lt = lead_time_sampler(inv_params, rng)

    def deliver(qty: float):
        yield env.timeout(sample_lt())
        state.on_hand += qty
        state.on_order -= qty

    def place(qty: float):
        if qty > 0:
            state.on_order += qty
            state.n_orders += 1
            env.process(deliver(qty))

    def daily_cycle():
        for t in range(len(demand)):
            # morning review: order per rule or per fixed MILP schedule
            if orders_schedule is not None:
                place(float(orders_schedule[t]))
            else:
                position = state.on_hand + state.on_order
                if position <= policy["reorder_point_s"]:
                    place(round_order(
                        policy["order_up_to_S"] - position,
                        inv_params["min_order_qty"], inv_params["order_multiple"],
                    ))
            yield env.timeout(0.5)  # demand hits mid-day
            d = float(demand[t])
            sold = min(state.on_hand, d)
            state.on_hand -= sold
            state.filled += sold
            state.demanded += d
            if sold < d:
                state.stockout_days += 1
            yield env.timeout(0.5)  # end of day: holding cost on what remains
            state.holding += state.on_hand

    env.process(daily_cycle())
    env.run()

    holding_cost = state.holding * float(econ["holding_cost_day"])
    ordering_cost = state.n_orders * float(econ["ordering_cost"])
    stockout_cost = (state.demanded - state.filled) * float(econ["stockout_penalty"])
    return {
        "fill_rate": state.filled / state.demanded if state.demanded > 0 else 1.0,
        "stockout_days": state.stockout_days,
        "holding_cost": holding_cost,
        "ordering_cost": ordering_cost,
        "stockout_cost": stockout_cost,
        "total_cost": holding_cost + ordering_cost + stockout_cost,
    }
