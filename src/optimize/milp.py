"""Multi-period lot-sizing MILP (PuLP + CBC): exact optimum with real constraints.

Couples all SKUs of a store through shared warehouse capacity — something the
closed-form (s,S) cannot see. Demand is the p50 forecast; uncertainty enters
via per-SKU safety-stock floors (from the quantile spread). Lead time is its
mean (stochastic lead time is the simulator's job, M4).

Trade-off vs (s,S): optimal but O(SKUs x periods) binaries -> scales poorly;
(s,S) is instant but ignores coupling. The project ships both on purpose.
"""

from __future__ import annotations

import numpy as np
import pulp


def solve_milp(
    demand: np.ndarray,          # (n_skus, T) p50 demand
    safety_stock: np.ndarray,    # (n_skus,) inventory floor
    initial_inventory: np.ndarray,  # (n_skus,)
    holding_cost_day: np.ndarray,   # (n_skus,)
    ordering_cost: np.ndarray,      # (n_skus,)
    lead_time: int,
    min_order: int,
    multiple: int,
    capacity: float | None = None,
    time_limit_s: int = 120,
) -> dict:
    n, t_len = demand.shape
    big_m = float(demand.sum(axis=1).max() * 2 + multiple)

    prob = pulp.LpProblem("inventory_lot_sizing", pulp.LpMinimize)
    # q = n_lots * multiple  (integer lots enforce the order-multiple constraint)
    lots = pulp.LpVariable.dicts("lots", ((k, t) for k in range(n) for t in range(t_len)),
                                 lowBound=0, cat="Integer")
    y = pulp.LpVariable.dicts("order", ((k, t) for k in range(n) for t in range(t_len)),
                              cat="Binary")
    inv = pulp.LpVariable.dicts("inv", ((k, t) for k in range(n) for t in range(t_len)),
                                lowBound=0)

    prob += pulp.lpSum(
        ordering_cost[k] * y[k, t] + holding_cost_day[k] * inv[k, t]
        for k in range(n) for t in range(t_len)
    )

    for k in range(n):
        for t in range(t_len):
            arrival = lots[k, t - lead_time] * multiple if t - lead_time >= 0 else 0
            prev = inv[k, t - 1] if t > 0 else initial_inventory[k]
            prob += inv[k, t] == prev + arrival - float(demand[k, t])
            # Inventory floor = safety stock (chance-constraint approximation).
            # Only enforced once orders *can* arrive (t >= lead_time): the first
            # L days are not controllable by any policy in a finite horizon.
            if t >= lead_time:
                prob += inv[k, t] >= float(safety_stock[k])
            # min-order / linking constraints
            prob += lots[k, t] * multiple <= big_m * y[k, t]
            prob += lots[k, t] * multiple >= min_order * y[k, t]

    if capacity is not None:
        for t in range(t_len):
            prob += pulp.lpSum(inv[k, t] for k in range(n)) <= capacity

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)

    orders = np.zeros((n, t_len))
    for k in range(n):
        for t in range(t_len):
            orders[k, t] = (lots[k, t].value() or 0) * multiple
    return {
        "status": pulp.LpStatus[prob.status],
        "objective": pulp.value(prob.objective),
        "orders": orders,
    }
