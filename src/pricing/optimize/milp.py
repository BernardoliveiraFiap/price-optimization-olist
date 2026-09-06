"""Portfolio price optimisation as a mixed-integer linear program.

A per-product optimum is a one-liner: with constant-elasticity demand and
|beta| > 1 the margin-maximising price has a closed form. That is not the
problem a pricing team actually has. Their problem is that prices are set
*jointly* under guardrails -- do not raise the average basket by more than a
few percent, do not lose revenue while chasing margin, do not move a price
more than customers will tolerate.

Those guardrails couple the products, and coupling is what turns this into an
optimisation problem rather than arithmetic.

Formulation
-----------
Each product ``i`` gets a grid of candidate prices ``k`` (a multiplier around
today's price). Binary ``x[i, k] = 1`` selects one. Because demand, revenue
and margin at each grid point are *precomputed constants*, the objective and
all constraints are linear in ``x`` -- the non-linearity of the demand curve
is absorbed into the coefficients. That is what makes a solver like CBC able
to handle thousands of products.

    maximise    sum_ik  margin[i,k] * x[i,k]
    subject to  sum_k   x[i,k] = 1                          for every i
                sum_ik  revenue[i,k] * x[i,k] >= floor      (revenue guardrail)
                sum_ik  w[i] * delta[k] * x[i,k] <= cap     (avg price guardrail)
                sum_ik  units[i,k] * x[i,k] >= volume_floor (optional)

Demand at a candidate price uses constant elasticity::

    q(p) = q0 * (p / p0) ** beta

which is exactly the model the elasticity estimators fit, so the optimiser and
the econometrics agree on what a price change does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from ..config import (
    DEFAULT_MARGIN_RATE,
    MAX_AVG_PRICE_INCREASE,
    MIN_REVENUE_RATIO,
    PRICE_BAND,
    PRICE_GRID_STEPS,
)


@dataclass
class OptimisationConfig:
    price_band: float = PRICE_BAND
    grid_steps: int = PRICE_GRID_STEPS
    max_avg_price_increase: float = MAX_AVG_PRICE_INCREASE
    min_revenue_ratio: float = MIN_REVENUE_RATIO
    min_volume_ratio: float | None = None
    margin_rate: float = DEFAULT_MARGIN_RATE
    solver_msg: bool = False
    time_limit_s: int = 120


@dataclass
class OptimisationResult:
    prices: pd.DataFrame
    summary: dict
    status: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        s = self.summary
        return (
            f"[{self.status}] margin {s['margin_baseline']:,.0f} -> "
            f"{s['margin_optimised']:,.0f} "
            f"({s['margin_uplift_pct']:+.2f}%)  |  revenue "
            f"{s['revenue_uplift_pct']:+.2f}%  |  avg price "
            f"{s['avg_price_change_pct']:+.2f}%"
        )


def build_price_grid(
    products: pd.DataFrame, config: OptimisationConfig
) -> pd.DataFrame:
    """Expand each product into its candidate price points.

    ``products`` needs: ``unit_id``, ``price`` (current), ``units`` (baseline
    weekly demand at that price) and ``elasticity``. An optional ``unit_cost``
    column overrides the flat margin-rate assumption.
    """
    required = {"unit_id", "price", "units", "elasticity"}
    missing = required - set(products.columns)
    if missing:
        raise KeyError(f"products is missing columns: {sorted(missing)}")

    deltas = np.linspace(-config.price_band, config.price_band, config.grid_steps)
    grid = products.loc[products.index.repeat(len(deltas))].copy()
    grid["delta"] = np.tile(deltas, len(products))
    grid["grid_k"] = np.tile(np.arange(len(deltas)), len(products))

    grid["cand_price"] = grid["price"] * (1.0 + grid["delta"])
    # Constant-elasticity demand response.
    grid["cand_units"] = grid["units"] * (1.0 + grid["delta"]) ** grid["elasticity"]

    if "unit_cost" in grid.columns:
        cost = grid["unit_cost"]
    else:
        # Olist ships no cost data, so a flat gross-margin assumption stands in.
        # It is a headline assumption, not a detail: see sensitivity analysis.
        cost = grid["price"] * (1.0 - config.margin_rate)
    grid["unit_cost"] = cost

    grid["cand_revenue"] = grid["cand_price"] * grid["cand_units"]
    grid["cand_margin"] = (grid["cand_price"] - cost) * grid["cand_units"]
    return grid.reset_index(drop=True)


def _baseline(products: pd.DataFrame, config: OptimisationConfig) -> dict:
    cost = (
        products["unit_cost"]
        if "unit_cost" in products.columns
        else products["price"] * (1.0 - config.margin_rate)
    )
    revenue = products["price"] * products["units"]
    margin = (products["price"] - cost) * products["units"]
    return {
        "revenue": float(revenue.sum()),
        "margin": float(margin.sum()),
        "units": float(products["units"].sum()),
        "revenue_weights": (revenue / revenue.sum()).to_numpy(),
    }


def optimise_prices(
    products: pd.DataFrame, config: OptimisationConfig | None = None
) -> OptimisationResult:
    """Solve the constrained portfolio pricing MILP."""
    cfg = config or OptimisationConfig()
    grid = build_price_grid(products, cfg)
    base = _baseline(products, cfg)

    units_index = {u: i for i, u in enumerate(products["unit_id"].to_numpy())}
    weight_by_unit = dict(
        zip(products["unit_id"], base["revenue_weights"], strict=True)
    )

    problem = pulp.LpProblem("portfolio_pricing", pulp.LpMaximize)
    x = {
        (row.unit_id, row.grid_k): pulp.LpVariable(
            f"x_{units_index[row.unit_id]}_{row.grid_k}", cat="Binary"
        )
        for row in grid.itertuples()
    }

    # Objective: total gross margin.
    problem += pulp.lpSum(
        row.cand_margin * x[(row.unit_id, row.grid_k)] for row in grid.itertuples()
    )

    # One price per product.
    for unit_id, chunk in grid.groupby("unit_id", sort=False):
        problem += (
            pulp.lpSum(x[(unit_id, k)] for k in chunk["grid_k"]) == 1,
            f"one_price_{units_index[unit_id]}",
        )

    # Revenue guardrail: chasing margin must not quietly shrink the top line.
    problem += (
        pulp.lpSum(
            row.cand_revenue * x[(row.unit_id, row.grid_k)]
            for row in grid.itertuples()
        )
        >= cfg.min_revenue_ratio * base["revenue"],
        "revenue_floor",
    )

    # Average price guardrail, weighted by each product's share of revenue, so
    # a tiny product cannot buy headroom for a big one.
    problem += (
        pulp.lpSum(
            weight_by_unit[row.unit_id] * row.delta * x[(row.unit_id, row.grid_k)]
            for row in grid.itertuples()
        )
        <= cfg.max_avg_price_increase,
        "avg_price_cap",
    )

    if cfg.min_volume_ratio is not None:
        problem += (
            pulp.lpSum(
                row.cand_units * x[(row.unit_id, row.grid_k)]
                for row in grid.itertuples()
            )
            >= cfg.min_volume_ratio * base["units"],
            "volume_floor",
        )

    solver = pulp.PULP_CBC_CMD(msg=cfg.solver_msg, timeLimit=cfg.time_limit_s)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]

    # np.asarray, not a bare list: a Python list of bools is ambiguous to
    # pandas and can be read as a list of column labels.
    selected = np.asarray(
        [bool(x[(r.unit_id, r.grid_k)].value()) for r in grid.itertuples()],
        dtype=bool,
    )
    chosen = grid.loc[selected].copy()
    chosen = chosen.rename(
        columns={
            "price": "price_before",
            "units": "units_before",
            "cand_price": "price_after",
            "cand_units": "units_after",
        }
    )
    chosen["price_change_pct"] = 100.0 * chosen["delta"]

    summary = {
        "status": status,
        "n_products": int(len(products)),
        "revenue_baseline": base["revenue"],
        "revenue_optimised": float(chosen["cand_revenue"].sum()),
        "margin_baseline": base["margin"],
        "margin_optimised": float(chosen["cand_margin"].sum()),
        "units_baseline": base["units"],
        "units_optimised": float(chosen["units_after"].sum()),
    }
    summary["margin_uplift_pct"] = 100.0 * (
        summary["margin_optimised"] / summary["margin_baseline"] - 1.0
    )
    summary["revenue_uplift_pct"] = 100.0 * (
        summary["revenue_optimised"] / summary["revenue_baseline"] - 1.0
    )
    summary["volume_change_pct"] = 100.0 * (
        summary["units_optimised"] / summary["units_baseline"] - 1.0
    )
    summary["avg_price_change_pct"] = 100.0 * float(
        (chosen["delta"].to_numpy() * base["revenue_weights"]).sum()
    )
    summary["share_price_up"] = float((chosen["delta"] > 1e-9).mean())
    summary["share_price_down"] = float((chosen["delta"] < -1e-9).mean())

    keep = [
        "unit_id",
        "price_before",
        "price_after",
        "price_change_pct",
        "units_before",
        "units_after",
        "elasticity",
        "cand_revenue",
        "cand_margin",
    ]
    if "category" in chosen.columns:
        keep.insert(1, "category")
    return OptimisationResult(
        prices=chosen[keep].sort_values("unit_id").reset_index(drop=True),
        summary=summary,
        status=status,
    )


def optimise_unconstrained(
    products: pd.DataFrame, config: OptimisationConfig | None = None
) -> OptimisationResult:
    """Greedy per-product optimum, ignoring the portfolio guardrails.

    Useful as an upper bound: the gap against :func:`optimise_prices` is the
    price the business pays for its own guardrails, which is a number a
    pricing owner can actually argue about.
    """
    cfg = config or OptimisationConfig()
    grid = build_price_grid(products, cfg)
    base = _baseline(products, cfg)

    best = grid.loc[grid.groupby("unit_id")["cand_margin"].idxmax()].copy()
    best = best.rename(
        columns={
            "price": "price_before",
            "units": "units_before",
            "cand_price": "price_after",
            "cand_units": "units_after",
        }
    )
    best["price_change_pct"] = 100.0 * best["delta"]

    summary = {
        "status": "Unconstrained",
        "n_products": int(len(products)),
        "revenue_baseline": base["revenue"],
        "revenue_optimised": float(best["cand_revenue"].sum()),
        "margin_baseline": base["margin"],
        "margin_optimised": float(best["cand_margin"].sum()),
        "units_baseline": base["units"],
        "units_optimised": float(best["units_after"].sum()),
    }
    summary["margin_uplift_pct"] = 100.0 * (
        summary["margin_optimised"] / summary["margin_baseline"] - 1.0
    )
    summary["revenue_uplift_pct"] = 100.0 * (
        summary["revenue_optimised"] / summary["revenue_baseline"] - 1.0
    )
    summary["volume_change_pct"] = 100.0 * (
        summary["units_optimised"] / summary["units_baseline"] - 1.0
    )
    summary["avg_price_change_pct"] = 100.0 * float(
        (best["delta"].to_numpy() * base["revenue_weights"]).sum()
    )
    return OptimisationResult(
        prices=best.sort_values("unit_id").reset_index(drop=True),
        summary=summary,
        status="Unconstrained",
    )
