"""The MILP has to respect the guardrails it was given -- not approximately."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing.evaluate.impact import build_products, evaluate_policy
from pricing.optimize.milp import (
    OptimisationConfig,
    build_price_grid,
    optimise_prices,
    optimise_unconstrained,
)
from pricing.synth import SynthParams, make_panel


@pytest.fixture(scope="module")
def products() -> pd.DataFrame:
    panel, _ = make_panel(SynthParams(n_units=40, n_weeks=30))
    return build_products(panel, -1.8)


def test_grid_prices_bracket_the_current_price(products):
    cfg = OptimisationConfig(price_band=0.2, grid_steps=5)
    grid = build_price_grid(products, cfg)
    assert len(grid) == len(products) * cfg.grid_steps
    per_unit = grid.groupby("unit_id")["delta"]
    assert per_unit.min().max() == pytest.approx(-cfg.price_band)
    assert per_unit.max().min() == pytest.approx(cfg.price_band)


def test_exactly_one_price_per_product(products):
    result = optimise_prices(products, OptimisationConfig(grid_steps=5))
    assert result.status == "Optimal"
    assert len(result.prices) == len(products)
    assert result.prices["unit_id"].is_unique


def test_guardrails_hold(products):
    cfg = OptimisationConfig(
        grid_steps=7, max_avg_price_increase=0.02, min_revenue_ratio=0.99
    )
    result = optimise_prices(products, cfg)
    s = result.summary
    # Small numerical slack: CBC returns values like 0.020000000001.
    assert s["avg_price_change_pct"] <= 100 * cfg.max_avg_price_increase + 1e-6
    assert (
        s["revenue_optimised"]
        >= cfg.min_revenue_ratio * s["revenue_baseline"] - 1e-6
    )


def test_constraints_can_only_cost_margin(products):
    cfg = OptimisationConfig(grid_steps=7, max_avg_price_increase=0.0)
    constrained = optimise_prices(products, cfg)
    free = optimise_unconstrained(products, cfg)
    assert (
        constrained.summary["margin_optimised"]
        <= free.summary["margin_optimised"] + 1e-6
    )


def test_a_volume_floor_is_honoured(products):
    cfg = OptimisationConfig(grid_steps=7, min_volume_ratio=1.0)
    result = optimise_prices(products, cfg)
    assert result.status == "Optimal"
    assert result.summary["units_optimised"] >= result.summary["units_baseline"] - 1e-6


def test_unchanged_prices_produce_no_uplift(products):
    """A price list that changes nothing must score exactly zero."""
    flat = products.rename(
        columns={"price": "price_before", "units": "units_before"}
    ).copy()
    flat["price_after"] = flat["price_before"]
    scored = evaluate_policy(flat, -1.8)
    assert scored["margin_uplift_pct"] == pytest.approx(0.0, abs=1e-9)
    assert scored["volume_change_pct"] == pytest.approx(0.0, abs=1e-9)


def test_demand_falls_when_price_rises(products):
    grid = build_price_grid(products, OptimisationConfig(grid_steps=5))
    for _, chunk in grid.groupby("unit_id"):
        ordered = chunk.sort_values("delta")
        assert np.all(np.diff(ordered["cand_units"].to_numpy()) < 0)
