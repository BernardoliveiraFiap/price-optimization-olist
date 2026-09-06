"""The pipeline is allowed to refuse, and these are the refusals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricing.elasticity.loglog_fe import estimate_iv, estimate_twoway_fe
from pricing.evaluate.impact import build_products
from pricing.optimize.milp import OptimisationConfig, optimise_prices
from pricing.pipeline import choose_causal_estimator, credible_betas
from pricing.synth import SynthParams, make_panel


def test_credible_betas_rejects_positive_and_inelastic():
    table = pd.DataFrame(
        {
            "category": ["elastic", "inelastic", "positive", "noisy"],
            "elasticity": [-1.80, -0.60, 0.40, -2.00],
            "std_error": [0.20, 0.10, 0.30, 3.00],  # last one has |t| < 2
        }
    )
    accepted, annotated = credible_betas(table)
    assert set(accepted) == {"elastic"}
    assert annotated["credible"].tolist() == [True, False, False, False]


def test_a_strong_instrument_is_used():
    panel, _ = make_panel(SynthParams(n_units=80, n_weeks=60))
    estimator, reason = choose_causal_estimator(panel)
    assert estimator is estimate_iv
    assert "accepted" in reason


def test_a_weak_instrument_is_refused():
    """A useless instrument must fall back to fixed effects, not be trusted.

    Weak-IV estimates are biased toward OLS with intervals that do not cover,
    so silently reporting one as "the causal estimate" is worse than declining.
    """
    panel, _ = make_panel(SynthParams(n_units=80, n_weeks=60))
    rng = np.random.default_rng(0)
    panel = panel.copy()
    panel["cost_index"] = rng.normal(size=len(panel))  # pure noise

    estimator, reason = choose_causal_estimator(panel)
    assert estimator is estimate_twoway_fe
    assert "rejected" in reason

    result = estimate_iv(panel)
    assert result.diagnostics["weak_instrument"] is True


def test_infeasible_guardrails_raise_instead_of_returning_nothing():
    """An aborted solve used to surface as an empty frame far downstream."""
    panel, _ = make_panel(SynthParams(n_units=30, n_weeks=30))
    products = build_products(panel, -1.8)
    impossible = OptimisationConfig(grid_steps=5, min_revenue_ratio=5.0)

    with pytest.raises(RuntimeError, match="solver returned status"):
        optimise_prices(products, impossible)
