"""The estimators are checked against a DGP whose answer is known."""

from __future__ import annotations

import pytest

from pricing.elasticity.loglog_fe import (
    estimate_iv,
    estimate_ols,
    estimate_twoway_fe,
)
from pricing.synth import SynthParams, make_panel

SMALL = dict(n_units=80, n_weeks=60)


def test_panel_has_the_expected_shape():
    panel, truth = make_panel(SynthParams(**SMALL))
    assert len(panel) == SMALL["n_units"] * SMALL["n_weeks"]
    assert (panel["price"] > 0).all()
    assert (panel["units"] > 0).all()
    assert truth["beta_mean"] < 0


def test_fixed_effects_recover_truth_when_pricing_is_exogenous():
    """With the endogeneity knob at zero, FE must be close to unbiased.

    This is the sanity check that guards the estimator itself: if it cannot
    recover the truth in the easy case, nothing it says about the hard case is
    worth reading.
    """
    panel, truth = make_panel(SynthParams(**SMALL, price_on_xi=0.0))
    res = estimate_twoway_fe(panel)
    assert res.elasticity == pytest.approx(truth["beta_mean"], abs=0.10)


def test_pooled_ols_is_biased_towards_zero():
    """Latent quality lifts price and demand together, so OLS understates."""
    panel, truth = make_panel(SynthParams(**SMALL))
    res = estimate_ols(panel)
    assert res.elasticity > truth["beta_mean"]  # closer to zero
    assert abs(res.elasticity - truth["beta_mean"]) > 0.15


def test_iv_beats_fixed_effects_under_endogenous_pricing():
    panel, truth = make_panel(SynthParams(**SMALL))
    fe = estimate_twoway_fe(panel)
    iv = estimate_iv(panel)
    true_beta = truth["beta_mean"]
    assert abs(iv.elasticity - true_beta) < abs(fe.elasticity - true_beta)
    assert iv.diagnostics["first_stage_F"] > 10  # not a weak instrument


def test_iv_requires_the_instrument_column():
    panel, _ = make_panel(SynthParams(**SMALL))
    with pytest.raises(KeyError):
        estimate_iv(panel.drop(columns=["cost_index"]))
