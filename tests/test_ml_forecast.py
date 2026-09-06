"""The forecaster is held to the same standard as the estimators.

Two kinds of property are locked here. The first is mechanical: features must
not see the future and folds must not train on it -- a leak would make every
number in this module a lie in the flattering direction.

The second is the finding itself. The claim "a good forecaster implies a bad
elasticity" is only worth printing if it survives the synthetic panel, where
the true elasticity is written before anything is fitted and the endogeneity
can be switched off. Turning it off has to make the bias go away; leaving it
on has to bring the bias back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing.demand.ml_forecast import (
    ELASTICITY_BUMP,
    _feature_columns,
    _gbm,
    add_features,
    evaluate,
    expanding_folds,
    implied_elasticity,
)
from pricing.synth import SynthParams, make_panel

SMALL = dict(n_units=60, n_weeks=60)


def _toy_panel() -> pd.DataFrame:
    """Two units, ascending log_units, so lags are checkable by eye."""
    weeks = pd.date_range("2026-01-05", periods=6, freq="W-MON")
    rows = []
    for unit in (0, 1):
        for i, week in enumerate(weeks):
            rows.append(
                {
                    "unit_id": unit,
                    "week": week,
                    "category": "c",
                    "log_units": float(unit * 100 + i),
                    "log_price": 1.0 + 0.01 * i,
                    "freight": 1.0,
                    "review_score": 4.0,
                    "competition": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_lags_never_see_the_future():
    df = add_features(_toy_panel())

    for unit in (0, 1):
        block = df[df["unit_id"] == unit].sort_values("week")
        target = block["log_units"].to_numpy()

        # lag_1 is the previous row's target, and undefined on the first row.
        assert np.isnan(block["lag_1"].to_numpy()[0])
        assert np.allclose(block["lag_1"].to_numpy()[1:], target[:-1])

        # The rolling mean must never contain the current observation.
        roll = block["roll_mean"].to_numpy()
        assert np.isnan(roll[0])
        for i in range(1, len(target)):
            assert roll[i] < target[i]


def test_lags_do_not_leak_across_units():
    """Unit 1's history must never appear in unit 0's features."""
    df = add_features(_toy_panel())
    first_rows = df.sort_values(["unit_id", "week"]).groupby("unit_id").head(1)
    assert first_rows["lag_1"].isna().all()


def test_folds_never_train_on_the_future():
    panel, _ = make_panel(SynthParams(**SMALL))
    weeks = panel["week"]

    folds = list(expanding_folds(weeks, n_folds=4))
    assert len(folds) >= 2

    for train_mask, test_mask in folds:
        train_weeks = weeks[train_mask]
        test_weeks = weeks[test_mask]
        assert not train_weeks.empty and not test_weeks.empty
        assert train_weeks.max() < test_weeks.min()


def test_folds_expand():
    """Each fold keeps everything the previous one trained on."""
    panel, _ = make_panel(SynthParams(**SMALL))
    sizes = [mask.sum() for mask, _ in expanding_folds(panel["week"], n_folds=4)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_gbm_beats_the_persistence_baseline():
    panel, _ = make_panel(SynthParams(**SMALL))
    scores, _, _, _ = evaluate(add_features(panel), use_history=True, n_folds=3)
    by_name = {s.name: s.rmse for s in scores}
    assert by_name["gbm"] < by_name["persistence"]


def _probe_elasticity(price_on_xi: float) -> tuple[float, float]:
    """Fit the GBM on a synthetic panel and probe it. Returns (implied, truth)."""
    panel, truth = make_panel(SynthParams(**SMALL, price_on_xi=price_on_xi))
    df = add_features(panel)
    cols = _feature_columns(use_history=True)
    work = df.dropna(subset=cols + ["log_units"])

    model = _gbm()
    model.fit(work[cols], work["log_units"])
    return implied_elasticity(model, work, cols), truth["beta_mean"]


def test_probe_reads_a_negative_slope_when_pricing_is_exogenous():
    """With the confound switched off, the probe must see a real price effect.

    This is the control that makes the headline finding meaningful: if the
    perturbation probe could not detect an elasticity even in the easy case,
    a near-zero reading on real data would say nothing about the data.
    """
    implied, truth = _probe_elasticity(price_on_xi=0.0)
    assert truth < -1.0
    assert implied < -0.5


def test_probe_is_pulled_towards_zero_by_endogenous_pricing():
    """Latent quality lifts price and demand together, so the probe understates.

    Same failure as pooled OLS, reached by a completely different model class:
    the bias lives in the data, not in the estimator.
    """
    exogenous, _ = _probe_elasticity(price_on_xi=0.0)
    endogenous, _ = _probe_elasticity(price_on_xi=0.25)
    assert endogenous > exogenous


def test_bump_size_is_small_enough_to_stay_local():
    assert 0.0 < ELASTICITY_BUMP <= 0.10
