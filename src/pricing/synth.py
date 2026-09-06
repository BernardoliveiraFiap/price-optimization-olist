"""Synthetic marketplace panel with a *known* price elasticity.

This module exists for one reason: on real data the true elasticity is
unobservable, so an estimate cannot be validated. Here the data-generating
process (DGP) is written down explicitly, which turns "does my estimator
work?" into a question with a checkable answer.

The DGP deliberately reproduces the two failure modes that break naive
price-response models:

1. **Unobserved demand shocks.** A latent shock ``xi`` moves demand and also
   pushes the seller's price up (people charge more when demand is hot).
   ``corr(log p, error) > 0`` biases OLS *towards zero*, i.e. demand looks
   less elastic than it is. Fixed effects only remove the persistent part.
2. **Non-linear observed confounding.** Freight and competition enter demand
   through non-linear functions. A linear control is misspecified; a flexible
   ML nuisance model (see :mod:`pricing.elasticity.dml`) is not.

A cost shifter ``z`` (wholesale cost index) is generated as a valid
instrument: it moves price, and is excluded from demand by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import RANDOM_SEED


@dataclass(frozen=True)
class SynthParams:
    """Ground truth for the synthetic market."""

    n_units: int = 300
    n_weeks: int = 104
    n_categories: int = 6

    # True elasticity per category, drawn in this range (always negative).
    elasticity_range: tuple[float, float] = (-2.6, -0.9)
    # Latent product quality: the classic omitted variable. It lifts both
    # willingness to pay and the price a seller can charge, and it is
    # time-invariant -- so a product fixed effect removes it and pooled OLS
    # cannot.
    quality_on_demand: float = 0.70
    quality_on_price: float = 0.50

    # Latent demand shock: AR(1) persistence and innovation scale.
    xi_rho: float = 0.55
    xi_sd: float = 0.30

    # Cost shifter (instrument): AR(1).
    cost_rho: float = 0.70
    cost_sd: float = 0.30

    # Pricing rule. Week-to-week price moves are mostly cost- and
    # promotion-driven, with a smaller demand-chasing component -- the usual
    # picture for a long-tail marketplace of small sellers.
    # ``price_on_xi`` is the endogeneity knob: set it to 0.0 and OLS becomes
    # consistent, which is a useful sanity check in tests.
    price_on_cost: float = 0.90
    price_on_xi: float = 0.25
    price_noise_sd: float = 0.12

    demand_noise_sd: float = 0.25
    seasonal_amp: float = 0.25
    trend_per_year: float = 0.10


    seed: int = RANDOM_SEED
    category_names: tuple[str, ...] = field(
        default=("bed_bath_table", "health_beauty", "sports_leisure",
                 "computers_accessories", "furniture_decor", "watches_gifts")
    )


def _ar1(rng: np.random.Generator, n_units: int, n_weeks: int,
         rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) paths, one per unit, shape (n_units, n_weeks)."""
    out = np.empty((n_units, n_weeks))
    # Start from the stationary distribution so there is no burn-in artefact.
    out[:, 0] = rng.normal(0.0, sd / np.sqrt(1.0 - rho**2), size=n_units)
    innovations = rng.normal(0.0, sd, size=(n_units, n_weeks))
    for t in range(1, n_weeks):
        out[:, t] = rho * out[:, t - 1] + innovations[:, t]
    return out


def make_panel(params: SynthParams | None = None) -> tuple[pd.DataFrame, dict]:
    """Generate the panel and return it alongside the ground-truth dict."""
    p = params or SynthParams()
    rng = np.random.default_rng(p.seed)

    n, t = p.n_units, p.n_weeks
    unit_ids = np.arange(n)
    category_id = rng.integers(0, p.n_categories, size=n)

    # --- ground truth ----------------------------------------------------
    beta_by_category = rng.uniform(*p.elasticity_range, size=p.n_categories)
    beta_unit = beta_by_category[category_id]

    # --- latent processes ------------------------------------------------
    xi = _ar1(rng, n, t, p.xi_rho, p.xi_sd)          # unobserved demand shock
    cost = _ar1(rng, n, t, p.cost_rho, p.cost_sd)     # observed instrument

    # --- observed confounders --------------------------------------------
    # Competition (number of rival offers) reacts to the demand shock too, so
    # it is a genuine confounder rather than decoration.
    competition = np.clip(
        rng.poisson(lam=np.exp(0.9 + 0.8 * xi)), 1, 40
    ).astype(float)
    base_freight = rng.gamma(shape=4.0, scale=4.0, size=n)[:, None]
    freight = base_freight * np.exp(rng.normal(0.0, 0.15, size=(n, t)))

    # Latent quality: never handed to any estimator. The star rating is only a
    # noisy proxy for it, which is why controlling for reviews narrows the
    # omitted-variable bias without closing it.
    quality = rng.normal(0.0, 1.0, size=n)[:, None]
    review_score = np.clip(
        4.1 + 0.35 * quality + rng.normal(0.0, 0.35, size=n)[:, None], 1.0, 5.0
    )

    # --- prices -----------------------------------------------------------
    log_base_price = (                                        # ~R$ 50 median
        3.9
        + p.quality_on_price * quality
        + rng.normal(0.0, 0.35, size=n)[:, None]
    )
    log_price = (
        log_base_price
        + p.price_on_cost * cost
        + p.price_on_xi * xi
        + 0.010 * competition
        + rng.normal(0.0, p.price_noise_sd, size=(n, t))
    )

    # --- demand -----------------------------------------------------------
    alpha = (                                        # baseline unit popularity
        1.6
        + p.quality_on_demand * quality
        + rng.normal(0.0, 0.35, size=n)[:, None]
    )
    week_idx = np.arange(t)[None, :]
    gamma = (
        p.seasonal_amp * np.sin(2 * np.pi * week_idx / 52.0)
        + p.trend_per_year * week_idx / 52.0
    )
    # Non-linear observed confounding: log for freight, sqrt for competition.
    f_freight = -0.22 * np.log1p(freight)
    g_comp = -0.10 * np.sqrt(competition)

    log_q = (
        alpha
        + gamma
        + beta_unit[:, None] * log_price
        + f_freight
        + g_comp
        + 0.15 * review_score
        + xi
        + rng.normal(0.0, p.demand_noise_sd, size=(n, t))
    )
    dates = pd.date_range("2017-01-02", periods=t, freq="W-MON")
    panel = pd.DataFrame(
        {
            "unit_id": np.repeat(unit_ids, t),
            "week": np.tile(dates, n),
            "category": np.repeat(
                [p.category_names[c % len(p.category_names)] for c in category_id], t
            ),
            "price": np.exp(log_price).ravel(),
            "units": np.exp(log_q).ravel(),
            "freight": freight.ravel(),
            "competition": competition.ravel(),
            "review_score": np.repeat(review_score.ravel(), t),
            # Observed cost shifter, usable as an instrument.
            "cost_index": cost.ravel(),
            # Kept only for diagnostics/tests; never fed to an estimator.
            "_xi_latent": xi.ravel(),
        }
    )
    panel["revenue"] = panel["price"] * panel["units"]
    panel["log_price"] = np.log(panel["price"])
    panel["log_units"] = np.log(panel["units"])

    truth = {
        "beta_by_category": {
            p.category_names[c % len(p.category_names)]: float(beta_by_category[c])
            for c in range(p.n_categories)
        },
        # Unit-weighted average elasticity: the target of a pooled estimator.
        "beta_mean": float(np.mean(beta_unit)),
        "params": p,
    }
    return panel, truth
