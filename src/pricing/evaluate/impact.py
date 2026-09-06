"""Turning a price list into a number, and stress-testing that number.

Two things separate this from a model report:

1. **The uplift is model-based and says so.** No counterfactual exists -- no
   one re-ran 2017 at different prices -- so the honest claim is "under the
   estimated demand curve, this price list earns X% more margin", plus how
   fast that claim decays when the curve is wrong.
2. **The cost of the naive estimate is priced.** The optimiser is run twice:
   once believing the biased pooled-OLS elasticity, once believing the IV
   estimate. Both price lists are then judged against the *same* assumed
   truth. The gap is what the endogeneity problem costs in reais -- which is
   the reason the econometrics section exists at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_MARGIN_RATE


def build_products(
    panel: pd.DataFrame,
    elasticity_by_category: dict[str, float] | float,
    recent_weeks: int = 8,
    min_units: float = 0.5,
) -> pd.DataFrame:
    """Collapse the panel into the "today" snapshot the optimiser prices.

    Current price and baseline demand are averages over the last
    ``recent_weeks`` weeks, which is steadier than a single week without
    dragging in prices from a year ago.
    """
    weeks = np.sort(panel["week"].unique())[-recent_weeks:]
    recent = panel[panel["week"].isin(weeks)]

    agg = {"price": "mean", "units": "mean"}
    if "unit_cost" in recent.columns:
        agg["unit_cost"] = "mean"
    if "category" in recent.columns:
        products = (
            recent.groupby(["unit_id", "category"], observed=True)
            .agg(agg)
            .reset_index()
        )
    else:
        products = recent.groupby("unit_id").agg(agg).reset_index()

    products = products[products["units"] >= min_units].copy()

    if isinstance(elasticity_by_category, dict):
        fallback = float(np.mean(list(elasticity_by_category.values())))
        products["elasticity"] = (
            products["category"].map(elasticity_by_category).fillna(fallback)
        )
    else:
        products["elasticity"] = float(elasticity_by_category)

    return products.reset_index(drop=True)


def evaluate_policy(
    priced: pd.DataFrame,
    true_elasticity: dict[str, float] | float | pd.Series,
    margin_rate: float = DEFAULT_MARGIN_RATE,
) -> dict:
    """Score a price list under an assumed *true* demand curve.

    ``priced`` is an :class:`~pricing.optimize.milp.OptimisationResult.prices`
    frame. The optimiser's own belief about elasticity is ignored here: what
    is measured is what happens if reality follows ``true_elasticity``.
    """
    d = priced.copy()
    if isinstance(true_elasticity, dict):
        fallback = float(np.mean(list(true_elasticity.values())))
        beta = d["category"].map(true_elasticity).fillna(fallback)
    elif isinstance(true_elasticity, pd.Series):
        beta = d["unit_id"].map(true_elasticity)
    else:
        beta = pd.Series(float(true_elasticity), index=d.index)

    ratio = d["price_after"] / d["price_before"]
    units_real = d["units_before"] * ratio**beta
    cost = (
        d["unit_cost"]
        if "unit_cost" in d.columns
        else d["price_before"] * (1.0 - margin_rate)
    )

    revenue_before = float((d["price_before"] * d["units_before"]).sum())
    margin_before = float(((d["price_before"] - cost) * d["units_before"]).sum())
    revenue_after = float((d["price_after"] * units_real).sum())
    margin_after = float(((d["price_after"] - cost) * units_real).sum())

    return {
        "revenue_before": revenue_before,
        "revenue_after": revenue_after,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "margin_uplift_pct": 100.0 * (margin_after / margin_before - 1.0),
        "revenue_uplift_pct": 100.0 * (revenue_after / revenue_before - 1.0),
        "volume_change_pct": 100.0
        * (float(units_real.sum()) / float(d["units_before"].sum()) - 1.0),
    }
