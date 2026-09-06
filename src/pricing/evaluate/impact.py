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
from ..optimize.milp import OptimisationConfig, optimise_prices


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


def cost_of_naive_elasticity(
    products: pd.DataFrame,
    beta_naive: dict[str, float] | float,
    beta_causal: dict[str, float] | float,
    config: OptimisationConfig | None = None,
) -> pd.DataFrame:
    """Price with the biased estimate, get judged by the credible one.

    Returns one row per belief, both scored under ``beta_causal``. The
    difference is the margin left on the table by trusting pooled OLS.
    """
    cfg = config or OptimisationConfig()
    rows = []
    for label, belief in (("naive OLS belief", beta_naive), ("IV belief", beta_causal)):
        p = products.copy()
        if isinstance(belief, dict):
            fallback = float(np.mean(list(belief.values())))
            p["elasticity"] = p["category"].map(belief).fillna(fallback)
        else:
            p["elasticity"] = float(belief)

        result = optimise_prices(p, cfg)
        scored = evaluate_policy(result.prices, beta_causal, cfg.margin_rate)
        rows.append(
            {
                "priced_with": label,
                "scored_under": "IV elasticity",
                "avg_price_change_pct": result.summary["avg_price_change_pct"],
                **{k: scored[k] for k in
                   ("margin_uplift_pct", "revenue_uplift_pct", "volume_change_pct")},
            }
        )

    out = pd.DataFrame(rows)
    out["margin_left_on_table_pct"] = (
        out["margin_uplift_pct"].max() - out["margin_uplift_pct"]
    )
    return out


def sensitivity_to_elasticity(
    products: pd.DataFrame,
    beta_belief: dict[str, float] | float,
    scale_grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    config: OptimisationConfig | None = None,
) -> pd.DataFrame:
    """How much does the uplift move if the demand curve is wrong?

    The price list is fixed (optimised once, under ``beta_belief``) and then
    re-scored against elasticities scaled up and down. A conclusion that
    survives a 40% error in beta is worth acting on; one that does not is a
    conclusion about the model, not about the market.
    """
    cfg = config or OptimisationConfig()
    p = products.copy()
    if isinstance(beta_belief, dict):
        fallback = float(np.mean(list(beta_belief.values())))
        p["elasticity"] = p["category"].map(beta_belief).fillna(fallback)
    else:
        p["elasticity"] = float(beta_belief)

    result = optimise_prices(p, cfg)

    rows = []
    for scale in scale_grid:
        if isinstance(beta_belief, dict):
            truth = {k: v * scale for k, v in beta_belief.items()}
        else:
            truth = float(beta_belief) * scale
        scored = evaluate_policy(result.prices, truth, cfg.margin_rate)
        rows.append(
            {
                "elasticity_scale": scale,
                "assumed_beta_mean": (
                    float(np.mean(list(beta_belief.values())))
                    if isinstance(beta_belief, dict)
                    else float(beta_belief)
                )
                * scale,
                **{k: scored[k] for k in
                   ("margin_uplift_pct", "revenue_uplift_pct", "volume_change_pct")},
            }
        )
    return pd.DataFrame(rows)
