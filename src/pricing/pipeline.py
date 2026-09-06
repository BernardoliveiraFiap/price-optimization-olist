"""End-to-end run: estimate elasticity, optimise prices, price the mistakes.

    python -m pricing.pipeline --source synth
    python -m pricing.pipeline --source olist

Everything downstream of the panel is source-agnostic, so the synthetic market
is not a toy detour: it is the same pipeline with a dataset whose answer is
known.

The pipeline is allowed to refuse. Two gates stand between an estimate and a
price list, and on real Olist data both of them fire:

* an instrument whose first-stage F is below the weak-instrument threshold is
  not used, because a weak-IV estimate is biased toward OLS and its interval
  has no coverage -- reporting it as "the causal estimate" would be worse than
  reporting nothing;
* a category whose elasticity is not credibly below -1 is not priced, because
  with inelastic demand the constant-elasticity margin has no interior
  optimum. The "optimal" price is then whatever the band allows, which is a
  statement about the guardrails rather than about the market.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .config import FIGURES, PANEL_PATH
from .elasticity.loglog_fe import (
    estimate_by_category,
    estimate_iv,
    estimate_ols,
    estimate_twoway_fe,
)
from .evaluate.impact import (
    build_products,
    cost_of_naive_elasticity,
    guardrail_frontier,
    sensitivity_to_elasticity,
    sensitivity_to_margin_rate,
)
from .optimize.milp import OptimisationConfig, optimise_prices, optimise_unconstrained
from .synth import make_panel

REPORTS = FIGURES.parent

# Below 1 in absolute value there is no interior margin optimum, so a price
# recommendation would be an artefact of the price band.
MIN_ABS_ELASTICITY = 1.05
MIN_T_STAT = 2.0


def load_panel(source: str) -> tuple[pd.DataFrame, dict | None]:
    if source == "synth":
        panel, truth = make_panel()
        return panel, truth
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{PANEL_PATH} not found. Put the Olist CSVs in data/raw/ and run "
            "`python -m pricing.data` first."
        )
    return pd.read_parquet(PANEL_PATH), None


def credible_betas(table: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Split category estimates into the ones a decision can rest on and the rest.

    A positive estimate is a failed identification, not a Giffen good; an
    estimate between -1 and 0 is a real finding but not an actionable one.
    Both are kept in the reported table and excluded from the optimiser.
    """
    t = table.dropna(subset=["elasticity"]).copy()
    t["t_stat"] = t["elasticity"] / t["std_error"]
    t["credible"] = (t["elasticity"] < -MIN_ABS_ELASTICITY) & (
        t["t_stat"].abs() > MIN_T_STAT
    )
    ok = t[t["credible"]]
    return dict(zip(ok["category"], ok["elasticity"], strict=True)), t


def choose_causal_estimator(panel: pd.DataFrame):
    """Prefer the instrument, but only when the first stage actually holds."""
    if "cost_index" not in panel.columns:
        return estimate_twoway_fe, "no instrument in the panel"

    iv = estimate_iv(panel)
    f_stat = iv.diagnostics["first_stage_F"]
    if iv.diagnostics["weak_instrument"]:
        return estimate_twoway_fe, (
            f"instrument rejected: first-stage F = {f_stat:.2f}, below 10. "
            "2SLS would be biased toward OLS with no interval coverage"
        )
    return estimate_iv, f"instrument accepted: first-stage F = {f_stat:.1f}"


def run(source: str = "synth", write: bool = True) -> dict:
    panel, truth = load_panel(source)

    print(f"\nPanel: {len(panel):,} rows | {panel['unit_id'].nunique():,} products "
          f"| {panel['week'].nunique()} weeks\n")

    # --- 1. identification -------------------------------------------------
    pooled = [estimate_ols(panel), estimate_twoway_fe(panel)]
    if "cost_index" in panel.columns:
        pooled.append(estimate_iv(panel))
    print("Portfolio-wide elasticity")
    for res in pooled:
        print(f"  {res}")
    if truth:
        print(f"  {'TRUE (synthetic)':<22} beta = {truth['beta_mean']:+.3f}")

    estimator, reason = choose_causal_estimator(panel)
    print(f"\nEstimator for the decision: {estimator.__name__}\n  {reason}")

    naive_tbl = estimate_by_category(panel, estimate_ols)
    causal_tbl = estimate_by_category(panel, estimator)
    beta_causal, annotated = credible_betas(causal_tbl)
    beta_naive, _ = credible_betas(naive_tbl)

    print("\nElasticity by category")
    print(
        annotated[["category", "elasticity", "std_error", "t_stat", "n_obs", "credible"]]
        .to_string(index=False, float_format=lambda v: f"{v: .3f}")
    )
    print(
        f"\n  {len(beta_causal)} of {len(annotated)} categories are credibly "
        f"elastic (beta < -{MIN_ABS_ELASTICITY}, |t| > {MIN_T_STAT})"
    )

    if not beta_causal:
        print(
            "\nNo category clears the bar, so no price list is produced.\n"
            "Demand here is inelastic as measured: raising prices would look\n"
            "profitable in the model only because the band stops it, which is a\n"
            "conclusion about the band and not about the market.\n"
        )
        return {"elasticity_by_category": annotated, "optimisation": None}

    # --- 2. optimisation ---------------------------------------------------
    cfg = OptimisationConfig()
    covered = panel[panel["category"].isin(beta_causal)]
    coverage = float(
        (covered["price"] * covered["units"]).sum()
        / (panel["price"] * panel["units"]).sum()
    )
    products = build_products(covered, beta_causal)
    print(
        f"\nOptimiser on {len(products):,} products across "
        f"{len(beta_causal)} categories ({coverage:.0%} of panel revenue)"
    )

    constrained = optimise_prices(products, cfg)
    unconstrained = optimise_unconstrained(products, cfg)
    print(f"  guardrails on   {constrained}")
    print(f"  guardrails off  {unconstrained}")
    guardrail_cost = (
        unconstrained.summary["margin_uplift_pct"]
        - constrained.summary["margin_uplift_pct"]
    )
    print(f"  guardrails cost {guardrail_cost:.2f} pp of margin uplift")

    frontier = guardrail_frontier(products, beta_causal, cfg)
    print("\nWhat each guardrail costs")
    print(frontier.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    # --- 3. what the naive estimate costs ---------------------------------
    shared = {k: v for k, v in beta_naive.items() if k in beta_causal}
    if shared:
        naive_cost = cost_of_naive_elasticity(products, beta_naive, beta_causal, cfg)
        print("\nPricing with the biased elasticity, scored under the credible one")
        print(naive_cost.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    else:
        naive_cost = None
        print(
            "\nPooled OLS produces no usable elasticity for these categories, so\n"
            "there is no naive price list to compare against."
        )

    # --- 4. is the conclusion robust? -------------------------------------
    sens = sensitivity_to_elasticity(products, beta_causal, config=cfg)
    print("\nSensitivity: same price list, different truth")
    print(sens.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    margin_sens = None
    if "unit_cost" not in products.columns:
        margin_sens = sensitivity_to_margin_rate(products, beta_causal, config=cfg)
        print(
            "\nSensitivity to the gross-margin assumption (this dataset has no "
            "cost data,\nso this assumption drives the answer)"
        )
        print(margin_sens.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    if write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        annotated.to_csv(REPORTS / f"elasticity_by_category_{source}.csv", index=False)
        constrained.prices.to_csv(REPORTS / f"price_list_{source}.csv", index=False)
        sens.to_csv(REPORTS / f"sensitivity_{source}.csv", index=False)
        frontier.to_csv(REPORTS / f"guardrail_frontier_{source}.csv", index=False)
        if naive_cost is not None:
            naive_cost.to_csv(REPORTS / f"cost_of_naive_{source}.csv", index=False)
        if margin_sens is not None:
            margin_sens.to_csv(REPORTS / f"margin_sensitivity_{source}.csv", index=False)
        print(f"\nWrote CSV outputs to {REPORTS}")

    return {
        "elasticity_pooled": [r.as_row() for r in pooled],
        "elasticity_by_category": annotated,
        "estimator": estimator.__name__,
        "coverage": coverage,
        "optimisation": constrained.summary,
        "unconstrained": unconstrained.summary,
        "guardrail_frontier": frontier,
        "cost_of_naive": naive_cost,
        "sensitivity": sens,
        "margin_sensitivity": margin_sens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synth", "olist"], default="synth")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    run(source=args.source, write=not args.no_write)


if __name__ == "__main__":  # pragma: no cover
    main()
