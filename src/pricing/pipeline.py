"""End-to-end run: estimate elasticity, optimise prices, price the mistakes.

    python -m pricing.pipeline --source synth
    python -m pricing.pipeline --source olist

Everything downstream of the panel is source-agnostic, so the synthetic market
is not a toy detour: it is the same pipeline with a dataset whose answer is
known.
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
    sensitivity_to_elasticity,
)
from .optimize.milp import OptimisationConfig, optimise_prices, optimise_unconstrained
from .synth import make_panel

REPORTS = FIGURES.parent


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


def _betas(table: pd.DataFrame) -> dict[str, float]:
    """Category -> elasticity, dropping anything that came back positive.

    A positive estimated elasticity is a failed identification, not a Giffen
    good. Those categories fall back to the portfolio mean rather than telling
    the optimiser that raising the price sells more.
    """
    ok = table.dropna(subset=["elasticity"])
    ok = ok[ok["elasticity"] < 0]
    return dict(zip(ok["category"], ok["elasticity"], strict=True))


def run(source: str = "synth", write: bool = True) -> dict:
    panel, truth = load_panel(source)
    has_instrument = "cost_index" in panel.columns

    print(f"\nPanel: {len(panel):,} rows | {panel['unit_id'].nunique():,} products "
          f"| {panel['week'].nunique()} weeks\n")

    # --- 1. identification -------------------------------------------------
    pooled = [estimate_ols(panel), estimate_twoway_fe(panel)]
    if has_instrument:
        pooled.append(estimate_iv(panel))
    print("Portfolio-wide elasticity")
    for res in pooled:
        print(f"  {res}")
    if truth:
        print(f"  {'TRUE (synthetic)':<22} beta = {truth['beta_mean']:+.3f}")

    naive_tbl = estimate_by_category(panel, estimate_ols)
    causal_tbl = estimate_by_category(
        panel, estimate_iv if has_instrument else estimate_twoway_fe
    )
    beta_naive, beta_causal = _betas(naive_tbl), _betas(causal_tbl)

    print("\nElasticity by category (causal estimator)")
    print(
        causal_tbl[["category", "elasticity", "std_error", "n_obs"]].to_string(
            index=False, float_format=lambda v: f"{v: .3f}"
        )
    )

    # --- 2. optimisation ---------------------------------------------------
    cfg = OptimisationConfig()
    products = build_products(panel, beta_causal)
    constrained = optimise_prices(products, cfg)
    unconstrained = optimise_unconstrained(products, cfg)

    print(f"\nOptimiser on {len(products):,} products")
    print(f"  guardrails on   {constrained}")
    print(f"  guardrails off  {unconstrained}")
    guardrail_cost = (
        unconstrained.summary["margin_uplift_pct"]
        - constrained.summary["margin_uplift_pct"]
    )
    print(f"  guardrails cost {guardrail_cost:.2f} pp of margin uplift")

    # --- 3. what the naive estimate costs ---------------------------------
    naive_cost = cost_of_naive_elasticity(products, beta_naive, beta_causal, cfg)
    print("\nPricing with the biased elasticity, scored under the credible one")
    print(naive_cost.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    # --- 4. is the conclusion robust? -------------------------------------
    sens = sensitivity_to_elasticity(products, beta_causal, config=cfg)
    print("\nSensitivity: same price list, different truth")
    print(sens.to_string(index=False, float_format=lambda v: f"{v: .3f}"))

    if write:
        REPORTS.mkdir(parents=True, exist_ok=True)
        causal_tbl.to_csv(REPORTS / f"elasticity_by_category_{source}.csv", index=False)
        constrained.prices.to_csv(REPORTS / f"price_list_{source}.csv", index=False)
        naive_cost.to_csv(REPORTS / f"cost_of_naive_{source}.csv", index=False)
        sens.to_csv(REPORTS / f"sensitivity_{source}.csv", index=False)
        print(f"\nWrote CSV outputs to {REPORTS}")

    return {
        "elasticity_pooled": [r.as_row() for r in pooled],
        "elasticity_by_category": causal_tbl,
        "optimisation": constrained.summary,
        "unconstrained": unconstrained.summary,
        "cost_of_naive": naive_cost,
        "sensitivity": sens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synth", "olist"], default="synth")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    run(source=args.source, write=not args.no_write)


if __name__ == "__main__":  # pragma: no cover
    main()
