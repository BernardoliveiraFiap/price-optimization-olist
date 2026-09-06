"""Does the estimator actually recover the truth?

On real data the true elasticity is unobservable, so "my model says -1.4" is
an unfalsifiable claim. Here the data comes from :mod:`pricing.synth`, where
the true beta is written down before any estimation happens, so each estimator
can be scored on bias and on whether its confidence interval covers the truth.

Run it directly::

    python -m pricing.evaluate.validation --reps 20

The expected pattern -- and the reason the pipeline does not stop at OLS:

* pooled OLS is biased *towards zero*, because sellers raise prices when
  demand is hot, so part of the positive demand shock is credited to price;
* two-way FE removes the persistent part of that shock and cuts the bias;
* 2SLS with a cost shifter removes what is left, at the cost of a wider
  interval.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..elasticity.loglog_fe import estimate_iv, estimate_ols, estimate_twoway_fe
from ..synth import SynthParams, make_panel

ESTIMATORS = {
    "Pooled OLS": estimate_ols,
    "Two-way FE": estimate_twoway_fe,
    "2SLS (cost IV)": estimate_iv,
}


def run_once(seed: int, endogenous: bool = True) -> pd.DataFrame:
    """One synthetic market, all estimators, scored against the truth."""
    params = SynthParams(seed=seed, price_on_xi=0.60 if endogenous else 0.0)
    panel, truth = make_panel(params)
    true_beta = truth["beta_mean"]

    rows = []
    for name, fn in ESTIMATORS.items():
        res = fn(panel)
        rows.append(
            {
                "seed": seed,
                "method": name,
                "true_beta": true_beta,
                "estimate": res.elasticity,
                "bias": res.elasticity - true_beta,
                "rel_bias_pct": 100.0 * (res.elasticity - true_beta) / abs(true_beta),
                "std_error": res.std_error,
                "covers_truth": bool(res.ci_low <= true_beta <= res.ci_high),
            }
        )
    return pd.DataFrame(rows)


def run_validation(reps: int = 20, endogenous: bool = True) -> pd.DataFrame:
    """Monte Carlo over independent synthetic markets."""
    frames = [run_once(seed, endogenous) for seed in range(reps)]
    raw = pd.concat(frames, ignore_index=True)

    summary = (
        raw.groupby("method", sort=False)
        .agg(
            true_beta=("true_beta", "mean"),
            mean_estimate=("estimate", "mean"),
            bias=("bias", "mean"),
            rel_bias_pct=("rel_bias_pct", "mean"),
            rmse=("bias", lambda s: float(np.sqrt(np.mean(s**2)))),
            ci_coverage=("covers_truth", "mean"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument(
        "--exogenous",
        action="store_true",
        help="switch the endogeneity off; every estimator should then be unbiased",
    )
    args = parser.parse_args()

    summary = run_validation(reps=args.reps, endogenous=not args.exogenous)
    label = "EXOGENOUS pricing" if args.exogenous else "ENDOGENOUS pricing"
    print(f"\nEstimator recovery on synthetic data -- {label}, {args.reps} markets\n")
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda v: f"{v: .3f}"))
    print(
        "\nrel_bias_pct > 0 means the estimate is closer to zero than the truth,"
        "\ni.e. demand looks less price-sensitive than it really is.\n"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
