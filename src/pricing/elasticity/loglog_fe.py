"""Log-log price elasticity estimators: pooled OLS, two-way FE, and 2SLS.

All three answer the same question -- "if price rises 1%, how much does
quantity fall?" -- under progressively weaker assumptions:

* :func:`estimate_ols` assumes price is as good as randomly assigned once a
  handful of linear controls are included. It never is: sellers raise prices
  exactly when demand is strong, so price correlates with the error term and
  the estimate is biased *towards zero* (demand looks less elastic than it is).
* :func:`estimate_twoway_fe` sweeps out anything constant within a product
  (intrinsic desirability, brand) or within a week (seasonality, macro). What
  is left is bias from shocks that vary *within* a product over time.
* :func:`estimate_iv` instruments price with a cost shifter, the only one of
  the three that survives a time-varying unobserved demand shock -- as long as
  the exclusion restriction holds.

Reporting all three side by side is the point. The spread between them is the
honest uncertainty about identification, and it is handed to the optimiser as
a range instead of being collapsed into one confident number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_CONTROLS = ("log_freight", "competition", "review_score")

# Staiger-Stock rule of thumb. Below this the 2SLS point estimate is not
# merely imprecise, it is biased toward OLS and its confidence interval
# has no meaningful coverage, so it must not be used for a decision.
WEAK_INSTRUMENT_F = 10.0


@dataclass
class ElasticityResult:
    method: str
    elasticity: float
    std_error: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_units: int
    diagnostics: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"{self.method:<22} beta = {self.elasticity:+.3f} "
            f"(SE {self.std_error:.3f})  95% CI "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
        )

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "elasticity": self.elasticity,
            "std_error": self.std_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_obs": self.n_obs,
            "n_units": self.n_units,
            **self.diagnostics,
        }


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the columns every estimator expects. Idempotent."""
    out = df.copy()
    if "log_price" not in out:
        out["log_price"] = np.log(out["price"])
    if "log_units" not in out:
        out["log_units"] = np.log(out["units"])
    if "log_freight" not in out and "freight" in out:
        out["log_freight"] = np.log1p(out["freight"])
    return out


def _available(df: pd.DataFrame, controls: tuple[str, ...]) -> list[str]:
    return [c for c in controls if c in df.columns]


def _within_transform(
    df: pd.DataFrame,
    cols: list[str],
    unit_col: str = "unit_id",
    time_col: str = "week",
    max_iter: int = 50,
    tol: float = 1e-10,
) -> pd.DataFrame:
    """Two-way within transformation by alternating projections.

    For a balanced panel, one pass of ``x - mean_i - mean_t + mean`` is exact.
    Real panels are unbalanced, where that closed form is only an
    approximation, so the two projections are iterated until the group means
    have effectively vanished.
    """
    out = df[cols].astype(float).reset_index(drop=True)
    unit = pd.Series(df[unit_col].to_numpy())
    time = pd.Series(df[time_col].to_numpy())

    for _ in range(max_iter):
        before = out.to_numpy(copy=True)
        out = out - out.groupby(unit, observed=True).transform("mean")
        out = out - out.groupby(time, observed=True).transform("mean")
        if np.nanmax(np.abs(out.to_numpy() - before)) < tol:
            break
    return out


def _within_varying(df: pd.DataFrame, controls: list[str]) -> list[str]:
    """Controls with no within-unit variation are absorbed by the unit FE.

    Keeping them would just add perfectly collinear columns.
    """
    keep = []
    for c in controls:
        spread = df.groupby("unit_id")[c].std().fillna(0.0).max()
        if spread > 1e-12:
            keep.append(c)
    return keep


def estimate_ols(
    df: pd.DataFrame, controls: tuple[str, ...] = DEFAULT_CONTROLS
) -> ElasticityResult:
    """Pooled OLS with linear controls. The naive benchmark."""
    d = add_model_columns(df)
    ctrl = _available(d, controls)
    d = d.dropna(subset=["log_units", "log_price", *ctrl])

    X = np.column_stack([np.ones(len(d)), d["log_price"], *[d[c] for c in ctrl]])
    res = sm.OLS(d["log_units"].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["unit_id"].to_numpy()}
    )
    beta, se = float(res.params[1]), float(res.bse[1])
    return ElasticityResult(
        method="Pooled OLS",
        elasticity=beta,
        std_error=se,
        ci_low=beta - 1.96 * se,
        ci_high=beta + 1.96 * se,
        n_obs=len(d),
        n_units=int(d["unit_id"].nunique()),
        diagnostics={"r_squared": float(res.rsquared)},
    )


def estimate_twoway_fe(
    df: pd.DataFrame, controls: tuple[str, ...] = DEFAULT_CONTROLS
) -> ElasticityResult:
    """Product + week fixed effects, standard errors clustered by product."""
    d = add_model_columns(df)
    ctrl = _available(d, controls)
    d = d.dropna(subset=["log_units", "log_price", *ctrl])

    varying = _within_varying(d, ctrl)
    W = _within_transform(d, ["log_units", "log_price", *varying])

    X = np.column_stack([W["log_price"], *[W[c] for c in varying]])
    res = sm.OLS(W["log_units"].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": d["unit_id"].to_numpy()}
    )
    beta, se = float(res.params[0]), float(res.bse[0])
    return ElasticityResult(
        method="Two-way FE",
        elasticity=beta,
        std_error=se,
        ci_low=beta - 1.96 * se,
        ci_high=beta + 1.96 * se,
        n_obs=len(d),
        n_units=int(d["unit_id"].nunique()),
        diagnostics={"absorbed_controls": len(ctrl) - len(varying)},
    )


def estimate_iv(
    df: pd.DataFrame,
    instrument: str = "cost_index",
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
) -> ElasticityResult:
    """Two-stage least squares on the within-transformed panel.

    Stage 1 projects log price onto the cost shifter; stage 2 regresses log
    quantity on the fitted price. The covariance is the clustered 2SLS
    sandwich -- residuals are formed with the *actual* price, not the fitted
    one, otherwise the standard error is understated.
    """
    d = add_model_columns(df)
    if instrument not in d.columns:
        raise KeyError(f"instrument column {instrument!r} not in panel")
    ctrl = _available(d, controls)
    d = d.dropna(subset=["log_units", "log_price", instrument, *ctrl])

    varying = _within_varying(d, ctrl)
    W = _within_transform(d, ["log_units", "log_price", instrument, *varying])

    y = W["log_units"].to_numpy()
    price = W["log_price"].to_numpy()
    inst = W[instrument].to_numpy()
    exog = (
        np.column_stack([W[c] for c in varying])
        if varying
        else np.empty((len(d), 0))
    )
    groups = d["unit_id"].to_numpy()

    # Stage 1 ------------------------------------------------------------
    Z = np.column_stack([inst, exog]) if exog.size else inst.reshape(-1, 1)
    first = sm.OLS(price, Z).fit(cov_type="cluster", cov_kwds={"groups": groups})
    price_hat = np.asarray(first.fittedvalues)
    first_stage_f = float(first.tvalues[0] ** 2)  # weak-instrument diagnostic

    # Stage 2 ------------------------------------------------------------
    X2 = (
        np.column_stack([price_hat, exog]) if exog.size else price_hat.reshape(-1, 1)
    )
    params = np.linalg.lstsq(X2, y, rcond=None)[0]
    beta = float(params[0])

    X_actual = np.column_stack([price, exog]) if exog.size else price.reshape(-1, 1)
    resid = y - X_actual @ params
    bread = np.linalg.pinv(X2.T @ X2)
    meat = np.zeros((X2.shape[1], X2.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        u = (X2[m] * resid[m][:, None]).sum(axis=0)
        meat += np.outer(u, u)
    n_g = len(np.unique(groups))
    cov = bread @ meat @ bread * (n_g / max(n_g - 1, 1))
    se = float(np.sqrt(np.diag(cov))[0])

    return ElasticityResult(
        method="2SLS (cost IV)",
        elasticity=beta,
        std_error=se,
        ci_low=beta - 1.96 * se,
        ci_high=beta + 1.96 * se,
        n_obs=len(d),
        n_units=int(d["unit_id"].nunique()),
        diagnostics={
            "first_stage_F": first_stage_f,
            "weak_instrument": bool(first_stage_f < WEAK_INSTRUMENT_F),
        },
    )


def estimate_by_category(
    df: pd.DataFrame,
    estimator=estimate_twoway_fe,
    min_obs: int = 200,
    min_units: int = 5,
) -> pd.DataFrame:
    """Run an estimator separately per category.

    Heterogeneous elasticities are what make the optimiser interesting: a
    single portfolio-wide number would push every price in the same direction.
    """
    rows = []
    for category, chunk in df.groupby("category", observed=True):
        if len(chunk) < min_obs or chunk["unit_id"].nunique() < min_units:
            continue
        try:
            res = estimator(chunk)
        except Exception as exc:  # pragma: no cover - defensive
            rows.append(
                {"category": category, "elasticity": np.nan, "error": str(exc)}
            )
            continue
        rows.append({"category": category, **res.as_row()})
    return pd.DataFrame(rows).sort_values("elasticity").reset_index(drop=True)
