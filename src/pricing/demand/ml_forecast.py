"""Supervised demand forecasting, and why the best forecaster is the worst
pricing model.

What this module is for
-----------------------
The rest of the project estimates *one* number per category -- a price
elasticity -- with estimators whose assumptions are the point. This module
does the other thing a data scientist is asked for: predict next week's
quantity as accurately as possible, with no identification story at all.

Both questions are legitimate and they are not the same question. Running
them side by side is the cheapest way to show why:

    forecasting  ->  "what will demand be?"      judged by held-out error
    pricing      ->  "what would demand be IF
                      I changed the price?"      judged by whether the
                                                 counterfactual is credible

A gradient-boosted tree answers the first far better than a log-log
regression. It answers the second *confidently and wrongly*, and the size of
that error is measured in :func:`implied_elasticity`.

Why a price-blind model still wins the forecast
------------------------------------------------
Demand is autocorrelated: what a product sold last week is an excellent
predictor of what it sells this week. Once lagged demand is in the feature
set, the model can reach a good RMSE while learning almost nothing about
price. That is not a bug in the model -- it is the correct solution to the
problem it was given. It becomes a catastrophe only when someone reads the
fitted function as a demand curve and prices against it.

To rule out the objection that this is an artefact of the lag features, the
comparison is run twice: once with the unit's own history, and once with the
lags removed so price and product attributes are all the model has.

Validation protocol
-------------------
Splits are **expanding-window over calendar weeks**, never random. A random
``KFold`` on panel data trains on week 40 and tests on week 12 of the same
product, which leaks the future into the past and inflates every metric. The
folds here always predict weeks the model has not seen.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from pricing.config import RANDOM_SEED
from pricing.data import build_panel

# Lags in weeks. 1 and 2 carry short-run persistence; 4 carries the monthly
# rhythm without reaching so far back that most rows go missing.
LAGS = (1, 2, 4)
ROLL_WINDOW = 4

# Size of the counterfactual price bump used to read an elasticity out of a
# fitted model, in log points (~5%). Small enough to stay local, large enough
# that tree-based step functions actually move.
ELASTICITY_BUMP = 0.05

BASE_FEATURES = ["log_price", "freight", "review_score", "competition"]
HISTORY_FEATURES = [f"lag_{k}" for k in LAGS] + ["roll_mean", "lag_log_price"]


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach lagged and rolling features, computed strictly within a unit.

    Every lag is taken after sorting by week and grouping by ``unit_id``, so
    no row ever sees another product's history. The rolling mean is shifted
    by one period before averaging: including the current week would put the
    target inside its own feature.
    """
    df = panel.sort_values(["unit_id", "week"]).copy()
    grp = df.groupby("unit_id", observed=True)["log_units"]

    for k in LAGS:
        df[f"lag_{k}"] = grp.shift(k)

    # shift(1) first, then roll: the window ends at t-1, never at t. Both
    # steps have to happen *inside* the group -- rolling over the ungrouped
    # series would slide the window across the boundary between two products
    # and pull the next unit's early weeks into this one's history.
    df["roll_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(ROLL_WINDOW, min_periods=1).mean()
    )
    df["lag_log_price"] = df.groupby("unit_id", observed=True)["log_price"].shift(1)

    df["week_index"] = df["week"].rank(method="dense").astype(int)
    df["category_code"] = df["category"].astype("category").cat.codes
    return df


def _feature_columns(use_history: bool) -> list[str]:
    cols = list(BASE_FEATURES) + ["category_code", "week_index"]
    if use_history:
        cols += HISTORY_FEATURES
    return cols


def expanding_folds(weeks: pd.Series, n_folds: int = 4, min_train_frac: float = 0.5):
    """Yield ``(train_mask, test_mask)`` over an expanding window of weeks.

    The first fold trains on the earliest ``min_train_frac`` of the calendar
    and tests on the block that follows; each later fold moves the boundary
    forward, keeping everything before it for training.
    """
    ordered = np.sort(weeks.unique())
    start = int(len(ordered) * min_train_frac)
    if start < 1 or start >= len(ordered):
        raise ValueError("not enough distinct weeks to build folds")

    edges = np.linspace(start, len(ordered), n_folds + 1).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        train_weeks = ordered[:lo]
        test_weeks = ordered[lo:hi]
        yield weeks.isin(train_weeks).to_numpy(), weeks.isin(test_weeks).to_numpy()


@dataclass
class ModelScore:
    """Held-out error for one model, averaged over the expanding folds."""

    name: str
    rmse: float
    mae: float
    fold_rmse: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "fold_rmse": [round(v, 4) for v in self.fold_rmse],
        }


def _gbm() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def evaluate(df: pd.DataFrame, use_history: bool = True, n_folds: int = 4):
    """Score a persistence baseline, a ridge and a GBM on the same folds.

    Returns ``(scores, fitted_gbm, feature_columns, last_test_frame)``. The
    fitted model and the final test block are handed back so the elasticity
    probe can reuse them instead of refitting.
    """
    cols = _feature_columns(use_history)
    work = df.dropna(subset=cols + ["log_units"]).reset_index(drop=True)

    rows = {"persistence": [], "ridge": [], "gbm": []}
    gbm = None
    last_test = None

    for train_mask, test_mask in expanding_folds(work["week"], n_folds=n_folds):
        train, test = work[train_mask], work[test_mask]
        if train.empty or test.empty:
            continue

        y_train, y_test = train["log_units"], test["log_units"]

        # Persistence: last observed value for the unit. Where the lag is
        # missing (a unit's first appearance in the test block) fall back to
        # the training mean, which is what a naive operator would do.
        if use_history:
            naive = test["lag_1"].fillna(y_train.mean())
        else:
            naive = pd.Series(y_train.mean(), index=test.index)
        rows["persistence"].append((y_test, naive))

        scaler = StandardScaler().fit(train[cols])
        ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        ridge.fit(scaler.transform(train[cols]), y_train)
        rows["ridge"].append((y_test, ridge.predict(scaler.transform(test[cols]))))

        gbm = _gbm()
        gbm.fit(train[cols], y_train)
        rows["gbm"].append((y_test, gbm.predict(test[cols])))

        last_test = test

    scores = []
    for name, pairs in rows.items():
        if not pairs:
            continue
        fold_rmse = [float(np.sqrt(mean_squared_error(a, b))) for a, b in pairs]
        y_all = np.concatenate([np.asarray(a) for a, _ in pairs])
        p_all = np.concatenate([np.asarray(b) for _, b in pairs])
        scores.append(
            ModelScore(
                name=name,
                rmse=float(np.sqrt(mean_squared_error(y_all, p_all))),
                mae=float(mean_absolute_error(y_all, p_all)),
                fold_rmse=fold_rmse,
            )
        )

    return scores, gbm, cols, last_test


def implied_elasticity(model, frame: pd.DataFrame, cols: list[str]) -> float:
    """Read a price elasticity out of a fitted forecaster by perturbation.

    The model never states an elasticity, so we ask it the pricing question
    directly: hold every other feature fixed, move ``log_price`` by
    :data:`ELASTICITY_BUMP`, and measure how far the predicted log quantity
    moves. The ratio is a numerical derivative -- the elasticity the model
    would imply if anyone used it to set prices.

    Holding the lag features fixed is deliberate: it is exactly the
    counterfactual a pricing team would be asking for ("same product, same
    recent history, different price this week").
    """
    base = frame[cols].copy()
    bumped = base.copy()
    bumped["log_price"] = bumped["log_price"] + ELASTICITY_BUMP

    delta = model.predict(bumped) - model.predict(base)
    return float(np.mean(delta) / ELASTICITY_BUMP)


def run(n_folds: int = 4) -> dict:
    """Full comparison: forecast accuracy, then the elasticity each model implies."""
    panel, diagnostics = build_panel()
    df = add_features(panel)

    report: dict = {"panel": diagnostics, "settings": {"folds": n_folds}}

    for label, use_history in (("with_history", True), ("price_only", False)):
        scores, gbm, cols, last_test = evaluate(
            df, use_history=use_history, n_folds=n_folds
        )
        block = {"scores": [s.as_dict() for s in scores]}
        if gbm is not None and last_test is not None and not last_test.empty:
            block["implied_elasticity"] = round(
                implied_elasticity(gbm, last_test, cols), 4
            )
        report[label] = block

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="emit raw JSON only")
    args = parser.parse_args()

    report = run(n_folds=args.folds)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    panel = report["panel"]
    print(
        f"\nPanel: {panel['rows_kept']} rows, {panel['units_kept']} units, "
        f"{panel['weeks']} weeks\n"
    )
    for label in ("with_history", "price_only"):
        block = report[label]
        print(f"--- {label.replace('_', ' ')} ---")
        print(f"{'model':<14}{'RMSE':>9}{'MAE':>9}")
        for s in block["scores"]:
            print(f"{s['model']:<14}{s['rmse']:>9.4f}{s['mae']:>9.4f}")
        if "implied_elasticity" in block:
            print(f"implied elasticity from the GBM: {block['implied_elasticity']:+.4f}")
        print()


if __name__ == "__main__":
    main()
