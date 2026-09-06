"""Build the modelling panel from the raw Olist CSVs.

    python -m pricing.data --level product

The output has the same shape the synthetic generator produces, so every
downstream module is indifferent to which one it is fed.

Two choices deserve to be argued rather than assumed:

**Unit of analysis.** Olist is a long-tail marketplace: most of its ~32k
products were sold a handful of times, and a product with three sales in two
years carries no usable price variation. ``--level product`` keeps only
products with enough weeks of history; ``--level category`` aggregates to
category x week, trading product detail for far more precise slopes. The ETL
reports how many units survive at each level so the choice is made on the
data rather than on taste.

**The instrument.** Olist records no costs, so the cost shifter that makes the
synthetic 2SLS work has no direct counterpart. The stand-in is a Hausman-style
instrument: the average price of the *same category, same week, in other
states*, excluding the focal unit's own transactions. The argument is that
sellers across regions share input and freight cost shocks, while demand
shocks are regional. That exclusion restriction is an assumption, not a fact --
it fails if national demand shocks (a holiday, a trend) hit every region at
once. Week fixed effects absorb exactly those national shocks, which is why
the IV is always run *inside* the two-way within transformation.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import (
    DATA_RAW,
    MIN_WEEKS_PER_UNIT,
    OLIST_FILES,
    PANEL_PATH,
)


def _read(name: str) -> pd.DataFrame:
    path = DATA_RAW / OLIST_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\nDownload the Olist dataset from "
            "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            f"and extract the CSVs into {DATA_RAW}"
        )
    return pd.read_csv(path)


def load_transactions() -> pd.DataFrame:
    """One row per sold item, with the columns the panel needs."""
    orders = _read("orders")
    items = _read("items")
    products = _read("products")
    customers = _read("customers")
    reviews = _read("reviews")
    translation = _read("translation")

    orders = orders[orders["order_status"] == "delivered"]
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    tx = items.merge(
        orders[["order_id", "customer_id", "order_purchase_timestamp"]],
        on="order_id",
        how="inner",
    )
    tx = tx.merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    tx = tx.merge(
        translation, on="product_category_name", how="left"
    )
    tx = tx.merge(
        customers[["customer_id", "customer_state"]], on="customer_id", how="left"
    )

    review_by_order = (
        reviews.groupby("order_id", as_index=False)["review_score"].mean()
    )
    tx = tx.merge(review_by_order, on="order_id", how="left")

    tx["category"] = (
        tx["product_category_name_english"]
        .fillna(tx["product_category_name"])
        .fillna("unknown")
    )
    tx["week"] = tx["order_purchase_timestamp"].dt.to_period("W").dt.start_time
    tx = tx.rename(columns={"freight_value": "freight"})
    return tx[
        [
            "order_id",
            "product_id",
            "seller_id",
            "customer_state",
            "category",
            "week",
            "price",
            "freight",
            "review_score",
        ]
    ]


def _hausman_instrument(tx: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    """Mean log price of the same category and week, in *other* states.

    Built by leave-one-out on the state dimension: for each unit-week, the
    average excludes every transaction from the states that unit sold into,
    so the instrument never contains the focal unit's own price.
    """
    tx = tx.copy()
    tx["log_price"] = np.log(tx["price"])

    # Total and count per (category, week) and per (category, week, state).
    cw = tx.groupby(["category", "week"], observed=True)["log_price"].agg(
        ["sum", "count"]
    )
    cws = tx.groupby(["category", "week", "customer_state"], observed=True)[
        "log_price"
    ].agg(["sum", "count"])

    # For each unit-week, subtract the states it actually sold into.
    own = (
        tx.groupby([unit_col, "category", "week", "customer_state"], observed=True)
        .size()
        .reset_index(name="_n")
    )
    own = own.merge(
        cws.rename(columns={"sum": "state_sum", "count": "state_count"}),
        left_on=["category", "week", "customer_state"],
        right_index=True,
        how="left",
    )
    excl = (
        own.groupby([unit_col, "category", "week"], observed=True)[
            ["state_sum", "state_count"]
        ]
        .sum()
        .reset_index()
    )
    excl = excl.merge(
        cw.rename(columns={"sum": "cat_sum", "count": "cat_count"}),
        left_on=["category", "week"],
        right_index=True,
        how="left",
    )

    num = excl["cat_sum"] - excl["state_sum"]
    den = excl["cat_count"] - excl["state_count"]
    excl["cost_index"] = np.where(den > 0, num / den.replace(0, np.nan), np.nan)
    return excl[[unit_col, "week", "cost_index"]]


def build_panel(level: str = "product", min_weeks: int = MIN_WEEKS_PER_UNIT):
    """Aggregate transactions into a ``unit_id x week`` panel."""
    tx = load_transactions()
    unit_col = "product_id" if level == "product" else "category"

    # Competition proxy: distinct sellers active in the category that week.
    competition = (
        tx.groupby(["category", "week"], observed=True)["seller_id"]
        .nunique()
        .rename("competition")
        .reset_index()
    )

    panel = (
        tx.groupby([unit_col, "week"], observed=True)
        .agg(
            price=("price", "mean"),
            units=("price", "size"),
            freight=("freight", "mean"),
            review_score=("review_score", "mean"),
            category=("category", "first"),
        )
        .reset_index()
    )
    panel = panel.merge(competition, on=["category", "week"], how="left")

    instrument = _hausman_instrument(tx, unit_col)
    panel = panel.merge(instrument, on=[unit_col, "week"], how="left")

    # Review score is missing for orders never reviewed; fall back to the
    # unit's own mean, then to the global mean, so the control is usable.
    panel["review_score"] = panel.groupby(unit_col, observed=True)[
        "review_score"
    ].transform(lambda s: s.fillna(s.mean()))
    panel["review_score"] = panel["review_score"].fillna(panel["review_score"].mean())

    weeks_per_unit = panel.groupby(unit_col, observed=True)["week"].nunique()
    keep = weeks_per_unit[weeks_per_unit >= min_weeks].index
    kept = panel[panel[unit_col].isin(keep)].copy()

    kept = kept.rename(columns={unit_col: "unit_id"})
    if level == "category":
        kept["category"] = kept["unit_id"]

    kept = kept.dropna(subset=["price", "units", "cost_index"])
    kept = kept[kept["price"] > 0]
    kept["revenue"] = kept["price"] * kept["units"]
    kept["log_price"] = np.log(kept["price"])
    kept["log_units"] = np.log(kept["units"])

    diagnostics = {
        "level": level,
        "units_total": int(panel[unit_col].nunique()),
        "units_kept": int(kept["unit_id"].nunique()),
        "rows_kept": int(len(kept)),
        "weeks": int(kept["week"].nunique()) if len(kept) else 0,
        "min_weeks": min_weeks,
    }
    return kept.reset_index(drop=True), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["product", "category"], default="product")
    parser.add_argument("--min-weeks", type=int, default=MIN_WEEKS_PER_UNIT)
    args = parser.parse_args()

    panel, diag = build_panel(level=args.level, min_weeks=args.min_weeks)
    print("\nPanel built")
    for k, v in diag.items():
        print(f"  {k:<14} {v}")
    if diag["units_kept"] < 20:
        print(
            "\n  Only a handful of units survived the history filter. That is the "
            "long tail talking, not a bug -- try --level category."
        )
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    print(f"\nWrote {PANEL_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
