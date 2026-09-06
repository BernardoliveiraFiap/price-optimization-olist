"""Central paths and project-wide constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

PANEL_PATH = DATA_PROCESSED / "panel.parquet"

# Olist files expected under data/raw/ (Kaggle: olistbr/brazilian-ecommerce).
OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

RANDOM_SEED = 20260906

# Panel construction -----------------------------------------------------
FREQ = "W-MON"           # weekly buckets, Monday-anchored
MIN_WEEKS_PER_UNIT = 12  # a unit needs this much history to identify a slope
MIN_UNITS_PER_WEEK = 1

# Optimisation -----------------------------------------------------------
PRICE_GRID_STEPS = 11       # candidate prices per unit
PRICE_BAND = 0.30           # allow +/- 30% around the observed price
MAX_AVG_PRICE_INCREASE = 0.05   # portfolio-level guardrail (5%)
MIN_REVENUE_RATIO = 0.98        # do not lose more than 2% of revenue
DEFAULT_MARGIN_RATE = 0.35      # gross margin assumption (Olist has no cost data)
