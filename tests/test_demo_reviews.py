"""Guards the claim the demo makes about itself."""

from __future__ import annotations

import pandas as pd

from pricing.demand.demo_reviews import (
    KEYWORD_FREE_EXPECTATION,
    SAMPLE_REVIEWS,
    build_frames,
)
from pricing.demand.review_nlp import weak_labels


def test_frames_line_up():
    reviews, mapping = build_frames()
    assert len(reviews) == len(SAMPLE_REVIEWS)
    assert reviews["order_id"].is_unique
    assert set(reviews["order_id"]) == set(mapping["order_id"])


def test_keyword_free_reviews_really_are_keyword_free():
    """The demo's whole point is that these reviews cannot be labelled by rule.

    If a lexicon term ever creeps into one of them it would quietly enter the
    training set, and the reported accuracy would stop being held out.
    """
    reviews, _ = build_frames()
    labels = weak_labels(reviews["comment"])
    flagged = pd.concat([reviews[["order_id", "comment"]], labels], axis=1)
    subset = flagged[flagged["order_id"].isin(KEYWORD_FREE_EXPECTATION)]

    assert len(subset) == len(KEYWORD_FREE_EXPECTATION)
    leaked = subset[(subset["is_delivery"] == 1) | (subset["is_product"] == 1)]
    assert leaked.empty, (
        "these are supposed to match neither lexicon:\n"
        + leaked[["order_id", "comment"]].to_string(index=False)
    )


def test_every_product_has_enough_reviews_to_aggregate():
    _, mapping = build_frames()
    assert (mapping.groupby("product_id").size() >= 3).all()
