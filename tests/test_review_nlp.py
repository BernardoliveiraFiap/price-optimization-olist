"""Weak-supervision rules, tested without downloading an encoder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pricing.demand.review_nlp import (
    fit_heads,
    product_quality_signal,
    score_reviews,
    weak_labels,
)


def test_weak_labels_separate_the_two_topics():
    comments = pd.Series(
        [
            "a entrega atrasou duas semanas",        # delivery only
            "produto de otima qualidade, recomendo",  # product only
            "chegou rapido mas o produto veio quebrado",  # both -> not labelled
            "",                                        # neither -> not labelled
        ]
    )
    labels = weak_labels(comments)
    assert labels["is_delivery"].tolist() == [1, 0, 1, 0]
    assert labels["is_product"].tolist() == [0, 1, 1, 0]
    # Only the unambiguous ones become training rows.
    assert labels["labelled"].tolist() == [1, 1, 0, 0]


def test_heads_learn_a_separable_signal():
    """Two well-separated clusters must be learnable, or the head is broken."""
    rng = np.random.default_rng(0)
    n = 200
    delivery = rng.normal(1.0, 0.2, size=(n, 16))
    product = rng.normal(-1.0, 0.2, size=(n, 16))
    embeddings = np.vstack([delivery, product])
    labels = pd.DataFrame(
        {
            "is_delivery": [1] * n + [0] * n,
            "is_product": [0] * n + [1] * n,
            "labelled": [1] * (2 * n),
        }
    )
    model, metrics = fit_heads(embeddings, labels, epochs=200)
    assert metrics["val_accuracy_delivery"] > 0.9
    assert metrics["val_accuracy_product"] > 0.9

    scores = score_reviews(model, embeddings)
    assert len(scores) == 2 * n
    assert scores["delivery_topic"].between(0, 1).all()


def test_delivery_complaints_are_discounted_from_quality():
    """A 1-star review that is purely about the courier must not sink the
    product's quality signal the way the raw average does."""
    reviews = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "review_score": [1, 5, 5, 5],
        }
    )
    scores = pd.DataFrame(
        {
            "delivery_topic": [0.95, 0.05, 0.05, 0.05],
            "product_topic": [0.05, 0.95, 0.95, 0.95],
        }
    )
    mapping = pd.DataFrame(
        {"order_id": ["o1", "o2", "o3", "o4"], "product_id": ["p1"] * 4}
    )
    out = product_quality_signal(reviews, scores, mapping, min_reviews=1)
    row = out.iloc[0]
    assert row["raw_review_score"] == 4.0
    assert row["product_quality_signal"] > row["raw_review_score"]
