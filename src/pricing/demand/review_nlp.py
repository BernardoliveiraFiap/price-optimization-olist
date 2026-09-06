"""Splitting the star rating into product quality and delivery experience.

Why a neural model belongs here
-------------------------------
The panel already has a ``review_score``. It is a bad control, because a
1-to-5 star rating on Olist conflates two things that should not enter a
pricing model the same way:

    "produto excelente, mas a entrega atrasou duas semanas"   -> 2 stars
    "chegou super rapido, mas veio quebrado"                  -> 2 stars

Both are 2 stars. Only the first describes a product people would still pay
for. Willingness to pay follows perceived *product* quality; a late courier
depresses the rating without depressing the product's value. Controlling for
the raw star score therefore soaks up variation that belongs to logistics.

Separating the two requires reading the sentence, and the signal lives in free
Portuguese text -- which is exactly the setting where a pretrained language
model beats any feature a person would hand-craft. That is the justification
for reaching for deep learning here, and the only place in this project where
it is reached for.

How it works
------------
1. A pretrained multilingual sentence encoder maps each review comment to a
   384-dimensional vector. It is used frozen: there are no labels to fine-tune
   on, and the encoder already places "veio quebrado" near "produto com
   defeito" without being told they are related.
2. Labels come from **weak supervision**: two keyword lexicons (delivery
   vocabulary, product vocabulary) tag the unambiguous reviews. Roughly a
   third of comments match one lexicon and not the other; those become the
   training set.
3. A small two-head MLP is trained on the frozen embeddings to predict, for
   any review, how much it is about delivery and how much about the product.
   The heads generalise past the keywords -- that is the entire point of using
   embeddings rather than the keyword rules directly, and the held-out score
   reported by :func:`fit_heads` is what says whether they did.

The output is a per-product ``product_quality_signal`` that can replace
``review_score`` as a demand control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Weak-supervision lexicons. Deliberately narrow: precision matters more than
# recall, because the neural heads are what provide recall.
DELIVERY_TERMS = (
    "entrega", "entregue", "chegou", "prazo", "atraso", "atrasou", "atrasada",
    "correio", "correios", "transportadora", "frete", "rastreio", "rastreamento",
    "demorou", "demora", "antes do prazo", "no prazo", "nao chegou", "não chegou",
    "envio", "enviado", "postado",
)
PRODUCT_TERMS = (
    "produto", "qualidade", "material", "acabamento", "quebrado", "quebrou",
    "defeito", "danificado", "veio errado", "tamanho", "cor", "otimo produto",
    "ótimo produto", "excelente produto", "recomendo", "funciona", "resistente",
    "fragil", "frágil", "durabilidade", "original", "falsificado",
)


def _normalise(text: pd.Series) -> pd.Series:
    return text.fillna("").str.lower().str.strip()


def weak_labels(comments: pd.Series) -> pd.DataFrame:
    """Tag reviews that unambiguously talk about delivery or about the product."""
    norm = _normalise(comments)
    delivery = norm.apply(lambda s: any(t in s for t in DELIVERY_TERMS))
    product = norm.apply(lambda s: any(t in s for t in PRODUCT_TERMS))
    return pd.DataFrame(
        {
            "is_delivery": delivery.astype(int),
            "is_product": product.astype(int),
            # Only reviews that hit exactly one lexicon are trusted as labels.
            "labelled": (delivery ^ product).astype(int),
        }
    )


def encode(comments: pd.Series, batch_size: int = 256) -> np.ndarray:
    """Frozen sentence embeddings. Requires ``sentence-transformers``."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "sentence-transformers is needed for the review model:\n"
            "    pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(ENCODER_NAME)
    return model.encode(
        _normalise(comments).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def fit_heads(
    embeddings: np.ndarray,
    labels: pd.DataFrame,
    hidden: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[torch.nn.Module, dict]:  # noqa: F821
    """Train the two-head MLP on the weakly labelled subset.

    Held-out accuracy is reported per head. It is accuracy against *weak*
    labels, not ground truth, so it answers "did the network learn the rule
    and generalise it?" rather than "is the rule correct?" -- a distinction
    worth stating out loud rather than dressing the number up.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mask = labels["labelled"].to_numpy().astype(bool)
    X = torch.tensor(embeddings[mask], dtype=torch.float32)
    y = torch.tensor(
        labels.loc[mask, ["is_delivery", "is_product"]].to_numpy(),
        dtype=torch.float32,
    )

    n_val = max(int(len(X) * val_fraction), 1)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X[train_idx]), y[train_idx])
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(X[val_idx])) > 0.5).float()
        acc = (pred == y[val_idx]).float().mean(dim=0)

    metrics = {
        "n_labelled": int(mask.sum()),
        "n_total": int(len(embeddings)),
        "labelled_share": float(mask.mean()),
        "val_accuracy_delivery": float(acc[0]),
        "val_accuracy_product": float(acc[1]),
        "final_train_loss": float(loss.item()),
    }
    return model, metrics


def score_reviews(model, embeddings: np.ndarray) -> pd.DataFrame:
    """Delivery and product scores in [0, 1] for every review, labelled or not."""
    import torch

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.tensor(embeddings, dtype=torch.float32))
        ).numpy()
    return pd.DataFrame(
        {"delivery_topic": probs[:, 0], "product_topic": probs[:, 1]}
    )


def product_quality_signal(
    reviews: pd.DataFrame,
    scores: pd.DataFrame,
    order_to_product: pd.DataFrame,
    min_reviews: int = 3,
) -> pd.DataFrame:
    """Aggregate to a per-product quality signal.

    Each review's star score is weighted by how much the comment is about the
    product rather than the courier, so a 1-star review that is purely a
    delivery complaint barely moves the product's quality signal.
    """
    d = reviews.reset_index(drop=True).join(scores.reset_index(drop=True))
    d = d.merge(order_to_product, on="order_id", how="inner")

    weight = d["product_topic"] / (d["product_topic"] + d["delivery_topic"] + 1e-9)
    d["weight"] = weight
    d["weighted_score"] = d["review_score"] * weight

    agg = d.groupby("product_id", observed=True).agg(
        n_reviews=("review_score", "size"),
        raw_review_score=("review_score", "mean"),
        weight_sum=("weight", "sum"),
        weighted_sum=("weighted_score", "sum"),
        delivery_topic=("delivery_topic", "mean"),
    )
    agg["product_quality_signal"] = agg["weighted_sum"] / agg["weight_sum"].replace(
        0, np.nan
    )
    agg = agg[agg["n_reviews"] >= min_reviews]
    return agg.reset_index()[
        [
            "product_id",
            "n_reviews",
            "raw_review_score",
            "product_quality_signal",
            "delivery_topic",
        ]
    ]
