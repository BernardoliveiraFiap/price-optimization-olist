"""Runnable smoke test for the review-text model.

    python -m pricing.demand.demo_reviews

The Olist review corpus is not redistributable, so this module ships a small
hand-written sample of Portuguese marketplace reviews instead. It is enough to
exercise the whole path -- encode, weakly label, train the heads, score, and
aggregate to a per-product quality signal -- without any download beyond the
sentence encoder itself.

What it is meant to show, and what it is not: the sample is written by hand,
so the accuracy printed here says the network learned the keyword rule and
extended it, not that the rule is correct. The load-bearing cases are the ones
marked ``keyword-free`` below: reviews that hit **neither** lexicon, are
therefore absent from training, and still have to be scored correctly. Those
are the reviews a pure keyword rule cannot touch, and they are the reason a
language model earns its place here rather than a regex.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .review_nlp import (
    encode,
    fit_heads,
    product_quality_signal,
    score_reviews,
    weak_labels,
)

# (order_id, product_id, stars, comment)
SAMPLE_REVIEWS: list[tuple[str, str, int, str]] = [
    # --- p1: good product, terrible logistics -------------------------------
    ("o01", "p1", 1, "a entrega atrasou mais de duas semanas, um absurdo"),
    ("o02", "p1", 2, "produto muito bom mas os correios demoraram demais"),
    ("o03", "p1", 5, "excelente produto, material de primeira qualidade"),
    ("o04", "p1", 1, "prazo estourado, transportadora nao deu satisfacao"),
    ("o05", "p1", 5, "recomendo, funciona perfeitamente e o acabamento e otimo"),
    ("o06", "p1", 2, "ainda nao recebi meu pedido"),                # keyword-free
    ("o07", "p1", 5, "veio bem embalado e o tecido e macio"),       # keyword-free
    ("o08", "p1", 1, "comprei ha um mes e nada ate agora"),         # keyword-free
    ("o09", "p1", 4, "otimo produto, so demorou pra chegar"),
    ("o10", "p1", 5, "qualidade excelente, superou a expectativa"),
    ("o11", "p1", 2, "rastreamento parado ha dias, sem previsao"),
    ("o12", "p1", 5, "resistente e bonito, vale o preco"),
    # --- p2: fast delivery, bad product -------------------------------------
    ("o13", "p2", 5, "chegou antes do prazo, entrega rapida"),
    ("o14", "p2", 1, "produto veio quebrado, material fragil"),
    ("o15", "p2", 2, "defeito de fabrica, nao funciona direito"),
    ("o16", "p2", 5, "envio muito rapido, postado no mesmo dia"),
    ("o17", "p2", 1, "veio errado, tamanho e cor diferentes do anuncio"),
    ("o18", "p2", 2, "parece falsificado, nao e original"),
    ("o19", "p2", 5, "recebi em tres dias"),                        # keyword-free
    ("o20", "p2", 1, "desmontou na primeira semana de uso"),        # keyword-free
    ("o21", "p2", 2, "o plastico e muito fino, sensacao de barato"),  # keyword-free
    ("o22", "p2", 5, "entrega impecavel, chegou antes do prazo"),
    ("o23", "p2", 1, "quebrou ao tirar da caixa"),
    ("o24", "p2", 2, "durabilidade pessima, ja estragou"),
    # --- p3: good on both ---------------------------------------------------
    ("o25", "p3", 5, "produto otimo e a entrega foi no prazo"),
    ("o26", "p3", 5, "qualidade muito boa, recomendo a todos"),
    ("o27", "p3", 5, "chegou rapido e bem embalado"),
    ("o28", "p3", 4, "bom acabamento, material resistente"),
    ("o29", "p3", 5, "atendeu tudo que eu esperava"),               # keyword-free
    ("o30", "p3", 5, "correios entregaram antes do prazo previsto"),
    ("o31", "p3", 4, "vale cada centavo"),                          # keyword-free
    ("o32", "p3", 5, "produto de qualidade, comprarei novamente"),
    # --- p4: bad on both ----------------------------------------------------
    ("o33", "p4", 1, "nao chegou e o produto pelas fotos ja parecia ruim"),
    ("o34", "p4", 1, "atrasou muito e veio danificado"),
    ("o35", "p4", 2, "frete caro e material de pessima qualidade"),
    ("o36", "p4", 1, "transportadora sumiu com o pedido"),
    ("o37", "p4", 2, "acabamento grosseiro, defeito na costura"),
    ("o38", "p4", 1, "demorou um mes e ainda veio quebrado"),
    ("o39", "p4", 2, "arrependido da compra"),                      # keyword-free
    ("o40", "p4", 1, "sumiu no caminho, ninguem resolve"),          # keyword-free
]

# Reviews that hit neither lexicon, with the topic a person would assign.
KEYWORD_FREE_EXPECTATION = {
    "o06": "delivery",
    "o07": "product",
    "o08": "delivery",
    "o19": "delivery",
    "o20": "product",
    "o21": "product",
    "o29": "product",
    "o31": "product",
    "o39": "product",
    "o40": "delivery",
}


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(
        SAMPLE_REVIEWS, columns=["order_id", "product_id", "review_score", "comment"]
    )
    reviews = df[["order_id", "review_score", "comment"]]
    mapping = df[["order_id", "product_id"]]
    return reviews, mapping


def run(epochs: int = 400) -> dict:
    reviews, mapping = build_frames()
    labels = weak_labels(reviews["comment"])

    print(f"\n{len(reviews)} reviews | "
          f"{int(labels['labelled'].sum())} weakly labelled "
          f"({labels['labelled'].mean():.0%}) | "
          f"{int((~labels['labelled'].astype(bool)).sum())} left to the network\n")

    print("Encoding with a frozen sentence transformer...")
    embeddings = encode(reviews["comment"], batch_size=64)
    print(f"  embeddings: {embeddings.shape}\n")

    model, metrics = fit_heads(embeddings, labels, epochs=epochs)
    print("Head training")
    for k, v in metrics.items():
        print(f"  {k:<24} {v:.3f}" if isinstance(v, float) else f"  {k:<24} {v}")

    scores = score_reviews(model, embeddings)
    scored = reviews.join(scores)

    # The load-bearing check: reviews the lexicons could not label at all.
    held_out = scored[scored["order_id"].isin(KEYWORD_FREE_EXPECTATION)].copy()
    held_out["expected"] = held_out["order_id"].map(KEYWORD_FREE_EXPECTATION)
    held_out["predicted"] = [
        "delivery" if d > p else "product"
        for d, p in zip(
            held_out["delivery_topic"], held_out["product_topic"], strict=True
        )
    ]
    held_out["ok"] = held_out["expected"] == held_out["predicted"]

    print("\nKeyword-free reviews (never seen in training, no lexicon match)")
    print(
        held_out[["comment", "expected", "predicted", "delivery_topic", "product_topic"]]
        .to_string(index=False, float_format=lambda v: f"{v: .2f}", max_colwidth=48)
    )
    accuracy = float(held_out["ok"].mean())
    print(f"\n  correct: {int(held_out['ok'].sum())}/{len(held_out)} ({accuracy:.0%})")

    signal = product_quality_signal(reviews, scores, mapping, min_reviews=3)
    signal["delta_vs_stars"] = (
        signal["product_quality_signal"] - signal["raw_review_score"]
    )
    print("\nPer-product signal: star average vs quality signal")
    print(signal.to_string(index=False, float_format=lambda v: f"{v: .2f}"))
    print(
        "\nA product whose bad ratings are mostly about the courier scores higher\n"
        "than its star average; one whose complaints are about the item itself\n"
        "does not. Only the second kind should move willingness to pay.\n"
    )
    return {"metrics": metrics, "keyword_free_accuracy": accuracy, "signal": signal}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    run(epochs=args.epochs)


if __name__ == "__main__":  # pragma: no cover
    main()
