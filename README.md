# Price Optimization on Brazilian E-Commerce Data

Estimating price elasticity from observational marketplace data, and turning
that estimate into a constrained price list with a margin number attached.

## Overview

Most pricing analyses stop at a coefficient. This one runs the whole chain and
measures what each link is worth:

1. **Estimate** the price elasticity of demand from a product-week panel,
   using three estimators that make progressively weaker assumptions about how
   prices came to be what they are.
2. **Validate** those estimators against a synthetic market whose true
   elasticity is known, because on real data the answer is unobservable and
   "my model says -1.4" cannot be checked.
3. **Optimize** a portfolio price list with a mixed-integer program, under the
   guardrails a pricing team actually operates with.
4. **Price the mistakes** — how much margin is lost by trusting the biased
   estimate, and how fast the conclusion decays if the demand curve is wrong.

The headline finding is about step 4. Under endogenous pricing, naive OLS says
demand is 24% less price-sensitive than it is. Acting on that belief tells you
to **raise the average price 5%**; acting on the credible estimate tells you to
**hold the average and reallocate**. Judged against the same demand curve, the
naive price list gives up 11% of the achievable margin gain and sheds 6.5% of
volume.

## Results

### Do the estimators recover a known truth?

20 independent synthetic markets, true mean elasticity -2.103:

| Estimator | Mean estimate | Relative bias | RMSE | 95% CI coverage |
|---|---|---|---|---|
| Pooled OLS | -1.590 | +24.5% | 0.518 | 0% |
| Two-way fixed effects | -1.769 | +16.0% | 0.334 | 0% |
| 2SLS (cost instrument) | -2.103 | **+0.01%** | 0.009 | **100%** |

Positive bias means the estimate sits closer to zero than the truth: demand
looks less price-sensitive than it really is, which is exactly the direction
that makes a price increase look safe when it is not.

Turning the endogeneity off is the control experiment. Fixed effects then
become unbiased (+0.02%) while pooled OLS stays wrong (+19.7%), because latent
product quality still moves price and demand together. Each estimator fails in
precisely the case its assumptions rule out.

### What the price list is worth

297 products, elasticities estimated per category:

| | Average price | Margin | Revenue | Volume |
|---|---|---|---|---|
| Optimized list | -0.58% | **+6.89%** | +4.90% | +6.18% |
| Same optimizer, naive OLS elasticity | +5.00% | +6.12% | -1.24% | -6.54% |

### What the guardrails cost

| Policy | Margin uplift | Volume | Average price | Margin given up |
|---|---|---|---|---|
| None (per-product optimum) | +6.894% | +6.18% | -0.58% | — |
| Base policy | +6.894% | +6.18% | -0.58% | 0.000 pp |
| Grow volume 10% | +6.860% | +10.00% | -1.80% | 0.035 pp |
| Grow volume 10% + cut average price 3% | +6.816% | +10.49% | -3.00% | 0.078 pp |

The first two rows do not bind: the margin-optimal reallocation already
respects the standard guardrails. They only cost money once the business also
demands volume growth, which is the trade a pricing owner negotiates over.

### Is the conclusion robust?

The same price list, re-scored against elasticities scaled from 0.6x to 1.4x:

| Elasticity scale | 0.6x | 0.8x | 1.0x | 1.2x | 1.4x |
|---|---|---|---|---|---|
| Margin uplift | +1.38% | +3.72% | +6.89% | +10.96% | +16.00% |

The sign survives a 40% error in the demand curve; the magnitude does not.
That is the honest reading, and it is why the deliverable is a direction and a
range rather than a single number.

## Stack

- Python 3.12
- pandas, numpy — panel construction
- statsmodels — OLS, fixed effects, clustered standard errors
- PuLP + CBC — mixed-integer price optimization
- PyTorch + sentence-transformers — review-text model
- pytest, ruff, GitHub Actions

## Project structure

```
src/pricing/
  config.py                  paths, guardrail defaults, panel parameters
  synth.py                   synthetic market with a KNOWN elasticity
  data.py                    Olist CSVs -> product-week panel + instrument
  pipeline.py                end-to-end run (estimate -> optimize -> score)
  elasticity/
    loglog_fe.py             pooled OLS, two-way FE, 2SLS with clustered SEs
  demand/
    review_nlp.py            review text -> product quality vs delivery signal
  optimize/
    milp.py                  price-grid MILP with portfolio guardrails
  evaluate/
    validation.py            estimator recovery against the known truth
    impact.py                uplift, guardrail frontier, sensitivity
tests/                       15 tests: estimators, optimizer, review model
data/raw/                    Olist CSVs (not committed)
reports/                     generated CSV outputs
```

## Setup

Python 3.12 is required (PyTorch has no wheels for 3.14 yet).

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

`torch` is pulled from the default index by the commands above. On Linux that
means the CUDA build; for a CPU-only machine install it explicitly first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Encoding real review text also needs `sentence-transformers`, which is kept out
of the pinned requirements because nothing else in the project depends on it:

```bash
pip install sentence-transformers
```

The real-data path additionally needs the
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
extracted into `data/raw/`. Everything else runs without any download.

## Running

Validate the estimators against a known truth (no data needed):

```bash
python -m pricing.evaluate.validation --reps 20
```

Full pipeline on the synthetic market:

```bash
python -m pricing.pipeline --source synth
```

Build the panel from the Olist CSVs, then run on real data:

```bash
python -m pricing.data --level product
```

```bash
python -m pricing.pipeline --source olist
```

Tests and lint:

```bash
python -m pytest
```

```bash
python -m ruff check src tests
```

## Method

### Identification

Demand is modelled in logs, so the price coefficient is the elasticity
directly:

```
log q_it = alpha_i + gamma_t + beta * log p_it + controls + e_it
```

The problem is that `p_it` is not assigned at random. Two distinct confounders
show up in marketplace data, and they need different treatments:

- **Latent product quality** raises willingness to pay *and* the price a seller
  can charge. It is constant within a product, so a product fixed effect
  removes it. Pooled OLS cannot, and that is most of its bias.
- **Time-varying demand shocks** — a product trends, the seller raises the
  price that week. This survives fixed effects, because it varies *within* a
  product over time. Only an instrument removes it.

The instrument is a cost shifter: something that moves price without moving
demand except through price. In the synthetic market it is generated
explicitly. On Olist, where no cost is recorded, the stand-in is a
Hausman-style instrument — the average price of the same category, same week,
in *other states*, excluding the focal unit's own transactions. The argument is
that sellers across regions share input and freight cost shocks while demand
shocks are regional. That exclusion restriction is an assumption and is stated
as one: it fails under national demand shocks, which is why the IV is always
run inside the week fixed effects that absorb them.

Standard errors are clustered by product throughout. The 2SLS covariance is
the clustered sandwich with residuals formed from the *actual* price, not the
fitted one.

### Optimization

Each product gets a grid of candidate prices around today's price, and a
binary variable selects one. Because demand, revenue and margin are
precomputed at every grid point, the objective and constraints are linear in
those binaries — the non-linearity of the demand curve lives in the
coefficients, not the model:

```
maximise    sum_ik  margin[i,k] * x[i,k]
subject to  sum_k   x[i,k] = 1                        for every product i
            sum_ik  revenue[i,k] * x[i,k] >= floor
            sum_ik  w[i] * delta[k] * x[i,k] <= cap
            sum_ik  units[i,k] * x[i,k] >= volume floor
```

The average-price guardrail is weighted by each product's share of revenue, so
a low-volume item cannot buy headroom for a large one. Without the coupling
constraints the problem separates and the optimum has a closed form; the
guardrails are what make it an optimization problem rather than arithmetic.

### Review text model

The panel carries a `review_score`, and it is a poor control. A 1-to-5 star
rating conflates two things that should not enter a pricing model the same way:

```
"produto excelente, mas a entrega atrasou duas semanas"   -> 2 stars
"chegou super rapido, mas veio quebrado"                  -> 2 stars
```

Both are 2 stars, but only the first describes a product people would still pay
for. Willingness to pay follows perceived product quality; a late courier
depresses the rating without depressing the product's value.

Separating them requires reading the sentence, which is the one place in this
project where a neural model is the right tool rather than a decoration. A
frozen multilingual sentence encoder embeds each Portuguese comment; two
keyword lexicons weakly label the unambiguous reviews; and a small two-head MLP
learns to score any review on both axes, generalizing past the keywords. Star
ratings are then re-weighted toward the product axis to produce a per-product
quality signal.

## Validation

The synthetic generator is not a demo dataset — it is the test harness. It
writes down the true elasticity, the latent quality, the demand shock and the
cost shifter explicitly, then generates prices from a seller rule that reacts
to demand. Everything downstream is source-agnostic, so the real and synthetic
runs exercise identical code.

The test suite locks the properties that matter:

- fixed effects recover the truth when pricing is exogenous
- pooled OLS is biased toward zero when latent quality is present
- 2SLS beats fixed effects under endogenous pricing, with a first-stage F above 10
- the MILP assigns exactly one price per product
- every guardrail holds in the returned solution
- constraints can only ever cost margin, never add it
- an unchanged price list scores exactly zero uplift

CI runs lint, the tests, and the estimator-recovery experiment on every push,
so the table in this README is re-proved rather than pasted.

## Design notes

- **Sellers price near their optimum, not far below it.** An early version
  assumed a flat gross margin, which puts every product below its
  profit-maximizing price and reduces the optimizer to "raise everything until
  a guardrail binds" — the elasticity estimate then barely matters. Anchoring
  unit cost to the constant-elasticity markup rule, plus a mispricing
  perturbation, produces the realistic problem: some prices should go up,
  others down, and the money is in the reallocation.
- **Fixed effects are not automatically an improvement.** They discard
  between-product variation and keep within-product variation. If the
  within-product part is the contaminated one, FE can be *worse* than pooled
  OLS. It helps here because week-to-week price moves are mostly cost-driven,
  which is an assumption about the market, not a mathematical guarantee.
- **A positive estimated elasticity is a failed identification**, not a Giffen
  good. Those categories fall back to the portfolio mean instead of telling the
  optimizer that raising prices sells more.
- **The two-way within transformation is iterated**, not applied in one pass.
  The closed form is exact only for balanced panels, and real panels are not.

## Limitations

- **No counterfactual exists.** Nobody re-ran 2017 at different prices, so the
  uplift is what the estimated demand curve implies, not a measured outcome.
  The sensitivity table is the honest bound on that claim.
- **Olist records no costs.** The real-data run needs a margin assumption, and
  the answer moves with it. The synthetic run avoids this by generating costs.
- **Cross-price effects are ignored.** Products are treated as independent, so
  cannibalization between substitutes is not modelled.
- **Constant elasticity is a strong functional form.** It cannot represent
  reference-price effects, thresholds, or asymmetry between increases and cuts.
- **The Hausman instrument is defensible, not airtight.** It buys consistency
  in exchange for an exclusion restriction that cannot be tested here.

## About

An academic study project on causal inference and optimization applied to
retail pricing, built on the public Olist Brazilian e-commerce dataset. The
emphasis is on the parts that usually get skipped: validating an estimator
against a known truth, and measuring what a wrong estimate costs once a
decision depends on it.
