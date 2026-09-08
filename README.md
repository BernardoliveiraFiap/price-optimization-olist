# Price Optimization on Brazilian E-Commerce Data

Estimating price elasticity from observational marketplace data, and turning
that estimate into a constrained price list with a margin number attached.

### **[Read the project walkthrough](https://bernardoliveirafiap.github.io/price-optimization-olist/)**

A single page covering the problem, the architecture, what each module does,
the real output of every command, and the charts behind the numbers below.

### **[Run it yourself in Colab](https://colab.research.google.com/github/BernardoliveiraFiap/price-optimization-olist/blob/main/notebooks/roteiro-por-niveis.ipynb)** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BernardoliveiraFiap/price-optimization-olist/blob/main/notebooks/roteiro-por-niveis.ipynb)

A guided notebook in ten levels that executes this repository end to end on a
free Colab runtime, written in Portuguese. Each level states what its command
does, asks the questions the output should raise, and answers them against the
real numbers -- including the ones that reject this project's own preferred
estimator. Levels 1 and 2 set up the environment and the dataset; the rest map
one-to-one onto the `Makefile` targets.

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

### On synthetic data: do the estimators recover a known truth?

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

### Does the review model separate what the star rating conflates?

Run on a hand-written sample of 40 Portuguese marketplace reviews
(`python -m pricing.demand.demo_reviews`):

| Product | Star average | Quality signal | Shift |
|---|---|---|---|
| Good product, terrible logistics | 3.17 | **4.63** | +1.46 |
| Fast delivery, bad product | 2.67 | **1.57** | -1.10 |
| Good on both | 4.75 | 4.57 | -0.18 |
| Bad on both | 1.38 | 1.50 | +0.13 |

A product whose bad ratings are mostly about the courier scores more than a
star higher than its raw average; one whose complaints are about the item does
not move. Only the second kind should shift willingness to pay.

The keyword lexicons label 22 of the 40 reviews. The number that matters is
what happens on the 10 that match **neither** lexicon, are therefore absent
from training, and could never be handled by a rule: the heads get **9 of 10**
right, including "ainda nao recebi meu pedido" (delivery, 0.98) and "o plastico
e muito fino, sensacao de barato" (product, 1.00). The single miss is
"desmontou na primeira semana de uso", read as delivery. That gap between the
lexicon and the embedding is the argument for using a language model here
instead of a regex.

### On real Olist data: the pipeline refuses most of the portfolio

This is where the synthetic result and the real one part company, and the gap
is the point of the project.

The panel keeps 844 of 32,216 products -- the ones with at least 12 weeks of
sales -- for 16,278 product-weeks over 88 weeks. Everything else is the long
tail, sold a handful of times, carrying no usable price variation.

**The instrument does not survive contact with the data.**

| Estimator | Estimate | Standard error |
|---|---|---|
| Pooled OLS | -0.002 | 0.018 |
| Two-way fixed effects | **-0.756** | 0.092 |
| 2SLS (Hausman instrument) | +58.440 | 147.697 |

The 2SLS number is not imprecise, it is meaningless: the **first-stage F is
0.16** against a threshold of 10. The reason is visible in one statistic --
only **1.1% of the variance in log price is within-product**. A given Olist
listing barely changes price over its life, and the within transformation that
removes latent quality also removes almost everything the instrument had to
work with. The pipeline detects this and falls back to fixed effects rather
than reporting a weak-IV estimate as causal.

Pooled OLS returning -0.002 is its own finding: on this data, ignoring
confounding does not merely bias the elasticity, it erases it.

**Most categories cannot be priced.** Only 3 of 20 clear the gate, which
requires a point estimate past -1.05 *and* an estimate at least twice its own
standard error (`cool_stuff` -1.44, `watches_gifts` -1.22, `perfumery` -1.20).
That second test is against zero, not against -1: the 95% intervals of all
three still cross -1, and requiring the whole interval to clear -1 would reject
every category in the panel. The gate is a declared trade-off, not a claim of
statistical elasticity. The rest are inelastic as measured, where the constant-elasticity
margin has no interior optimum and a "recommendation" would just be the edge of
whatever band it was given. Two categories show positive point estimates, both
with |t| < 1 -- noise, not Giffen goods.

That leaves 75 products, **23% of panel revenue**, as the priceable portfolio:

| | Margin | Revenue | Volume | Average price |
|---|---|---|---|---|
| With guardrails | **+7.56%** | -1.04% | -6.01% | +5.00% (cap binding) |
| Without guardrails | +33.34% | -6.66% | -28.03% | +30.00% (band edge) |

**The guardrails cost 25.8 pp here, against 0.0 pp on synthetic data.** That
inversion is the argument for the solver: when demand is close to inelastic,
the unconstrained optimum runs to the edge of the price band and the policy
constraints stop being paperwork -- they become the entire decision.

| Policy | Margin uplift | Volume | Average price |
|---|---|---|---|
| None (per-product optimum) | +33.34% | -28.03% | +30.00% |
| Base policy | +7.56% | -6.01% | +5.00% |
| Grow volume 10% | +3.64% | +10.00% | +5.00% |
| Grow volume 10% + cut average price 3% | **-5.60%** | +10.00% | -3.00% |

The last row is worth stating plainly: that mandate does not cost margin
growth, it destroys margin.

**And the cost assumption drives the size of the prize.** Olist records no
costs, so a gross margin has to be assumed, and the optimum is proportional to
it:

| Assumed gross margin | 20% | 30% | 35% | 45% | 60% |
|---|---|---|---|---|---|
| Margin uplift | +17.52% | +9.76% | +7.56% | +4.62% | +2.15% |

The recommended average price change is +5.00% in every one of those scenarios,
pinned by the guardrail. So the *direction* is robust to the assumption and the
*magnitude* is almost entirely a function of it. Reporting the 7.56% without
that table would be reporting an assumption as a result.

### Is the conclusion robust?

The synthetic price list, re-scored against elasticities scaled from 0.6x to
1.4x:

| Elasticity scale | 0.6x | 0.8x | 1.0x | 1.2x | 1.4x |
|---|---|---|---|---|---|
| Margin uplift | +1.38% | +3.72% | +6.89% | +10.96% | +16.00% |

The sign survives a 40% error in the demand curve; the magnitude does not.
That is the honest reading, and it is why the deliverable is a direction and a
range rather than a single number.

### Can a machine-learning model do this instead?

This is the question a stakeholder always asks, so the repository answers it
with a fitted model rather than an opinion. A gradient-boosted tree is trained
to forecast weekly quantity, validated on **expanding windows over calendar
weeks** — never a random `KFold`, which on panel data would train on week 40 to
predict week 12 of the same product.

| Model | RMSE | MAE | Implied elasticity |
|---|---|---|---|
| Persistence (last week) | 0.7155 | 0.4880 | — |
| Ridge | 0.5739 | 0.4627 | — |
| **Gradient boosting** | **0.4489** | **0.3360** | **-0.281** |
| Gradient boosting, no history | 0.4670 | 0.3477 | **-0.050** |
| *Two-way fixed effects (causal)* | — | — | *-0.756* |

The tree cuts forecast error by **37%** against the naive baseline. Then it is
asked the pricing question — hold everything fixed, move `log_price`, measure
the response — and it answers **-0.281**, less than half the credible
elasticity. Strip the lag features and it collapses to **-0.050**, which is
naive OLS's `-0.002` all over again, reached by a completely different model
class.

**The best forecaster is the worst pricing model, and the gap is not small.**
Acting on -0.281 would price as if demand barely reacts.

There is a second reading worth having. The variant *with* history is the less
biased of the two, which looks backwards until you notice what lagged demand
is: a product's own recent sales encode its latent quality, so the lags absorb
unit-level heterogeneity the way a fixed effect does — badly, but not by
nothing. The features that make the model a good forecaster are the same ones
that partially clean its elasticity, and neither effect was designed.

## Stack

- Python 3.12
- pandas, numpy — panel construction and the two-way within transformation
- statsmodels — OLS with cluster-robust standard errors
- PuLP + CBC — mixed-integer price optimization
- scikit-learn — gradient-boosted demand forecaster and the time-series folds
- PyTorch — the review-scoring heads
- sentence-transformers — frozen multilingual encoder (optional extra)
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
    ml_forecast.py           GBM demand forecast + the elasticity it implies
    review_nlp.py            review text -> product quality vs delivery signal
    demo_reviews.py          runnable sample: the text path without Kaggle
  optimize/
    milp.py                  price-grid MILP with portfolio guardrails
  evaluate/
    validation.py            estimator recovery against the known truth
    impact.py                uplift, guardrail frontier, sensitivity
tests/                       30 tests: estimators, optimizer, gates,
                             review model, forecaster leakage
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

Forecast comparison and the elasticity the fitted model implies (needs the
panel built above):

```bash
python -m pricing.demand.ml_forecast --folds 4
```

The review-text model on a built-in sample (downloads the encoder on first
run, roughly 470 MB):

```bash
python -m pricing.demand.demo_reviews
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
- the demo's "keyword-free" reviews really do match neither lexicon, so the
  accuracy it reports stays held out
- a weak instrument is refused and the pipeline falls back to fixed effects
- positive and inelastic category estimates never reach the optimiser
- infeasible guardrails raise instead of returning a phantom solution

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
- **Olist cannot identify a causal elasticity.** With 1.1% of price variance
  within-product the instrument is dead on arrival, so the real-data estimate
  rests on fixed effects and the assumption that week-to-week price moves are
  not driven by demand. The synthetic experiment shows exactly how much bias
  survives when that assumption fails.
- **Olist records no costs.** The margin-rate assumption sets the size of the
  uplift, as the sensitivity table shows. Only the direction is robust to it.
- **77% of the panel revenue is not priced at all**, because its estimated
  elasticity does not clear -1.05 with an estimate twice its standard error.
  That is a real answer, not a gap to be filled by lowering the bar. The
  converse is also worth stating: the gate does not certify that the three
  priced categories are elastic beyond doubt, since their intervals still
  cross -1.
- **Cross-price effects are ignored.** Products are treated as independent, so
  cannibalization between substitutes is not modelled.
- **Constant elasticity is a strong functional form.** It cannot represent
  reference-price effects, thresholds, or asymmetry between increases and cuts.

## About

An academic study project on causal inference and optimization applied to
retail pricing, built on the public Olist Brazilian e-commerce dataset.

The emphasis is on the parts that usually get skipped: validating an estimator
against a known truth before trusting it, measuring what a wrong estimate costs
once a decision depends on it, and letting the pipeline decline to answer where
the data does not support one. On this dataset that last part does most of the
work -- the instrument is refused, 17 of 20 categories are refused, and what
survives is a bounded recommendation over a quarter of the portfolio.
