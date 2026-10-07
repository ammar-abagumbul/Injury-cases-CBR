# Experiment 7 — Feature-query CBR retrieval with PSLA-aware evaluation

## 1. Objective

A judge describes a new personal-injury case with a small set of features
(injuries, losses, demographics, severity, treatment intensity).  The system
retrieves the most similar decided cases from the extracted corpus using
**case-based reasoning (CBR)**, so that the judge can benchmark a PSLA award
against genuinely comparable precedents.

The distinguishing requirement versus Experiment 4 is that the retrieved cases
should also have **PSLA compensations that are not far apart** from the query's
(and from one another).  Experiment 4 ranked a handful of hand-picked
candidates and scored the ranking with Spearman ρ against a human order.
Experiment 7 scales this to the whole corpus and keeps CBR as the retrieval
method.

Retrieval itself is **clinical-first**: injuries and losses dominate, and PSLA
closeness is tracked as a separate compensation signal rather than optimised
during retrieval (a case's severity is a band of its award, so scoring on it
would encode the target).  A judge who already has an expected range can still
restrict the candidate pool with the PSLA-band filter.

---

## 2. Data

| Item | Value |
|---|---|
| Case JSONs | `output/experiments/corpus_extraction/cases/*.json` — **1,480** |
| Cases with a positive PSLA | **1,336** |
| PSLA span (nominal) | HK$2,000 – HK$2,500,000, 1969–2020 |
| Inflation source | `data/price_index.csv` (yearly index, 1947–2020) |

### PSLA normalisation

Awards span five decades, so nominal HKD is not comparable across cases.  Every
award is deflated to a common base year (default 2020):

```
real(amount, year) = amount * index[base_year] / index[year]
```

The judgment year comes from `metadata.judgment_date` (falling back to the
citation year).  All closeness and band metrics operate on `psla_real`.

---

## 3. Features

The judge query and the corpus cases share one flat feature vector, split into a
**clinical group** (the backbone of the score) and an **auxiliary group** (which
only adjusts within a clinical neighbourhood).  The core five mirror Experiment
4; the auxiliary attributes were added after measuring which features correlate
with PSLA closeness.

| Feature | Kind | Coverage | Group | Notes |
|---|---|---|---|---|
| `injuries` | multiple symbol | 1,363 | clinical | ICD-11 taxonomy + symmetric max-overlap |
| `losses` | multiple symbol | 1,371 | clinical | exact match + Jaccard over category sets |
| `age_at_accident` | integer (poly²) | 885 | auxiliary | |
| `age_at_trial` | integer (poly²) | 746 | auxiliary | |
| `operations_count` | integer (poly²) | 755 | auxiliary | treatment intensity |
| `hospitalisation_days` | integer (poly²) | 570 | auxiliary | treatment intensity |
| `n_injuries` | integer (poly²) | 1,480 | auxiliary | injury count |
| `n_losses` | integer (poly²) | 1,480 | auxiliary | loss count |
| `gender` | symbol (exact) | 968 | *excluded* | near-constant, compresses the ranking |
| `overall_category` | ordered symbol | 606 | *excluded* | award band; target-derived (see §4) |

**Why these groups?**  `overall_category` (severity) is, for a decided case, the
band of its year-adjusted award (see `notes.md` §8): a coarse, target-derived
encoding of the compensation.  It is therefore **not** a clinical feature and is
kept off the retrieval score; it is reconstructed from the award for diagnostics
and used by the optional PSLA-band filter.  A direct analysis of the corpus
showed severity and treatment intensity track PSLA best, but that is exactly why
tuning against PSLA-band relevance crowded out the injuries and losses — the
clinical signal — so the clinical group is now the dominant term by construction.

### Missing values

The corpus is sparse.  Within the **clinical group** a missing value is
*penalised*: the attribute keeps its weight in the denominator, so a case that
cannot be compared clinically is pushed down rather than let off.  The
**auxiliary group** uses available-case aggregation (missing attributes dropped
and weights renormalised).  Set `missing_policy: "special"` for the raw myCBR
amalgamation (a reference path, not the default).

---

## 4. CBR retrieval

A single myCBR concept (`PSLACase`) is built over the corpus:

1. Attribute descriptions and local similarity functions are created (see
   table above) — the ICD-11 taxonomy similarity and the max-overlap / Jaccard
   multiple configs are reused from Experiment 4.
2. `psla_amount` is added as a **solution** attribute and deactivated, so it
   never contributes to similarity.
3. The global similarity is the **clinical-first composite**.  The clinical
   group is a weighted mean of its local similarities with missing values
   *penalised*; the auxiliary group is an available-case weighted mean.  They are
   combined as `clinical_alpha * clinical + (1 - clinical_alpha) * auxiliary`
   (default `clinical_alpha = 0.7`) — auxiliary attributes only adjust within
   the clinical neighbourhood.
4. A **loose clinical gate** drops candidates whose graded injury similarity is
   below `min_injury_similarity` (default `0.1`) before ranking, when the query
   carries injuries.  The floor is deliberately low: it stops treatment/count
   agreement buying in a clinically unrelated case without demanding an exact
   ICD match.
5. Severity is reconstructed from the award for candidates (`severity_from_award`)
   and is **not** a similarity attribute: a decided case's category is the band
   of its award, so including it would encode the target.  A query uses the
   judge's own severity label.
6. With `missing_policy="special"` the raw `SequentialRetrieval` amalgamation is
   used instead (a leak-free reference path; a regression test asserts the
   linear available-case sum agrees with it when every attribute is present).
7. For large-scale evaluation, per-attribute similarities are cached as a
   `(queries, candidates, attributes)` tensor plus a presence mask, so any
   weight vector and `clinical_alpha` can be scored without recomputing
   similarities.

### PSLA-band-constrained retrieval

Known corpus PSLA is used to organise the corpus into compensation bands
(multiplicative bands, `features.psla_band`).  A judge query may supply
`psla_band: [min, max]` (real HKD) or `psla_target_real`, in which case the
candidate pool is restricted to that window before CBR ranking.  By default
ad-hoc retrieval only returns cases with a known real PSLA.

---

## 5. Evaluation

Two complementary protocols (plus a clinical signal), all on the 1,336 cases
with a real PSLA.

### 5.1 Leave-one-out PSLA closeness

Each case is used as a feature query against the rest (its own entry excluded).
A retrieved case is **relevant** when its real award is within a factor
`1 + tolerance` of the query's (default `0.5` ⇒ ratio in `[2/3, 1.5]`).
Reported per query and aggregated:

* PSLA absolute error (MAE, median), relative error, median `|ln(c/q)|`;
* within-tolerance rate;
* Spearman correlation between retrieval similarity and PSLA closeness over
  the whole candidate pool;
* injury (ICD) and loss overlap at k — confirming clinically similar cases are
  still found.

### 5.2 PSLA-band relevance (IR)

Treating same-band cases as the relevant set:

* precision@k, recall@k, graded NDCG@k;
* random baselines (expected precision/recall/NDCG of a random ranking).

### 5.3 Clinical relevance (injury/loss)

A candidate is **clinically relevant** when a graded clinical gain — a blend of
the graded ICD injury similarity and the loss-set overlap
(`clinical_injury_weight`, default 0.75) — clears a loose threshold
(`clinical_threshold`, default 0.15).  This is independent of PSLA and
deliberately soft, because the ICD extraction cannot be trusted to assign an
exact code.  Reported alongside the PSLA metrics: `clinical_precision@k`,
`clinical_recall@k`, `clinical_ndcg@k`, `injury_similarity@k`,
`min_injury_similarity@k` and `loss_similarity@k`.

### 5.4 Weight tuning

Weights are tuned by random search (log-uniform in `[0.1, 3.0]`, default 40
trials) to maximise the mean **clinical NDCG@k** on a **train** split (60% of
queries); the best weights are then scored on the held-out **test** split
against the untuned defaults.  The PSLA-band metrics are reported as a
non-regression sanity check rather than the objective.  Set
`tune_objective: "ndcg"` for the legacy PSLA-band objective.  The tuned weights
are persisted to `tuned_weights.json`, which ad-hoc judge queries pick up
automatically.

---

## 6. Judge query interface

A query is a YAML/JSON `FeatureQuery`:

```yaml
query_id: "example-judge-query"
gender: "Male"
age_at_accident: 45
age_at_trial: 50
injury_descriptions:
  - "fracture of the left tibia"
injuries:
  - "NC92.1"
losses:
  - "Loss of mobility"
overall_category: "Serious injury"
operations_count: 2
# psla_band: [150000, 400000]
k: 10
```

* `injuries` are ICD-11 codes; `injury_descriptions` are free text resolved
  against the ICD-11 descriptions by token overlap (a cheap, dependency-free
  bridge — full LLM ICD classification remains the higher-quality path).
* `losses` must be canonical loss-category names.
* `psla_band` / `psla_target_real` enable band-constrained retrieval.

---

## 7. Files

```
src/hklii_psla/experiments/experiment7/
  __init__.py        # public API
  price_index.py     # HK inflation deflation
  features.py        # CorpusCase + loading + PSLA helpers
  cbr_model.py       # myCBR model, available-case aggregation, similarity cache
  query.py           # FeatureQuery + free-text injury resolution
  retrieval.py       # RetrievalResult / CBRFeatureRetriever (band filtering)
  evaluation.py      # PSLA-closeness + IR metrics, Spearman, weight scoring
  exp7.py            # FeatureQueryRetrievalExperiment (LOO + ad-hoc modes)
  config.yaml        # benchmark config
  query_example.yaml # example judge query
tests/test_experiment7.py
```

Outputs (`output/experiments/feature_query_retrieval/`):

| File | Contents |
|---|---|
| `metrics.json` | corpus summary, default metrics, tuning history/results |
| `per_query.csv` | per-query metrics for the default weights |
| `tuned_weights.json` | best weights + held-out NDCG |

---

## 8. How to run

```bash
# Leave-one-out benchmark over the whole corpus (tunes weights)
.venv/bin/python -c "from pathlib import Path; \
from hklii_psla.experiments.experiment7.exp7 import FeatureQueryRetrievalExperiment; \
e=FeatureQueryRetrievalExperiment(Path('src/hklii_psla/experiments/experiment7/config.yaml')); \
e.run(); e.save_results()"

# Ad-hoc judge query
.venv/bin/python -c "from pathlib import Path; \
from hklii_psla.experiments.experiment7.exp7 import FeatureQueryRetrievalExperiment; \
e=FeatureQueryRetrievalExperiment(Path('src/hklii_psla/experiments/experiment7/query_example.yaml'))"
```

Or via the CLI (once the package is importable):

```bash
uv run psla run-experiment feature-query-retrieval \
    src/hklii_psla/experiments/experiment7/config.yaml
```

---

## 9. Results

Full re-run over the 1,336 PSLA cases (`output/experiments/feature_query_retrieval/`,
`k = 10`, tolerance ±50%, 40 tuning trials, 60/40 train/test split,
`clinical_alpha = 0.7`, injury gate `0.1`, objective clinical NDCG).

**Default (equal) weights — all 1,336 queries**

| Metric | Value |
|---|---|
| Mean clinical NDCG@k | 0.915 |
| Mean clinical precision@k | 0.976 |
| Mean injury similarity@k | 0.517 |
| Mean min injury similarity@k | 0.307 |
| Mean loss similarity@k | 0.642 |
| Mean PSLA-band NDCG@k | 0.207 |
| Random PSLA-band NDCG@k | 0.148 |
| Mean precision@k | 0.378 |
| Within-tolerance rate | 0.378 |
| Median PSLA MAE | HK$188,952 |
| Median log-ratio error | 0.548 |
| Median Spearman(similarity, PSLA) | 0.020 |
| Mean injury overlap@k | 0.533 |
| Mean loss overlap@k | 0.775 |

The clinical metrics are high because the relevance label is derived from the
same injury/loss similarities the ranker uses — it mainly confirms that the
ranker surfaces the best clinical matches.  PSLA-band NDCG is the closer thing
to an external signal and stays near the random baseline.

**Weight tuning (held-out test split, 60/40)**

| Metric | Default | Tuned |
|---|---|---|
| Mean clinical NDCG@k | 0.815 | **0.875** |
| Mean injury similarity@k | 0.505 | **0.624** |
| Mean min injury similarity@k | 0.291 | **0.486** |
| Mean PSLA-band NDCG@k | 0.215 | 0.207 |

Tuning lifts held-out clinical NDCG by ~7% relative.  The selected weights put
`injuries` at 1.17 and `losses` at 0.38 inside the clinical group (which carries
0.7 of the score); within the auxiliary group `n_losses` (2.34) and `n_injuries`
(1.66) act as tie-breakers.  PSLA-band NDCG moves from 0.215 to 0.207 — a
negligible change that is far smaller than the old PSLA-tuned spread and shows
the clinical and compensation objectives are largely orthogonal.

**Ad-hoc judge query.**  The worked example (`notes.md` §9 and
`demo/DEMO.md`) is now clinical-first: its top-8 minimum injury similarity rises
from 0.000 to 0.542, but only 4/8 awards fall within ±50% (was 6/8) — the
visible tension between clinical resemblance and quantum closeness.

### Takeaways

1. With the clinical group dominant and a loose injury gate, retrieval is
   clinically coherent: no result can enter on treatment/counts alone.
2. Clinical NDCG is near-saturated because the label is feature-derived; it
   shows the ranker works as specified, not that the cases are legally correct.
3. PSLA-band NDCG is essentially unchanged (~0.21 vs a ~0.15 random baseline),
   so clinical retrieval alone does not guarantee compensation closeness; a
   judge with an expected range should use the PSLA-band filter.
4. Severity is a target-derived award band and is kept off the clinical score;
   it is reconstructed from the award for diagnostics and the band filter.
5. Tuned weights are persisted (`tuned_weights.json`) and reused automatically
   by ad-hoc judge queries.

