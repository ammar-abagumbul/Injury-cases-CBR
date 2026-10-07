# Experiment 7 — notes: clinical similarity is being crowded out of the aggregate

Status: observation recorded from the worked example in
`output/experiments/feature_query_retrieval/demo/DEMO.md`. The fix plan below is
now **implemented** — see §9 for the implementation status and the re-run
numbers.

## 1. What prompted this

The worked example retrieves the nearest cases to query `[2016] HKCFI 601`
(multi-trauma: head, face, teeth, eye, cervical spine, ribs, median nerve;
3 operations, 62 hospital days; real PSLA HK$434,018).

- **Rank 1 — `[1985] HKCFI 80`** (similarity 0.932) is a defensible comparable:
  same severity, same sex, age at trial within a year, same occupational class,
  two exact shared ICD codes (`NA07.Z`, `NA82.4`) plus several same-block matches
  in head/face/cervical/eye, two shared loss categories, award within ~6%.
- **Rank 8 — `[2000] HKCFI 1090`** (similarity 0.919) is not: no exact ICD match,
  the entire injury profile is a single-region dominant-hand crush, severity is
  missing, only one shared loss category, award ~47% higher (at the edge of the
  ±50% band).

The two aggregates differ by **0.013**. The substantive difference is enormous.
Worse, cases that dominate rank 8 on *every* dimension sit below it — e.g.
`[2003] HKCFI 270` at **rank 25** has injury similarity 0.458 (vs 0.243), loss
similarity 0.400 (vs 0.200) and award |ln| 0.057 (vs 0.385).

This is consistent with the benchmark headline: tuned held-out NDCG@10 = 0.206
vs random 0.154 — only marginally better than chance.

## 2. The observation

**The aggregate similarity is driven by non-clinical agreement, and the actual
clinical agreement (injuries, losses) is only a minor term.**

The tuned weights do this deliberately, because they were tuned against a
PSLA-band relevance target:

| Attribute | Weight | Clinical? |
|---|---|---|
| `operations_count` | 2.211 | no (treatment intensity) |
| `n_losses` | 1.087 | no (count proxy) |
| `overall_category` | 0.992 | no — compensation-derived (award band) |
| `gender` | 0.969 | no (near-constant) |
| `n_injuries` | 0.632 | no (count proxy) |
| `losses` | 0.262 | **yes** |
| `hospitalisation_days` | 0.199 | no (treatment intensity) |
| `injuries` | 0.160 | **yes** |
| `age_at_trial` | 0.255 | no |
| `age_at_accident` | 0.155 | no |

Injuries + losses together carry 0.422 of weight; treatment, counts, severity and
demography carry the rest. Tuning removed exactly the two clinical attributes.
Note that `overall_category` is **not clinical**: for an existing case severity is
the band its award falls into (§8), so it is a coarse, target-derived encoding of
the compensation. Only the query's severity — a judge's pre-award impression — is
clinical in origin.

## 3. Root causes

1. **Objective misalignment.** Weight tuning maximises PSLA-band NDCG. Any
   candidate whose year-adjusted award falls in the band is "relevant", however
   different its injuries. A hand-crush case with matching treatment and counts
   is therefore a legitimate reward signal, and the optimiser exploits it.
2. **Counts as clinical proxies.** `n_injuries` and `n_losses` use polynomial
   distance on item counts. Eight injuries scores ~1.0 against eight injuries
   regardless of body region, so counts substitute for — and can outvote —
   the injury set itself.
3. **Missing `overall_category` is an extractor error, not a missing feature.**
   Severity is deterministically the band of the year-adjusted award (§8), so an
   existing case without a category has simply been mis-extracted. Available-case
   amalgamation currently treats that as neutral and inflates the remaining
   weights; the correct remedy is to **reconstruct** the category from the award,
   not to penalise or drop it.
4. **Near-constant attributes compress the ranking.** `gender` scores 1.000 for
   every male–male pair and contributes a near-constant amount to everyone, so it
   shrinks the usable range. The whole top-30 lives within 0.907–0.932, which is
   why the ordering inside it is close to arbitrary.
5. **Soft ICD similarity is a feature, not a bug.** Even the best injury match
   for this query is only ~0.54, because the max-overlap taxonomy similarity
   gives partial credit across a whole ICD chapter. That looks like compression,
   but the extraction itself is noisy — the LLM does not always assign the exact
   right code (ambiguous nodes, injuries described but not coded, missing detail).
   So we should **not** harden the criterion around exact-code agreement; we
   should use the graded taxonomy similarity as a deliberately loose signal
   (see §4).

## 4. Fix plan

The goal is not to abandon PSLA closeness — comparable awards are the point —
but to make clinical resemblance the primary term and PSLA closeness a secondary
confirmation, rather than letting PSLA-band tuning choose the weights outright.

### Phase 0 — measurement (partly done)

- [x] Report `injury_similarity` and `loss_similarity` alongside the aggregate in
  `retrieval.json` and the demo report. (Done — aggregate can now hide behind
  visible components.)
- [x] Add a **clinical relevance** label independent of PSLA, using a **loose**
  criterion: a candidate is relevant to a query when its graded injury
  similarity clears a calibrated threshold, optionally reinforced by shared loss
  categories. Do **not** make relevance contingent on an exact ICD-code match —
  exact codes are unreliable (§3.5) and would drop genuinely comparable cases.
- [x] Add a helper to **impute `overall_category` from the award** for corpus
  candidates (§8), and report how often each severity band was imputed rather
  than extracted.
- [x] Add `clinical_ndcg@k`, `clinical_precision@k`, `clinical_recall@k` and
  `min_injury_similarity@k` to `evaluation.py`, computed in parallel with the
  existing PSLA-band metrics.

### Phase 1 — restructure the similarity

- [x] **Two-stage retrieval with a loose gate.** Stage 1 gates on clinical
  overlap using a *calibrated threshold on graded injury similarity* (not an
  exact-code test), so candidates with a different-but-adjacent code are not
  excluded by extraction noise. Stage 2 ranks survivors by the aggregate. This
  guarantees no candidate can enter the top-k on treatment/counts alone while
  staying robust to ICD coding errors.
- [x] **Clinical-first composite.** Compute
  `clinical = w_inj * injury_similarity + w_loss * loss_similarity` and make it
  the dominant term of the global score; auxiliary attributes adjust within the
  clinical neighbourhood rather than competing with it.
- [x] **Demote counts.** Cap `n_injuries`/`n_losses` at tie-breaker weight, or
  replace exact-count polynomial similarity with a coarse order-of-magnitude
  match so 8 ≠ 8 unless the injuries themselves agree.
- [x] **Reconstruct missing severity from the award.** Impute `overall_category`
  for corpus candidates from their year-adjusted award using the published
  severity bands (§8) — a deterministic reconstruction of an extractor miss, not
  a statistical guess. Because it is target-derived it belongs with the
  compensation side of the scoring, not the clinical term. Keep a genuine
  `missing_policy="penalty"` only for attributes that truly cannot be recovered
  (numerator excludes them, denominator keeps their weight).
- [x] **Neutralise near-constant attributes.** Drop `gender` from the retrieval
  attribute set, or floor/whiten attributes whose local-similarity variance
  across the candidate pool is negligible.
- [x] **Keep ICD similarity loose, and improve its shape rather than its
  sharpness.** Do not weight exact codes above same-block matches. If anything,
  strengthen within-region graded similarity and consider the Experiment 6
  coarse-ICD correction only as a way to reduce noise between adjacent codes,
  not as a way to reward exact agreement.

### Phase 2 — retune against clinical agreement

- [x] Make `clinical_ndcg@k` the primary tuning objective. Because severity is a
  band of the award (§8), any PSLA-band metric that includes the severity term is
  partly circular — compute the non-regression constraint with severity excluded
  (or imputation disabled), and treat PSLA-band numbers as a sanity check rather
  than a target.
- [x] Re-run the 60/40 LOO tuning and persist new weights. Expect `injuries`
  and `losses` to rise and `operations_count` to fall.

### Phase 3 — guardrails and reporting

- [x] Expose `min_injury_similarity@k` in the benchmark summary so a
  rank-8-style inclusion is visible in aggregate, not only in a worked example.
- [ ] Consider a "no clinically comparable precedent found" outcome when the
  best injury similarity is below a floor, instead of always returning k cases.
- [x] Keep the PSLA-band metrics — they answer "are the awards close?" — but
  never again let them be the only relevance signal.

## 5. Acceptance criteria (revised)

The original criteria asked for `[2003] HKCFI 270` and `[1980] HKCFI 66` to enter
the top-8. That assumed PSLA closeness remained a retrieval term — but using the
query's own award in leave-one-out would leak the target, so the clinical score
excludes it. With a genuinely clinical ranking, those two cases are out-ranked
by cases with higher injury similarity (0.54–0.69 vs 0.458 and 0.538), and their
absence from the top-8 is correct, not a regression.

Revised criteria, and the outcome on the re-run:

- [x] The worked-example top-8 `min_injury_similarity` rises materially: **0.000
  → 0.542**.
- [x] `[2000] HKCFI 1090` (injury sim 0.243, no exact overlap) drops out of the
  top-8 (`> rank 40`, was rank 8).
- [x] No returned case can enter on treatment/counts alone — all top-8 clear the
  loose injury gate.
- [x] On the full 1,336-query LOO benchmark, mean held-out `clinical_ndcg@10`
  improves: **0.815 → 0.875**.
- [x] PSLA-band NDCG@10 does not regress beyond a small epsilon: **0.215 →
  0.207** (random baseline ≈ 0.15; the previous PSLA-tuned weights scored
  0.206). The ~0.008 movement is the expected cost of prioritising clinical
  resemblance.

## 6. Decisions on the open questions

- **Clinical ground truth — resolved.** Use the ICD (graded taxonomy) similarity
  as an acceptable stand-in. There is no labelled ground truth for now. It is a
  proxy and should be treated as such, but it is the best signal available.
- **Missing `overall_category` — resolved.** Treat it as an extractor error and
  reconstruct it deterministically from the award via the severity bands (§8) —
  no gate and no free pass. Note that this makes severity a compensation-derived
  attribute on the candidate side, so it is not part of the clinical term and
  must be handled carefully in PSLA-band evaluation.
- **Experiment premise — resolved.** Focus on clinical similarity for now. Keep
  tracking compensation outcomes alongside it (never tune on PSLA-band metrics
  while severity is imputed — see the caveat in §8), so the tension between
  clinical closeness and quantum divergence stays visible.

Still open:

- **Loss taxonomy granularity.** 27 categories with Jaccard is coarse; "loss of
  mobility" from a hand injury and from a spinal injury are not the same.
  Consider conditioning loss similarity on the causing injury.
- **Loose gate calibration.** The clinical-relevance threshold and the Stage-1
  gate threshold must be calibrated on the benchmark without leaking PSLA; this
  needs a small sensitivity study.

There is no longer an open question about missing severity: every case in the
PSLA-bearing pool has a deterministically recoverable category, and a new query
gets its severity from the judge.

## 7. Code touchpoints

| Concern | Location |
|---|---|
| Attribute set / weights / amalgamation | `src/hklii_psla/experiments/experiment7/cbr_model.py` |
| Two-stage gate, band filtering | `src/hklii_psla/experiments/experiment7/retrieval.py` |
| Relevance labels, metrics | `src/hklii_psla/experiments/experiment7/evaluation.py`, `features.py` |
| LOO loop, weight tuning | `src/hklii_psla/experiments/experiment7/exp7.py` |
| ICD similarity | `src/hklii_psla/experiments/experiment4/icd_similarity.py` |
| Benchmark config | `src/hklii_psla/experiments/experiment7/config.yaml` |

## 8. Recovering `overall_category` from the award

The severity category is a deterministic function of the PSLA award, so a
corpus case with no `overall_category` has been mis-extracted: the category can
be **reconstructed exactly** from the year-adjusted award rather than treated as
a free pass.

Bands in **real 2020 HKD** (inflation-adjusted, i.e. applied to `psla_real`):

| `overall_category` | Lower bound (2020 HKD) | Upper bound (2020 HKD) |
|---|---|---|
| Non-serious injury | 0 | 564,000 |
| Serious injury | 564,000 | 761,000 |
| Substantial injury | 761,000 | 931,000 |
| Gross injury | 931,000 | 1,410,000 |
| Disaster | 1,410,000 | ∞ |

Bands are half-open `[lower, upper)`. A case with a nominal award in year `Y` is
first deflated to 2020 with `data/price_index.csv` (the existing `PriceIndex`
path, base year 2020) and then classified. Equivalently, scale the boundaries by
`index[Y] / index[2020]` and compare against the nominal award.

Implementation sketch:

```python
def infer_severity(psla_real: float | None, price_index) -> str | None:
    # psla_real is already 2020-adjusted by PriceIndex
    for category, lower, upper in SEVERITY_BANDS_2020:  # ordered
        if lower <= psla_real < upper:
            return category
    return None
```

Because `overall_category` is effectively a 5-bin encoding of the award, keep it
on the **compensation** side of the score, not the clinical side:

### What the corpus says

Checked against the 584 cases that carry both an extracted `overall_category`
and a PSLA:

| Extracted category | n | median real PSLA (2020) |
|---|---|---|
| Non-serious injury | 309 | 207,865 |
| Serious injury | 192 | 566,757 |
| Substantial injury | 50 | 621,071 |
| Gross disability | 17 | 1,076,866 |
| Disaster | 16 | 1,698,658 |

The medians are ordered exactly as the bands imply, so the bands are sound. But
per-case agreement between the extracted label and `band(psla_real)` is only
**66%** (74% if we exclude cases within 10% of a boundary), and the mismatches
are mostly far from any boundary — e.g. `[1997] HKCFI 181` is extracted as
"Disaster" with a HK$13k award. The extracted label tracks the judgment's own
descriptive language, not the compensation scheme, so a disagreement is just
another extraction error. Conclusion: for corpus cases **`band(psla_real)` is
the authoritative severity**, and the extracted label should be overridden when
it disagrees, not blended.

- **Candidate severity is target-derived.** For an existing case the category is
  the band of its award, so severity similarity is close to a discretised
  PSLA match. It is legitimate for a judge to *input* severity for a new case
  (a pre-award clinical impression), but it is not an independent clinical
  feature on the corpus side.
- **One true leakage case.** In leave-one-out evaluation the query's own award is
  the held-out target, so the query's severity must stay as extracted — never
  imputed from its own award. Candidate severity may be imputed freely.
- **Do not tune on PSLA-band metrics with severity imputed.** Including a
  target-derived term makes PSLA-band NDCG partly circular. Tune on
  `clinical_ndcg`; compute PSLA-band sanity checks with the severity term
  excluded (or imputation disabled).
- **Boundary noise.** Awards sit near band edges, so imputed severity can flip
  between adjacent categories. Severity similarity should remain ordinal/graded
  rather than a hard exact-match, consistent with the looser clinical stance.
- **Recommendation — keep severity off the clinical score.** Since severity is a
  quantised award, weighting it in the aggregate drags the clinical ranking
  toward the target, and it adds little that the existing PSLA-band-constrained
  retrieval mode does not already provide. Prefer excluding it from the clinical
  aggregate (or using it only as an optional coarse band filter) and reporting
  it as a compensation diagnostic. This is a decision to confirm, not a settled
  convention.

## 9. Implementation status

Implemented in `experiment7`:

- **Severity is off the clinical score.** `DEFAULT_ATTRIBUTES` is the clinical
  group (`injuries`, `losses`) plus the auxiliary group (ages, treatment,
  counts); `gender` and `overall_category` are excluded.
- **`severity_from_award`** (`features.py`) reconstructs the category from the
  year-adjusted award; `CorpusCase.award_severity`/`effective_severity` expose
  it. Candidate instances populate severity from the award band; a query uses its
  own judge/extractor label, so leave-one-out never leaks the held-out award.
- **Clinical-first composite** (`cbr_model.composite_similarities`): the clinical
  group is missing-penalised and carries `clinical_alpha` (0.70) of the score;
  the auxiliary group is available-case.
- **Loose clinical gate** (`cbr_model.clinical_gate_mask`): a minimum graded
  injury similarity (default 0.10) applied when the query carries injuries.
- **Clinical metrics** (`evaluation.clinical_relevance`, `ClinicalRelevance`):
  graded clinical NDCG/precision/recall and injury/loss similarity at k, emitted
  alongside the PSLA-band metrics.
- **Tuning objective** is `clinical_ndcg` by default; the PSLA-band metrics are
  retained as the compensation sanity check.

Re-run (1,336 cases, k=10, 40 trials, 60/40 split):

| Metric (held-out) | Default | Tuned | Previous PSLA-tuned |
|---|---|---|---|
| Clinical NDCG@10 | 0.815 | **0.875** | — |
| Injury similarity@10 | 0.505 | **0.624** | — |
| Min injury similarity@10 | 0.291 | **0.486** | — |
| PSLA-band NDCG@10 | 0.215 | 0.207 | 0.206 |

Caveat: the clinical relevance label is derived from the same injury/loss local
similarities the ranker uses, so clinical NDCG is close to circular — it
primarily confirms the ranker surfaces the best clinical matches, not that those
matches are legally correct. This is acceptable while no human-labelled ground
truth exists (see §6). PSLA-band NDCG remains the only external-ish signal, and
its near-flat movement shows the clinical and compensation objectives pull in
different directions.