# Experiment 6 — Corpus clean-up (notes & status)

## Original problem statement

## Major BUMMMMM
* We forgot to include Burns for the ICD-codes *


We have so far performed an extraction run on a corpus of legal documents (approx. 1400).
The extractions are according to the schema that you can find in **schema.py**.

However, the extraction remains imperfect at times due to the following reasons.

1. Some extractions report null values even though the features can be properly inferred/extracted from the legal documents. Whilst difficult to certainly identify which null fields are true negatives, we can use some heuristics to vet out the extractions that need to be redone. 
  a. We can specify certain fields which act as major red flags. For example, a null field for Gender indicates that the model could do better.
  b. We can measure the proportion of fields that are null. An overwelmingly null dominated extraction should raise a red flag. 

2. Some extractions fail to correctly find the correct neutral citation. For our use case, we should always assume that the neutral citation is inlcuded in the case. A simple regex change can identify these defects.

3. Some injuries have null icd codes. This should be treated differently from the case stated in 1. A null icd code indicates that the injury extracted by the model in fact is not an injury (on its own). But rather, a symptom, after effect, manifestation, etc ... 

4. Some injuries have nodes that pack in a lot of body parts at once on some nodes. A major example of that is the abdomen, pelvic, lumbar, etc ... We need to filter out some of these fields so that we further process them and force the LLM into a decision and go the tree at least down a level.

5.

---

# What Experiment 6 is

Experiment 6 is the **quality-control and clean-up layer** on top of the Experiment 5
corpus extraction. It does **not** re-extract whole cases and does **not** renumber
`inj_00x` ids. It is a one-off, hand-curated clean dataset, produced through **partial,
reviewable edits** — not a reproducible automated pipeline.

It has two halves:

1. **Audit** (`audit.py`, CLI `psla audit-corpus`) — deterministic, no LLM. Walks the
   extracted `Case` JSONs and emits a triage table of *why* each case needs attention.
2. **Patcher** (`patcher.py`, CLI `psla patch-corpus`) — a partial-edit agent that
   proposes minimal diffs (`citation`, `psych`, `coarse`) and can apply them in place.

---

# Corpus & baseline run

| Item | Value |
|---|---|
| Source text | `old_pi_cases_web_text/*.txt` — 1,480 HK personal-injury judgments |
| Extractor | `experiments/experiment5/exp_5.py` (async single-pass, model `gpt` via Azure) |
| Extracted cases | `output/experiments/corpus_extraction/cases/*.json` — **1,478** |
| Progress log | `output/experiments/corpus_extraction/progress.csv` — 1,478 rows, all `status=ok` |
| Remaining gaps | 2 sources never ran: `HKCFI_1999_1392.html.txt`, `HKDC_2015_256.html.txt` |

Nothing has been **applied** to the case files yet — `cases_backup/` does not exist. All
patches produced so far are dry-run proposals.

---

# Design decisions (agreed)

* **Coarse ICD nodes** → declarative **denylist** (`coarse_nodes.json`). If a selected code
  is on the list, flag the case and force the LLM one level deeper.
* **Psychiatric handling** → keep null-ICD psych injuries where they are in `injuries`;
  additionally guarantee exactly **one `Personality change` loss per case**, description
  taken **verbatim** from the item's `source` quotes. Bag-of-words catches wider issues
  (PTSD, post-traumatic disorder, …). `caused_by_injury_id` = primary psych injury id,
  else a linked manifestation's id, else `""`.
* **Neutral citation** → **extraction-first** (see below).
* **Pre-existing degenerative conditions** → *proposed but unconfirmed*: flag
  `NULL_ICD_PREEXISTING`, keep the injury entry, set
  `metadata.has_pre_existing_injuries`.
* **Burns gap** → deferred; tracked by the user in `data/ICD-11.json`.
* Do **not** renumber `inj_00x` ids (would break `caused_by_injury_id`).
* Patches: **propose** to a new file; **apply** overwrites the original in place, after a
  **one-time backup** to `cases_backup/`.

---

# Artifacts

### Configs (`src/hklii_psla/experiments/experiment6/`)
| File | Role |
|---|---|
| `coarse_nodes.json` | 128-node coarse denylist (90 multiple, 29 region_catchall, 9 broad_region). **User will tune this.** |
| `red_flag_fields.json` | 10 high/medium-severity fields that trigger `RED_FLAG_NULL*` when null |
| `psychiatric_keywords.json` | 47 keywords + exclusions; `target_loss_category = Personality change` |
| `qc_config.json` | null-ratio threshold/buckets, treatment patterns, source-noise patterns, citation pattern + court allowlist |

### Tools
| File | Role |
|---|---|
| `audit.py` | Deterministic QC audit; flag definitions; `parse_citation_from_source`, `recover_citation_from_extracted`, `resolve_citation`, `_make_matcher` |
| `patcher.py` | Partial-edit agent: ops `citation`, `psych`, `coarse`; validate → propose → apply + backup; CLI |
| `extractor/icd_classifier.py` | Refactored tree walk: forbidden options, `build_icd_index`, `areclassify_injury`, `_walk_from`/`_awalk_from`, `force_deep` |
| `cli.py` | Added `psla audit-corpus` and `psla patch-corpus` |

### Outputs (`output/experiments/corpus_extraction/`)
| Path | Contents |
|---|---|
| `cases/*.json` | 1,478 extracted `Case` JSONs (untouched) |
| `qc/qc.csv`, `qc/qc_report.json` | Audit triage table + aggregate report |
| `patches/*.patch.json` | Dry-run proposals, tagged by ops (`<stem>.citation.patch.json`, `<stem>.citation-psych.patch.json`) |
| `progress.csv` (+ `.bak`) | Resumable run log; `.bak` is the pre-migration original |

---

# Audit flags (glossary)

Null/red-flag: `RED_FLAG_NULL` (high-severity field null), `RED_FLAG_NULL_MEDIUM`,
`NULL_RATIO_HIGH`, `NULL_ICD_PSYCH` / `_PREEXISTING` / `_BURN` / `_SYMPTOM` / `_OTHER`,
`PREEXISTING_METADATA_MISMATCH`.

Citation: `CITATION_MISSING`, `CITATION_MALFORMED`, `CITATION_EXTRA_TEXT`,
`CITATION_MISMATCH`, `CITATION_CONFLICT`, `CITATION_FROM_SOURCE`, `ACTION_MISSING`.

ICD/injuries: `COARSE_ICD_NODE`, `ICD_STOPPED_EARLY`, `ICD_CODE_NOT_IN_TREE`,
`INJURY_MULTI_BODY_PART`, `ZERO_INJURIES`.

Relations/losses: `UNLINKED_MANIFESTATION`, `UNLINKED_LOSS`, `ZERO_LOSSES`,
`NO_PSLA_AMOUNT`, `NO_COMPARABLE`.

Psych: `PSYCH_DETECTED`, `PSYCH_NO_PERSONALITY_CHANGE_LOSS`.

Source/other: `TREATMENT_CONTAMINATED`, `SOURCE_NOISE`, `SOURCE_MISSING`,
`MULTI_PLAINTIFF`, `JSON_UNREADABLE`.

Current counts (1,478 audited, 1,446 flagged):

```
RED_FLAG_NULL_MEDIUM 903   COARSE_ICD_NODE 650        INJURY_MULTI_BODY_PART 629
NO_COMPARABLE 537          RED_FLAG_NULL 511          NULL_RATIO_HIGH 506
UNLINKED_MANIFESTATION 481 PSYCH_DETECTED 331         PSYCH_NO_PERSONALITY_CHANGE_LOSS 285
UNLINKED_LOSS 255          CITATION_MISMATCH 235      CITATION_MALFORMED 234
CITATION_EXTRA_TEXT 219    NULL_ICD_OTHER 218         ICD_STOPPED_EARLY 207
NULL_ICD_PSYCH 155
```

---

# The citation fix (extraction-first)

The extracted `metadata.neutral_citation` is trusted **first**; the source header is only a
fallback / cross-check. `resolve_citation()` returns one of:

| outcome | meaning |
|---|---|
| `clean` | already exactly `[YYYY] COURT NUM` |
| `extra_text` | correct citation recovered from the extraction, surrounding text stripped |
| `from_source` | extraction yielded nothing → source header supplies it |
| `conflict` | extraction and source **disagree** → left for review, not overwritten |
| `unrecoverable` | neither yields one |

Recovery uses a **court allowlist** (`HKCFI, HKDC, HKCA, UKPC`) so law-report series
(`HKC`, `HKLRD`) are never mistaken for neutral citations. `search` (not `match`) is used so
a party-name prefix does not defeat recovery.

Corpus result: **219** `extra_text`, **16** `from_source`, **0** conflicts, rest clean. The
trap cases resolve correctly, e.g. `[1988] HKC 795 → [1988] HKCFI 461` and
`HKDC 394 → [2007] HKDC 394` (year only available from the source).

---

# Patcher operations

* **`citation`** — extraction-first as above; on conflict, no change; fills an empty
  `action_number` from the source header.
* **`psych`** — if psychiatric content is detected (bag of words over injuries +
  manifestations) and no `Personality change` loss exists, append exactly one; description
  from the detected item's `source`; `caused_by_injury_id` linkage as decided above.
* **`coarse`** — LLM re-walk of the ICD-11 tree for injuries with a coarse / prematurely
  stopped code, forbidding the coarse denylist and forcing one more level. Sequential per
  injury, may make multiple calls; needs a model + ICD index.

Every proposal is validated with `Case.model_validate` before it can be applied. Dry-run
proposals currently on disk: **235** `citation` (235 with changes) and **487**
`citation-psych`.

---

# Experiment 5 hardening

Done so that future corpus runs cannot hang and are resumable:

* `settings.DEFAULT_TIMEOUT` (bumped 60→300s) wired into the OpenRouter / Ollama / Azure
  chat models as their HTTP timeout.
* Per-case hard timeout via `asyncio.wait_for` in `exp_5.py`.
* **Resumable, append-mode `progress.csv`**: on start, rows with `status == "ok"` (and a
  surviving JSON) are skipped; the file is rewritten with the current schema and a one-time
  `.bak`; failed/partial/timed-out/missing rows are re-processed.
* New columns `status`, `error_stage`, `model`; `ExtractionResult.status`
  (`ok` / `partial` / `timeout` / `failed`) distinguishes a Stage-2 ICD failure from a clean
  success (previously both reported `success=True`).
* Optional `fallback_provider` retries Stage 1 failures that look like content-filter
  rejections.

Gap closure: 8 of the 10 outstanding cases were processed (the 5 earlier content-filter
failures now succeeded on `gpt`, plus `HKCFI_2014_1771`, `HKDC_2008_122`, `HKDC_2009_1510`,
and the Stage-2 fix for `HKCFI_2009_880`). **2 remain**: `HKCFI_1999_1392`, `HKDC_2015_256`.

---

# Current state

* 1,478 / 1,480 sources extracted (`status=ok`); 2 gaps remain (above).
* Audit regenerated: 1,478 cases, 1,446 flagged.
* Dry-run proposals only; **nothing applied**; `cases_backup/` absent.
* Tests: **43 passing**; ruff clean.

---

# Commands

```bash
uv run psla audit-corpus

# deterministic ops (dry run, then --apply)
uv run psla patch-corpus --ops citation,psych \
    --select CITATION_MALFORMED,CITATION_MISMATCH,PSYCH_NO_PERSONALITY_CHANGE_LOSS
uv run psla patch-corpus --ops citation,psych \
    --select CITATION_MALFORMED,CITATION_MISMATCH,PSYCH_NO_PERSONALITY_CHANGE_LOSS --apply

# after tuning coarse_nodes.json
uv run psla patch-corpus --ops coarse --provider gpt --select COARSE_ICD_NODE

uv run pytest tests/ -v
```

---

# Handoff notes for the experimenter agent

1. **Close the last 2 gaps.** Re-run Experiment 5; the resumable `progress.csv` will pick up
   only `HKCFI_1999_1392` and `HKDC_2015_256`.
2. **Tune `coarse_nodes.json`**, then run the `coarse` op (dry run → review → `--apply`).
3. **Apply the deterministic `citation` / `psych` patches** once reviewed (still dry-run).
4. **Decide the pre-existing-conditions representation** (proposed `NULL_ICD_PREEXISTING` +
   keep entry + `has_pre_existing_injuries`).
5. **Burns taxonomy gap** — user-tracked in `data/ICD-11.json`; out of scope here.
6. Optional: `metadata.action_number` is present for all cases but ~339 do not match a strict
   `COURT N/NNNN` shape — worth a look if action numbers matter downstream.
7. Gold-set evaluation + regression tests (step 6) were deliberately deferred.

Read `AGENTS.md` for project conventions before making changes.
