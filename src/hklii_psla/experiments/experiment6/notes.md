# Experiment 6 — Corpus clean-up (notes & status)

> **Current snapshot of the corpus lives in
> `output/experiments/corpus_extraction/CORPUS_STATE.md`.** That file is authoritative for
> counts; this file is the design log and experiment history.

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
| Source text | `old_pi_cases_web_text/*.txt` — **1,480** HK personal-injury judgments |
| Extractor | `experiments/experiment5/exp_5.py` (async single-pass, model `gpt` via Azure) |
| Extracted cases | `output/experiments/corpus_extraction/cases/*.json` — **1,480** |
| Progress log | `output/experiments/corpus_extraction/progress.csv` — **1,480 rows, all `status=ok`** |
| Gaps | **none** — every source has a case file |

The corpus is **complete**. (The earlier baseline of 1,478 cases with 2 gaps, and the note
that nothing had been applied yet, are obsolete — see "Session history" below.)

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
* **Pre-existing degenerative conditions** → **deferred** (see Outstanding). No
  `NULL_ICD_PREEXISTING` repair is performed; `metadata.has_pre_existing_injuries` is left
  untouched and will be handled by a separate extraction run.
* **Burns gap** → deferred; tracked by the user in `data/ICD-11.json`.
* Do **not** renumber `inj_00x` ids (would break `caused_by_injury_id`).
* Patches: **propose** to a file; **apply** overwrites the original in place, after a
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
| `cases/*.json` | **1,480** cleaned `Case` JSONs (authoritative) |
| `cases_backup/*.json` | **485** pre-edit originals (first-write backup) |
| `qc/qc.csv`, `qc/qc_report.json` | Audit triage table + aggregate report |
| `qc/applied_patches.csv` | **513** applied changes (235 citation + 278 psych) |
| `qc/psych_false_positives.json` | 7 psych cases deliberately skipped + reasons |
| `qc/renamed_files.csv` | 235 old→new case-filename mappings (from the 2026-09-23 rename) |
| `patches/README.md` | Note on the deleted transient proposals |
| `progress.csv` (+ `.bak`) | Resumable run log; `.bak` is the pre-migration original |
| `CORPUS_STATE.md` | Current corpus snapshot (counts, flags, outstanding work) |

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

Current counts (1,480 audited, 1,443 flagged) — full table in `CORPUS_STATE.md`:

```
RED_FLAG_NULL_MEDIUM 898   COARSE_ICD_NODE 651        INJURY_MULTI_BODY_PART 630
NO_COMPARABLE 538          RED_FLAG_NULL 512          NULL_RATIO_HIGH 507
UNLINKED_MANIFESTATION 482 PSYCH_DETECTED 331         UNLINKED_LOSS 277
NULL_ICD_OTHER 219         ICD_STOPPED_EARLY 207      NULL_ICD_PSYCH 155
NO_PSLA_AMOUNT 142         TREATMENT_CONTAMINATED 124 ZERO_LOSSES 109
NULL_ICD_PREEXISTING 107   MULTI_PLAINTIFF 97         ZERO_INJURIES 57
NULL_ICD_SYMPTOM 54        SOURCE_NOISE 35            NULL_ICD_BURN 26
PREEXISTING_METADATA_MISMATCH 16   PSYCH_NO_PERSONALITY_CHANGE_LOSS 7
ICD_CODE_NOT_IN_TREE 1
```

All `CITATION_*` and `ACTION_MISSING` flags are **0**.

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

**Corpus result: 1,480 / 1,480 citations are now clean** (after the 235 applied citation
patches; 219 were recovered from the extracted field, 16 from the source header). Trap cases
resolved correctly, e.g. `[1988] HKC 795 → [1988] HKCFI 461` and
`HKDC 394 → [2007] HKDC 394`.

---

# Patcher operations

* **`citation`** — extraction-first as above; on conflict, no change; fills an empty
  `action_number` from the source header.
* **`psych`** — if psychiatric content is detected (bag of words over injuries +
  manifestations) and no `Personality change` loss exists, append exactly one; description
  from the detected item's `source`; `caused_by_injury_id` linkage as decided above.
* **`coarse`** — LLM re-walk of the ICD-11 tree for injuries with a coarse / prematurely
  stopped code, forbidding the coarse denylist and forcing one more level. Sequential per
  injury, may make multiple calls; needs a model + ICD index. **Not yet run.**

Every proposal is validated with `Case.model_validate` before it can be applied. The
transient per-case proposal files have been cleaned up — the durable record of what was
applied is `qc/applied_patches.csv`; see `patches/README.md`.

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

All outstanding gaps were subsequently closed; the corpus is 1,480/1,480.

---

# Current state

* **1,480 / 1,480 sources extracted** (`status=ok`); no gaps.
* **All 1,480 case JSONs validate against `Case`.**
* **1,480 / 1,480 neutral citations clean**; no `CITATION_*` or `ACTION_MISSING` flags.
* **1,480 / 1,480 filenames match their neutral citation** (`YYYY_COURT_NUM.json`); 235 renamed.
* **Deterministic `citation` / `psych` patches applied**; `cases_backup/` holds 485 originals.
* **7 anatomical psych false positives skipped** (documented); `Personality change` losses = 348.
* **Coarse ICD op deliberately not applied** (separate pass). Pre-existing conditions deferred.
* Tests: **43 passing**; ruff clean on the `experiment6` + `tests` scope.
* Transient patch proposals cleaned up; `qc/applied_patches.csv` is the applied-change record.

---

# Session history

### Session update (2026-09-23)

1. **Audit re-verified.** `uv run psla audit-corpus` reproduced the prior
   `qc_report.json` / `qc.csv` **byte-for-byte** (1,478 cases / 1,446 flagged), so all
   figures recorded were confirmed before any edits.
2. **Gaps closed.** Re-ran Experiment 5; `progress.csv` is now **1,480 rows, all
   `status=ok`** (`HKCFI_1999_1392`, `HKDC_2015_256` both succeeded).
3. **Citation patches applied.** 235 proposals, all reviewed (219 recovered from the
   extracted field, 16 from the source header, including `[1988] HKC 795 → [1988] HKCFI 461`
   and `HKDC 394 → [2007] HKDC 394`). Applied in place with backups. Re-audit:
   **all `CITATION_*` flags = 0**; clean cases 32 → 37.
4. **Psych patches applied — with 7 anatomical false positives skipped.** The bag-of-words
   triggered on physical `depress*` in 7 cases (eyeball, nasal bridge, tibial plateau, scar,
   scapula, …). Per the agreed call, those cases were **skipped** rather than patched;
   the rest (**278**) each received exactly one `Personality change` loss. The skip list and
   reasons live in `qc/psych_false_positives.json`. `PSYCH_NO_PERSONALITY_CHANGE_LOSS`
   is now **7** (exactly the skipped cases). 348 `Personality change` losses total; the psych
   op introduced **0 dangling `caused_by_injury_id` links**.
5. **Pre-existing conditions:** **out of scope** for this pass (user decision — better
   handled in a separate extraction run). `has_pre_existing_injuries` is left untouched.
6. **Action numbers:** only **28** cases are genuinely malformed; **314** of the ~342 strict
   failures are merely zero-padded (`HCA000447/1968`). Normalisation is **pending a decision**.
7. **Coarse ICD correction:** excluded, to be done separately.
8. **Final clean-up + reporting.** Deleted **1,002** transient per-case proposal files
   (235 `citation`, 489 superseded `citation-psych`, 278 `psych`) and the superseded
   `qc/patch_review.csv`. Added `qc/applied_patches.csv` (513 applied changes),
   `patches/README.md`, and `CORPUS_STATE.md`. Re-ran the audit (1,480 / 1,443 flagged) and
   the full test suite (43 passing).
9. **Case filenames normalised to the neutral citation.** The extractor builds filenames
   from `metadata.neutral_citation`, so the 235 citation-patched cases still carried stale
   names (`1969_HKCFI_66_HCA_447_1968.json`, `1988_HKC_795.json`, party-name fallbacks, …).
   Renamed all **235** to `YYYY_COURT_NUM.json` (0 collisions; 1,245 were already correct).
   `progress.csv`, `qc/applied_patches.csv`, `qc/psych_false_positives.json` and the parallel
   `cases_backup/` entries were updated; the old→new map is `qc/renamed_files.csv`. Re-ran
   the audit (1,480 / 1,443 flagged) and validated all 1,480 cases.

Artifacts added: `qc/psych_false_positives.json`, `qc/applied_patches.csv`,
`qc/renamed_files.csv`, `patches/README.md`, `CORPUS_STATE.md`, `cases_backup/` (485 originals).

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

> Note: re-running the deterministic `citation` / `psych` ops against the current cases is a
> no-op except for the 7 documented psych false positives.

---

# Handoff notes for the experimenter agent

1. ~~**Close the last 2 gaps.**~~ **DONE** — corpus is 1,480/1,480.
2. **Tune `coarse_nodes.json`**, then run the `coarse` op (dry run → review → `--apply`).
   *Not done; deliberately deferred.*
3. ~~**Apply the deterministic `citation` / `psych` patches** once reviewed.~~ **DONE** —
   235 citation + 278 psych applied; 7 anatomical psych false positives skipped.
4. **Pre-existing conditions** — **deferred** (user decision: separate extraction run).
   `NULL_ICD_PREEXISTING` / `has_pre_existing_injuries` left as-is.
5. **Burns taxonomy gap** — user-tracked in `data/ICD-11.json`; out of scope here.
6. Optional: `metadata.action_number` — **28** genuinely malformed; **314** zero-padded.
   Normalisation is pending a decision.
7. Gold-set evaluation + regression tests (step 6) were deliberately deferred.

Read `AGENTS.md` for project conventions before making changes.
See `output/experiments/corpus_extraction/CORPUS_STATE.md` for the current corpus snapshot.
