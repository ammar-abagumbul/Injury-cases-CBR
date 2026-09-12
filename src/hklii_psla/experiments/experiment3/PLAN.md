# Plan — Experiment 3 Changes (a)–(c)

Status: **PROPOSED — awaiting approval. No code has been changed yet.**

---

## Current Implementation (summary)

`exp_3.py` (`ComparableCaseExtractionExperiment`):

1. Globs `*.txt` from `case_files_path` (raw judgments only), extracts each primary
   case with `SinglePassExtractor`, saves to `output_dir/primary/<name>.json`.
2. For each `PSLAComparableCase` cited in `result.case.psla.comparable_cases`:
   fetches the judgment via `JudiciaryClient` (with in-memory fetch/extract caches),
   saves the raw text via `save_judgment(..., prefix=stem)` → currently named
   `{key-stem}-{action_number}.txt` (e.g. `action_DCPI_2723_2018-DCPI_2723_2018.txt`),
   or `temporary_fix.txt` when the action number is missing (silent collision bug).
3. Extracts the comparable judgment, saves to `output_dir/comparable_cases/<name>.json`.
4. Records everything in `ComparableLink` dataclasses; `save_results()` writes
   `manifest.json` (full link dump) and prints a summary table.

Relevant facts discovered:

- `JudiciaryClient._extract_action_number()` normalizes action numbers to
  `COURT_NUM_VOL` (e.g. `DCPI_2723_2018`) — raw action numbers contain `/` and
  cannot be used verbatim in filenames.
- `Judgment.neutral_citation` is never populated by `JudiciaryClient.retrieve()`
  (always `None`); only `action_number` and `text` are set.
- `save_judgment` is shared with `judgement_retriever/cli.py`, so changes to it
  affect both callers (both benefit from the fallback fix below).
- `CaseMetadata.neutral_citation` / `action_number` default to `""` (empty string),
  not `None`.

---

## Change A — Support pre-extracted JSON input

### Config

Add to `Exp3Config` (and `config.yaml`):

```yaml
case_files_format: "raw"   # "raw" (.txt judgments) | "extracted" (.json Case files)
```

```python
case_files_format: Literal["raw", "extracted"] = "raw"
```

### Behavior

- `"raw"` (default, current behavior): glob `*.txt`, extract each primary case.
- `"extracted"`: glob `*.json`, load each via `Case.model_validate(json.load(...))`.
  - **Primary extraction is skipped entirely** — no LLM call for primaries.
    (The LLM is still created for comparable-case extraction.)
  - Loaded cases are **not** re-saved into `output_dir/primary/` (avoids
    duplicating data); the CSV (Change C) records the source file name.
  - A synthetic `ExtractionResult(case=loaded_case, model_name="pre-extracted")`
    (zero duration, zero tokens) is appended to `batch.runs` with
    `strategy="pre-extracted"` so the run manifest stays complete and
    auditable. Invalid JSON / schema violations are recorded as failed runs
    (`result.case=None`, error message) and skipped for comparable processing.

### Validation

- No files of the expected extension → `RuntimeError` (as today).
- Directory contains **both** `.txt` and `.json` → hard error with a clear
  message (you specified "not a mix"; failing fast prevents silently ignoring
  half the directory).

---

## Change B — Comparable judgment naming: `<action_number>.txt`

### Behavior

- In `exp_3.py`, call `save_judgment(judgment, output_dir=...)` **without**
  the `prefix` argument → files named `DCPI_2723_2018.txt` (the normalized
  action number produced by `JudiciaryClient`).
- Fix the fallback chain in `save_judgment` (`storage.py`), replacing the
  `temporary_fix.txt` placeholder (which silently overwrites different
  judgments with the same name):

  1. `judgment.action_number` → `<action_number>.txt`
  2. else `judgment.neutral_citation` → `citation_to_filename()` →
     `HKDC_2020_1745.txt` (already implemented, currently unused)
  3. else → raise `ValueError` with a clear message.

  `exp_3.py` catches the `ValueError`, records the link with
  `fetch_status="error"`, and continues.

- Note: the *normalized* form (`DCPI_2723_2018.txt`) is used because raw
  action numbers contain `/`. The **raw** action number (e.g.
  `DCPI 2723/2018`) is preserved in the CSV (Change C) via the extracted
  `Case.metadata.action_number`.
- Duplicate fetches of the same judgment (cited by multiple primaries) write
  the same filename — content is identical, so overwrite is harmless, and the
  CSV will contain one row per (source, comparable) pair.

---

## Change C — CSV relationship table (source ↔ comparable)

### New output: `output_dir/comparable_links.csv`

Written in `save_results()` with `csv.writer`. One row per
(source case, cited comparable case) pair — including rows for fetch
failures, so the table is a complete record of every cited comparable.

### Columns

| # | Column | Source of value | When null/empty |
|---|--------|-----------------|-----------------|
| 1 | `source_file_name` | basename of the input file (`.txt` or `.json`) | never |
| 2 | `source_neutral_citation` | primary `Case.metadata.neutral_citation` | empty metadata |
| 3 | `source_action_number` | primary `Case.metadata.action_number` | empty metadata |
| 4 | `comparable_file_name` | basename of fetched judgment (`DCPI_2723_2018.txt`) | not fetched |
| 5 | `comparable_neutral_citation` | extracted comparable `Case.metadata.neutral_citation`, fallback to the citation as cited in the primary (`cc.neutral_citation`) | neither available |
| 6 | `comparable_action_number` | extracted comparable `Case.metadata.action_number`, fallback to `cc.action_number` as cited | neither available |
| 7 | `fetch_status` | `fetched` / `not_found` / `error` / `skipped_no_identifier` | never |
| 8 | `extraction_status` | `success` / `failed` / `not_attempted` | never |
| 9 | `comparable_case_json_file` | basename of the comparable extraction JSON | not extracted |

Columns 7–9 are the "etc." — they make the table self-contained for
filtering broken links. Say the word if you want them dropped.

- **Null representation:** empty cell (standard CSV convention). Alternative:
  literal `null` string — tell me if you prefer that.
- **`manifest.json` is kept** as-is (it additionally holds full paths and error
  messages, useful for debugging). The CSV becomes the canonical
  relationship table. Happy to drop the manifest if you'd rather have a
  single artifact.

### Implementation

- Extend `ComparableLink` with: `primary_file_name` (basename),
  `judgment_file_name`, `comparable_case_neutral_citation`,
  `comparable_case_action_number`, `comparable_case_json_file` — populated
  in `_process_comparable_case()` after extraction.
- Add a small `_write_links_csv()` helper called from `save_results()`.

---

## Files to be modified

| File | Change |
|------|--------|
| `src/hklii_psla/experiments/experiment3/exp_3.py` | A: input-format flag + JSON loading; B: drop `prefix`; C: extend `ComparableLink`, CSV writer |
| `src/hklii_psla/experiments/experiment3/config.yaml` | add `case_files_format: "raw"` |
| `src/hklii_psla/judgement_retriever/storage.py` | B: fallback chain in `save_judgment`, remove `temporary_fix.txt` |

No changes to `schemas.py`, `JudiciaryClient`, or other experiments.

---

## Validation plan

Testing is handled by the user — no test cases written or run by the agent.
Each objective (a), (b), (c) is committed to git separately.

---

## Open questions

1. CSV nulls: empty cells (proposed) or literal `null`?
2. Keep `manifest.json` alongside the CSV (proposed), or replace it?
3. Mixed `.txt` + `.json` input directory: hard error (proposed) or warning?
4. For `extracted` input, should the loaded cases still appear in the run
   manifest as `strategy="pre-extracted"` runs (proposed), or be omitted?