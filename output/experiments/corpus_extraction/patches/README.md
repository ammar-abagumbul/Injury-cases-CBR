# Patches — cleaned 2026-09-23

This directory previously held **1,002 transient per-case proposal files** produced by the
Experiment 6 patcher (`psla patch-corpus`):

| Files | Ops | Status |
|---|---|---|
| 235 `*.citation.patch.json` | `citation` | **applied** in place |
| 489 `*.citation-psych.patch.json` | `citation` + `psych` | **superseded** (early combined run) |
| 278 `*.psych.patch.json` | `psych` | **applied** in place |

All of them have served their purpose and were deleted in the final clean-up. The patcher
still writes new proposals here by default, so the directory is kept.

The authoritative record of what was actually changed is:

* `../qc/applied_patches.csv` — the **513 applied changes** (235 citation + 278 psych),
  with before/after values and evidence notes.
* `../qc/psych_false_positives.json` — the **7 psych cases deliberately skipped**
  (anatomical `depress*` false positives).
* `../cases_backup/` — the **485 pre-edit originals** (first-write backups).

The current case files under `../cases/` are the source of truth. Do not re-apply the
deleted proposals; re-running the deterministic ops against the current cases is a no-op
except for the 7 documented false positives.
