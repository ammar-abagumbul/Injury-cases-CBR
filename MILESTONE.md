# MILESTONE.md — Stage 1 Foundation

## Milestone: Project Scaffold & Core Infrastructure

**Date**: 2026-08-07  
**Status**: ✅ Complete

---

## What Was Built

### 1. Project Initialization

- Set up `uv`-based Python 3.12 project (Python 3.13 incompatible with SHAP/llvmlite)
- All Stage 1 dependencies installed: `pydantic`, `langchain` (+ providers), `ollama`, `instructor`, `pytest`, `numpy`, `pandas`, `matplotlib`, `plotly`
- `.env.example` created for API key configuration

### 2. Pydantic Schemas (`src/hklii_psla/schemas.py`)

Full type-safe data model covering all features from `FEATURES.md`:

| Model | Purpose |
|---|---|
| `CaseMetadata` | Citation, action number, case name, date, plaintiff count |
| `PlaintiffBackground` | Gender, age, occupation, salary, education |
| `Injury` / `InjurySummary` | Injury descriptions, types, body parts, severity category |
| `Treatment` | Hospitalisation, operations, sick leave, treatments |
| `Loss` / `LossSummary` | All 27 canonical loss categories (per `REVISED_LOSS.md`) |
| `PSLAAward` / `PSLAComparableCase` | PSLA award amount + benchmark comparables |
| `DeathInfo` | Consciousness, time between accident and death |
| `InjuryLossRelation` | Explicit injury→loss edges with paragraph evidence |

All 27 loss categories available as `ALL_LOSS_CATEGORIES` list.

### 3. Extraction Framework (`src/hklii_psla/extractor/`)

Three extraction strategies implemented:

| Strategy | File | Description |
|---|---|---|
| **Single-pass** | `single_pass.py` | One LLM call per case |
| **Section-by-section** | `section_by_section.py` | Split judgment by headers, extract each domain separately |
| **Multi-agent** | `multi_agent.py` | 9 specialised agents (metadata, plaintiff, injury, treatment, loss, PSLA, death, relation) |

Shared infrastructure in `base.py`:
- `BaseExtractor` abstract class
- `ExtractionResult` dataclass (with `success` property)
- Prompt templates (`zero-shot`, `few-shot`)
- Best-effort JSON parsing (direct → code block → bare object)
- `_dict_to_case()` with required-key validation

### 4. Model Factory (`src/hklii_psla/model_factory.py`)

LangChain-based unified interface:
- `deepseek`, `qwen`, `glm`, `ollama` (and a few more) -> `ChatOpenRouter` (via `OPENROUTER_API_KEY`)

- `ollama` → `ChatOllama` (Qwen3, local)

### 5. Evaluation Module (`src/hklii_psla/evaluation.py`)

`evaluate(predicted, gold)` returns `ExtractionMetrics`:
- **Scalar fields**: Exact match (with None/empty tolerance)
- **List fields**: Set-based overlap (precision/recall/F1)
- **Empty lists**: Both empty → perfect F1=1.0
- **Edge metrics**: Separate precision/recall/F1 for `InjuryLossRelation`
- **Hallucination tracking**: Counts FP items in list fields

### 6. Experiment Runner (`src/hklii_psla/experiments/runner.py`)

Orchestrates all Stage 1 experiments:

| Experiment | What it compares |
|---|---|
| **1.1** | Model providers (GPT, Claude, Gemini, Ollama) |
| **1.2** | Prompt design (zero-shot vs few-shot) |
| **1.3** | Extraction strategy (single-pass vs section-by-section vs multi-agent) |
| **1.4** | Schema validation (deferred to evaluation phase) |
| **1.5** | Injury-loss relation extraction |

Saves results to `output/experiments/` as structured JSON.

### 7. CLI (`src/hklii_psla/cli.py`)

Entry point: `psla` (via `uv run psla <command>`)

| Command | Purpose |
|---|---|
| `psla run-stage1` | Run all Stage 1 experiments |
| `psla run-experiment 1.1` | Run a specific experiment |
| `psla extract <file>` | Extract a single judgment |
| `psla validate-schemas` | Validate Pydantic models |
| `psla list-providers` | Show available model providers |

### 8. Test Suite

**30 tests, all passing:**

| File | Tests | What's covered |
|---|---|---|
| `test_schemas.py` | 18 | Case, Metadata, Plaintiff, Injury, Treatment, Loss, PSLA, Death, Relations, loss categories |
| `test_extractor.py` | 7 | JSON parsing (valid, code block, bare block, invalid), dict-to-Case conversion |
| `test_evaluation.py` | 5 | Perfect match, null prediction, edge metrics, hallucination counting |

---

## File Summary

```
HKLII-PSLA-Experiments/
├── .env.example                          # API key template
├── .python-version                       # Pinned to 3.12
├── AGENTS.md                             # Conventions for follow-up agents
├── MILESTONE.md                          # This file
├── pyproject.toml                        # Project config, dependencies, CLI entry
├── sample_cases/                         # 6 raw judgment .txt files
├── benchmark/                            # (empty — gold standard goes here)
├── data/                                 # (empty — CSV data)
├── output/experiments/                   # (created at runtime)
├── src/hklii_psla/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── evaluation.py
│   ├── model_factory.py
│   ├── schemas.py
│   ├── extractor/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── multi_agent.py
│   │   ├── section_by_section.py
│   │   └── single_pass.py
│   └── experiments/
│       ├── __init__.py
│       └── runner.py
└── tests/
    ├── test_evaluation.py
    ├── test_extractor.py
    └── test_schemas.py
```

---

## Next Steps (Stage 1 Continuation)

1. **Run experiments**: Configure API keys in `.env`, then `uv run psla run-stage1`
2. **Gold standard benchmarking**: Populate `benchmark/` with human-extracted JSON files
3. **Evaluate extraction quality**: Run `evaluate()` against gold standard for each experiment
4. **Experiment 1.4 analysis**: Compute validation pass rate, repair success rate, hallucination rate
5. **Select best configuration**: Freeze model + prompt + strategy for Stage 2

---

## Decisions Made

- **Python 3.12**: Required for SHAP compatibility (Stage 4). `uv python pin 3.12` done.
- **No external LLM calls in tests**: All 30 tests are pure unit tests (no API keys needed).
- **Extractor pattern**: Abstract `BaseExtractor` → three concrete strategies, each returns `ExtractionResult`.
- **JSON parsing**: Best-effort (direct → code block → bare regex) to handle diverse LLM outputs.
- **Empty list handling**: Both empty → F1=1.0 (avoids penalising for correct emptiness).
