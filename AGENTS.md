# AGENTS.md — Project Conventions & Technical Choices

> This file documents decisions made during project setup.
> Follow-up agents **must** adhere to these conventions.

---

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.12 | SHAP/llvmlite incompatible with 3.13; 3.12 is latest stable compatible |
| **Package manager** | `uv` (v0.7.10) | Fast, modern, lockfile-based. Replaces pip/poetry |
| **Build backend** | `hatchling` | Default for `uv init --package` |
| **LLM framework** | `langchain` (v1.3+) | Unified interface across different providers (deepseek, qwen, GLM, etc ...)
| **Schema validation** | `pydantic` v2 | Type-safe extraction output; `model_validate()` for repair pass |
| **CLI** | `typer` + `rich` | Clean CLI with colored output |
| **Testing** | `pytest` | Standard; `pytest-asyncio` available if needed |
| **Config** | `pydantic-settings` | Loads from `.env` + environment variables |
| **Evaluation** | `numpy` | Macro-averaged precision/recall/F1; edge metrics for relations |

---

## Project Structure

```
src/hklii_psla/
  __init__.py          # Public API exports
  schemas.py           # All Pydantic models (Case, Injury, Loss, etc.)
  config.py            # Settings via pydantic-settings
  model_factory.py     # Provider → LangChain model factory
  cli.py               # Typer CLI (psla command)
  evaluation.py        # Metrics computation (evaluate())
  extractor/
    base.py            # BaseExtractor, ExtractionResult, prompts
    single_pass.py     # SinglePassExtractor
    section_by_section.py  # SectionBySectionExtractor
    multi_agent.py     # MultiAgentExtractor
  experiments/
    runner.py          # Orchestrates experiments 1.1–1.5

tests/
  test_schemas.py
  test_extractor.py
  test_evaluation.py

sample_cases/          # Raw judgment .txt files
benchmark/             # Gold-standard human-extracted JSON (to be populated)
data/                  # Additional data (CSV, etc.)
output/experiments/    # Experiment results (JSON)
```

---

## Key Conventions

### Schemas (`schemas.py`)
- All extraction output conforms to `Case` Pydantic model
- Loss categories use the **REVISED_LOSS.md** taxonomy (27 canonical categories)
- `InjuryLossRelation` is extracted **only when explicitly supported** by the judgment
- `ALL_LOSS_CATEGORIES` list is the canonical source of truth for loss category names

### Extraction
- All extractors inherit from `BaseExtractor` and return `ExtractionResult`
- `ExtractionResult.case` is `None` on failure; check `result.success`
- JSON parsing is best-effort: tries direct parse, then markdown code blocks, then bare objects
- `_dict_to_case()` requires top-level keys `metadata`, `plaintiff`, `injuries`, `treatment`, `losses`, `psla`

### Model Factory
- Provider keys: `deepseek`, `qwen`, `glm`, `ollama`, (and a few more)
- Model names configured via `Settings` (env vars or `.env`)
- Ollama uses `OLLAMA_BASE_URL` (default `http://localhost:11434`)

### Evaluation
- `evaluate(predicted, gold)` returns `ExtractionMetrics`
- Scalar fields: exact match (with None/empty tolerance)
- List fields: set-based overlap (precision/recall/F1)
- Empty lists on both sides → perfect agreement (F1=1.0)
- Edge metrics computed separately for `injury_loss_relations`

### CLI
- Entry point: `psla` (defined in `pyproject.toml`)
- Commands: `run-stage1`, `run-experiment`, `extract`, `validate-schemas`, `list-providers`
- Use `uv run psla <command>` to invoke

### Testing
- Run: `uv run pytest tests/ -v`
- All 30 tests must pass before considering a milestone complete

---

## API Keys

Set in `.env` (copy from `.env.example`):
```
OPENROUTER_API_KEY=

```

Ollama runs locally — no key needed.

---

## Loss Taxonomy (27 categories)

1. Loss of bodily integrity
2. Loss of the senses
3. Loss from scarring or disfigurement
4. Loss of mobility
5. Loss of independence
6. Loss of ability to do domestic tasks
7. Loss of mental ability
8. Loss of communication ability
9. Personality change
10. Loss of confidence in going out in the public
11. Loss of ability to enjoy food and drink
12. Loss of ability to enjoy quiet or solitude
13. Loss of family life
14. Loss of ability to participate in parenthood or grandparenthood
15. Loss of marriage prospects
16. Breakdown of the family
17. Loss of sexual function and sexual life
18. Loss of ability to give birth
19. Loss of ability to give natural childbirth
20. Loss of social life
21. Loss of ability to engage in sports and hobbies
22. Loss of holidays or special occasions
23. Loss of congenial employment
24. Loss of ability to pursue education
25. Loss of ability to volunteer or engage in community service
26. Loss of expectation of life
27. Other
