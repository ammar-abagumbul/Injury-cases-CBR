"""
Base extractor interface and shared utilities.

All extraction strategies use LangChain's `with_structured_output()` to enforce
Pydantic schema conformance natively.  Manual JSON format strings are eliminated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from hklii_psla.schemas import Case, ALL_LOSS_CATEGORIES


@dataclass
class TokenCount:
    """Breakdown of token usage for an LLM call or aggregation of calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenCount) -> TokenCount:
        return TokenCount(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


SYSTEM_PROMPT_BASE = (
    "You are a legal information extraction system specialised in Hong Kong "
    "personal injury judgments.\n\n"
    "Extract the requested structured information from the provided judgment text.\n\n"
    "Rules:\n"
    "- Extract only information explicitly stated in the judgment.\n"
    "- Use null for any field not mentioned.\n"
    "- Do NOT infer information not present in the text.\n"
    "- Dates should be in YYYY-MM-DD format where possible.\n"
    "- Monetary amounts should be numeric (float) values in HKD.\n"
    "- Counts (days, number of operations, etc.) should be integers.\n"
    "- For the 'category' field in Loss items, use EXACTLY one of the canonical "
    "loss category names listed below.\n"
    "- Include verbatim references in evidence fields when the relationship is "
    "explicitly stated."
    "- Since you will be doing a JSON extraction, ensure your output is valid JSON."
    "avoid overly using skip characters between double quotes.\n"
)

ZERO_SHOT_USER_TEMPLATE = (
    "Extract all structured information from this judgment.\n\n"
    "---\n{judgment_text}\n---\n\n"
)

# Shorter prompt for section-by-section extraction
SECTION_PROMPT_TEMPLATE = (
    "Extract the following section from this judgment excerpt:\n\n"
    "Section: {section_name}\n\n"
    "Judgment excerpt:\n---\n{section_text}\n---"
)


@dataclass
class ExtractionResult:
    """Result of running extraction on a single case."""

    case: Case | None
    raw_response: str | None = None
    model_name: str = ""
    prompt_style: str = ""
    duration_ms: float = 0.0
    token_count: TokenCount = field(default_factory=TokenCount)
    error: str | None = None
    debug: str | None = None

    @property
    def success(self) -> bool:
        return self.case is not None



class BaseExtractor(ABC):
    """Abstract base for all extraction strategies.

    Subclasses call ``self.structured_model`` — a version of the chat model
    wrapped with ``.with_structured_output(Case)`` (or another Pydantic model)
    — to extract data natively.
    """

    def __init__(self, model: BaseChatModel, model_name: str = ""):
        self.model = model
        self.model_name = model_name

    @abstractmethod
    def extract(self, judgment_text: str) -> ExtractionResult:
        """Extract structured Case from raw judgment text."""
        ...

    def _case_extractor(self, include_raw: bool = False):
        """Return the chat model configured to produce a ``Case`` directly."""
        return self.model.with_structured_output(
            Case,
            # method="function_calling",
            include_raw=include_raw
        )

    def _build_user_prompt(self, judgment_text: str) -> str:
        return ZERO_SHOT_USER_TEMPLATE.format(
            judgment_text=judgment_text,
        )

    @staticmethod
    def load_judgment(path: Path) -> str:
        """Load judgment text from a file."""
        with open(path, encoding="utf-8") as f:
            return f.read()
