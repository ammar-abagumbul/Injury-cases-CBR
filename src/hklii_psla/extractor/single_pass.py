"""
Single-pass extraction strategy.

The entire judgment is passed to the LLM in a single call.
Uses ``with_structured_output(Case)`` so the model returns a Pydantic object directly.
"""

from __future__ import annotations

import time

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from hklii_psla.extractor.base import (
    BaseExtractor,
    ExtractionResult,
    SYSTEM_PROMPT_BASE,
)
from hklii_psla.schemas import Case


class SinglePassExtractor(BaseExtractor):
    """Extract all features from the judgment in a single LLM call."""

    def extract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        model_name = self.model_name or getattr(self.model, "model_name", "unknown")
        t0 = time.perf_counter()

        user_prompt = self._build_user_prompt(judgment_text)

        # Truncate if needed (most models have context limits)
        max_input = 120_000  # safe upper bound for most models
        if len(user_prompt) > max_input:
            user_prompt = user_prompt[:max_input] + "\n\n[JUDGMENT TRUNCATED DUE TO LENGTH]"

        try:
            structured_model = self._case_extractor()
            result: Any = structured_model.invoke([
                SystemMessage(content=SYSTEM_PROMPT_BASE),
                HumanMessage(content=user_prompt),
            ])
            case: Case = result
            token_count = _estimate_tokens(SYSTEM_PROMPT_BASE) + _estimate_tokens(user_prompt)
            duration_ms = (time.perf_counter() - t0) * 1000
            return ExtractionResult(
                case=case,
                raw_response=None,  # structured output doesn't return raw text
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                token_count=token_count,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - t0) * 1000
            print(e)
            return ExtractionResult(
                case=None,
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _get_content(response) -> str:
    """Extract string content from a LangChain message response."""
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            # Some models return a list of content blocks
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)
    return str(response)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)
