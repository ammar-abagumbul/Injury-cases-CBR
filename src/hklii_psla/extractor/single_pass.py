"""
Single-pass extraction strategy.

The entire judgment is passed to the LLM in a single call.
Uses `with_structured_output(Case, include_raw=True)` so that parsing
failures can be inspected without losing the raw model response.

After successful Stage 1 extraction, an optional Stage 2 classifies each
extracted injury with an ICD-11 code by interactively navigating the
ICD-11 taxonomy tree.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from hklii_psla.config import settings
from hklii_psla.extractor.base import (
    SYSTEM_PROMPT_BASE,
    BaseExtractor,
    ExtractionMetadata,
    ExtractionResult,
    TokenCount,
)
from hklii_psla.extractor.icd_classifier import (
    ICDClassificationError,
    aclassify_injuries,
    classify_injuries,
    load_icd_tree,
)
from hklii_psla.extractor.utils import (
    build_structured_output_error,
    extract_token_counts,
)
from hklii_psla.schemas import Case, Injury

logger = logging.getLogger(__name__)


class SinglePassExtractor(BaseExtractor):
    """
    Extract all features from the judgment in a single LLM call,
    then optionally classify injuries with ICD-11 codes.
    """

    def extract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        # Stage 1: structured extraction
        result = self._stage1_extract(judgment_text, prompt_style)

        if not result.success or not result.case:
            self._attach_metadata(result)
            return result

        injuries = result.case.injuries.injuries

        if not injuries:
            self._attach_metadata(result)
            return result

        # Stage 2: ICD-11 classification
        try:
            classified_injuries, stage2_tc = self._stage2_classify(injuries)
            result.case.injuries.injuries = classified_injuries
            result.token_count = result.token_count + stage2_tc
        except ICDClassificationError as exc:
            result.error = (
                f"Stage 2 ICD-11 classification failed: {exc}"
            )

        self._attach_metadata(result)
        return result

    async def aextract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        """Async twin of :meth:`extract`."""
        # Stage 1: structured extraction
        result = await self._astage1_extract(judgment_text, prompt_style)

        if not result.success or not result.case:
            self._attach_metadata(result)
            return result

        injuries = result.case.injuries.injuries

        if not injuries:
            self._attach_metadata(result)
            return result

        # Stage 2: ICD-11 classification
        try:
            classified_injuries, stage2_tc = await self._astage2_classify(injuries)
            result.case.injuries.injuries = classified_injuries
            result.token_count = result.token_count + stage2_tc
        except ICDClassificationError as exc:
            result.error = (
                f"Stage 2 ICD-11 classification failed: {exc}"
            )

        self._attach_metadata(result)
        return result

    async def aextract_many(
        self,
        judgments: list[str],
        prompt_style: str = "zero-shot",
        max_concurrency: int | None = None,
    ) -> list[ExtractionResult]:
        """Extract many judgments concurrently.

        Each judgment is processed by :meth:`aextract` behind a semaphore so
        at most ``max_concurrency`` model calls are in flight at once. Results
        are returned in the same order as ``judgments``. A failure in one task
        never cancels the others.
        """
        if not judgments:
            return []

        if max_concurrency is None:
            max_concurrency = settings.MAX_CONCURRENT_REQUESTS

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _extract_one(judgment_text: str) -> ExtractionResult:
            async with semaphore:
                try:
                    return await self.aextract(judgment_text, prompt_style)
                except Exception as exc:
                    logger.error(f"Concurrent extraction failed: {exc}")
                    return ExtractionResult(
                        case=None,
                        model_name=self.model_name,
                        prompt_style=prompt_style,
                        error=str(exc),
                    )

        return await asyncio.gather(
            *(_extract_one(judgment) for judgment in judgments)
        )

    @staticmethod
    def _attach_metadata(result: ExtractionResult) -> None:
        """Record extraction-stage runtime metadata on the result."""
        result.extraction_metadata = ExtractionMetadata(
            model_name=result.model_name,
            extraction_token_count=result.token_count,
            extraction_duration_ms=result.duration_ms,
        )

    def _stage1_extract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        model_name = self.model_name or getattr(
            self.model,
            "model_name",
            "unknown",
        )

        t0 = time.perf_counter()

        user_prompt = self._build_user_prompt(judgment_text)

        # Truncate if needed.
        max_input = 120_000
        result: Any = None

        if len(user_prompt) > max_input:
            user_prompt = (
                user_prompt[:max_input]
                + "\n\n[JUDGMENT TRUNCATED DUE TO LENGTH]"
            )

        try:
            structured_model = self._case_extractor(include_raw=True)

            result = structured_model.invoke([
                SystemMessage(content=SYSTEM_PROMPT_BASE),
                HumanMessage(content=user_prompt),
            ])

        except Exception as exc:
            # The model/API/invocation itself failed.
            duration_ms = (time.perf_counter() - t0) * 1000
            error = str(exc)
            logger.error(f"API invocation failed: {exc}")

            return ExtractionResult(
                case=None,
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                error=error,
                debug=str(result)
            )

        duration_ms = (time.perf_counter() - t0) * 1000
        token_count = extract_token_counts(result)

        case = result.get("parsed")

        if isinstance(case, Case):
            return ExtractionResult(
                case=case,
                raw_response=None,
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                token_count=token_count,
                error=None,
                debug=None,
            )

        # Structured-output parsing failure
        error = build_structured_output_error(result)

        return ExtractionResult(
            case=None,
            raw_response=None,
            model_name=model_name,
            prompt_style=prompt_style,
            duration_ms=duration_ms,
            token_count=token_count,
            error=error,
            debug=result.get('raw'),
        )

    async def _astage1_extract(
        self,
        judgment_text: str,
        prompt_style: str = "zero-shot",
    ) -> ExtractionResult:
        """Async twin of :meth:`_stage1_extract`."""
        model_name = self.model_name or getattr(
            self.model,
            "model_name",
            "unknown",
        )

        t0 = time.perf_counter()

        user_prompt = self._build_user_prompt(judgment_text)

        # Truncate if needed.
        max_input = 120_000
        result: Any = None

        if len(user_prompt) > max_input:
            user_prompt = (
                user_prompt[:max_input]
                + "\n\n[JUDGMENT TRUNCATED DUE TO LENGTH]"
            )

        try:
            structured_model = self._case_extractor(include_raw=True)

            result = await structured_model.ainvoke([
                SystemMessage(content=SYSTEM_PROMPT_BASE),
                HumanMessage(content=user_prompt),
            ])

        except Exception as exc:
            # The model/API/invocation itself failed.
            duration_ms = (time.perf_counter() - t0) * 1000
            error = str(exc)
            logger.error(f"API invocation failed: {exc}")

            return ExtractionResult(
                case=None,
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                error=error,
                debug=str(result)
            )

        duration_ms = (time.perf_counter() - t0) * 1000
        token_count = extract_token_counts(result)

        case = result.get("parsed")

        if isinstance(case, Case):
            return ExtractionResult(
                case=case,
                raw_response=None,
                model_name=model_name,
                prompt_style=prompt_style,
                duration_ms=duration_ms,
                token_count=token_count,
                error=None,
                debug=None,
            )

        # Structured-output parsing failure
        error = build_structured_output_error(result)

        return ExtractionResult(
            case=None,
            raw_response=None,
            model_name=model_name,
            prompt_style=prompt_style,
            duration_ms=duration_ms,
            token_count=token_count,
            error=error,
            debug=result.get('raw'),
        )

    def _stage2_classify(
        self,
        injuries: list[Injury],
    ) -> tuple[list[Injury], TokenCount]:
        """
        Run ICD-11 classification on extracted injuries.

        Returns the classified injuries and aggregate token counts
        for all classification LLM calls.
        """
        icd_tree = load_icd_tree(
            settings.DATA_DIR / "ICD-11.json"
        )

        return classify_injuries(
            self.model,
            injuries,
            icd_tree,
        )

    async def _astage2_classify(
        self,
        injuries: list[Injury],
    ) -> tuple[list[Injury], TokenCount]:
        """Async twin of :meth:`_stage2_classify`."""
        icd_tree = load_icd_tree(
            settings.DATA_DIR / "ICD-11.json"
        )

        return await aclassify_injuries(
            self.model,
            injuries,
            icd_tree,
            max_concurrency=settings.MAX_CONCURRENT_REQUESTS,
        )
