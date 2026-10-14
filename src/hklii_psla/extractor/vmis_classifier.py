"""
VMIS surgical-procedure classification for extracted treatments.

Each ``TreatmentItem`` flagged ``is_surgical`` is mapped onto the schedule.

Walks the VMIS Schedule of Surgical Procedures (``data/VMIS.json``) top-down
with the LLM, mirroring the ICD-11 injury classifier:

  1. Select the body system   ("MUSCULOSKELETAL SYSTEM")
  2. Select the sub-region    ("Fracture / dislocation")
  3. Select the procedure     ("Open reduction for arm / leg / patella ... Major")

At every level the model may answer "None of these", in which case the
procedure is not in the schedule and all ``vmis_*`` fields stay ``None``.
The complexity tier (Minor / Intermediate / Major / Complex) comes from the
schedule itself, never from the model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..schemas import TreatmentItem, VmisCategory
from .base import TokenCount
from .icd_classifier import _make_choice_model
from .utils import extract_token_counts


class VmisClassificationError(Exception):
    """Raised when VMIS classification fails for any treatment."""


def load_vmis_tree(path: Path) -> list[dict[str, Any]]:
    """Load ``data/VMIS.json`` (built by ``scripts/build_vmis_json.py``)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise VmisClassificationError(
            f"VMIS JSON must be a list, got {type(data).__name__}"
        )
    return data


VMIS_SYSTEM_PROMPT = (
    "You are a medical coding assistant. Your task is to match a surgical "
    "procedure described in a legal judgment to the Hong Kong VMIS Schedule of "
    "Surgical Procedures.\n\n"
    "Rules:\n"
    "- Read the procedure description carefully.\n"
    "- Select the SINGLE option that best matches the procedure.\n"
    "- Match on the operation actually performed, not on the injury it treated.\n"
    "- Choose the listed operation of the same kind on the same body area, even "
    "if the wording differs (e.g. screw or plate fixation is a fixation procedure).\n"
    "- Choose 'None of these' only when no listed operation is of the same kind. "
    "Do not force a match.\n"
    "- Respond ONLY with a valid option index."
)

_NONE_OPTION = "None of these — the procedure is not in the schedule"


def _node_label(node: dict[str, Any]) -> str:
    return node.get("procedure") or node.get("region") or node.get("section", "?")


def _build_level_prompt(
    description: str, source: str, level_name: str, children: list[dict[str, Any]]
) -> str:
    options = [_node_label(child) for child in children] + [_NONE_OPTION]
    option_lines = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    return (
        f"Procedure: {description}\n\n"
        f"Source from document: {source}\n\n"
        f"Current level: {level_name}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Choose the SINGLE best match. Respond with only the option index."
    )


def _resolve_choice(choice: int, num_children: int) -> int | None:
    """Child index, or ``None`` when the model chose 'None of these'."""
    if choice == num_children:
        return None
    if 0 <= choice < num_children:
        return choice
    raise VmisClassificationError(
        f"Invalid choice index {choice} (expected 0-{num_children})"
    )


def _structured(model: BaseChatModel, num_children: int):
    return model.with_structured_output(
        _make_choice_model(num_children + 1), include_raw=True
    )


def _messages(user_prompt: str) -> list:
    return [SystemMessage(content=VMIS_SYSTEM_PROMPT), HumanMessage(content=user_prompt)]


def _parsed_choice(result: Any) -> int:
    parsed = result["parsed"]
    if not isinstance(parsed, BaseModel):
        raise VmisClassificationError(f"Model returned no parsable choice: {result.get('raw')}")
    return parsed.choice  # type: ignore[attr-defined]


def _with_vmis(
    item: TreatmentItem,
    system: dict[str, Any] | None = None,
    region: dict[str, Any] | None = None,
    leaf: dict[str, Any] | None = None,
) -> TreatmentItem:
    return item.model_copy(update={
        "vmis_system": system["section"] if system else None,
        "vmis_region": region["region"] if region else None,
        "vmis_procedure": leaf["procedure"] if leaf else None,
        "vmis_category": VmisCategory(leaf["category"]) if leaf else None,
    })


def _classify_single_item(
    model: BaseChatModel,
    item: TreatmentItem,
    tree: list[dict[str, Any]],
) -> tuple[TreatmentItem, TokenCount]:
    """Walk system → region → procedure for one surgical treatment."""
    total_tc = TokenCount()
    if not item.treatment.strip():
        return item, total_tc

    source = "\n".join(item.source)
    node: dict[str, Any] = {"children": tree, "section": "root"}
    path: list[dict[str, Any]] = []

    for _ in range(3):
        children = node["children"]
        prompt = _build_level_prompt(item.treatment, source, _node_label(node), children)
        try:
            result = _structured(model, len(children)).invoke(_messages(prompt))
        except Exception as e:
            raise VmisClassificationError(
                f"LLM call failed at '{_node_label(node)}': {e}"
            ) from e
        total_tc = total_tc + extract_token_counts(result)

        idx = _resolve_choice(_parsed_choice(result), len(children))
        if idx is None:
            return _with_vmis(item), total_tc
        node = children[idx]
        path.append(node)

    return _with_vmis(item, *path), total_tc


async def _aclassify_single_item(
    model: BaseChatModel,
    item: TreatmentItem,
    tree: list[dict[str, Any]],
) -> tuple[TreatmentItem, TokenCount]:
    """Async twin of :func:`_classify_single_item`."""
    total_tc = TokenCount()
    if not item.treatment.strip():
        return item, total_tc

    source = "\n".join(item.source)
    node: dict[str, Any] = {"children": tree, "section": "root"}
    path: list[dict[str, Any]] = []

    for _ in range(3):
        children = node["children"]
        prompt = _build_level_prompt(item.treatment, source, _node_label(node), children)
        try:
            result = await _structured(model, len(children)).ainvoke(_messages(prompt))
        except Exception as e:
            raise VmisClassificationError(
                f"LLM call failed at '{_node_label(node)}': {e}"
            ) from e
        total_tc = total_tc + extract_token_counts(result)

        idx = _resolve_choice(_parsed_choice(result), len(children))
        if idx is None:
            return _with_vmis(item), total_tc
        node = children[idx]
        path.append(node)

    return _with_vmis(item, *path), total_tc


def classify_treatments(
    model: BaseChatModel,
    items: list[TreatmentItem],
    vmis_tree: list[dict[str, Any]],
) -> tuple[list[TreatmentItem], TokenCount]:
    """Map each surgical treatment to its VMIS entry and complexity tier.

    Only items flagged ``is_surgical`` are classified. Non-surgical items, and
    surgery that is not in the schedule, keep ``vmis_*`` as ``None``.

    Raises
    ------
    VmisClassificationError
        If any LLM call fails or returns an invalid choice.
    """
    classified: list[TreatmentItem] = []
    total_tc = TokenCount()
    for item in items:
        if item.is_surgical:
            item, tc = _classify_single_item(model, item, vmis_tree)
            total_tc = total_tc + tc
        classified.append(item)
    return classified, total_tc


async def aclassify_treatments(
    model: BaseChatModel,
    items: list[TreatmentItem],
    vmis_tree: list[dict[str, Any]],
    max_concurrency: int = 5,
) -> tuple[list[TreatmentItem], TokenCount]:
    """Async twin of :func:`classify_treatments`; items run concurrently."""
    if not any(item.is_surgical for item in items):
        return items, TokenCount()

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _classify_one(item: TreatmentItem) -> tuple[TreatmentItem, TokenCount]:
        if not item.is_surgical:
            return item, TokenCount()
        async with semaphore:
            return await _aclassify_single_item(model, item, vmis_tree)

    results = await asyncio.gather(*(_classify_one(item) for item in items))

    total_tc = TokenCount()
    for _, tc in results:
        total_tc = total_tc + tc
    return [item for item, _ in results], total_tc
