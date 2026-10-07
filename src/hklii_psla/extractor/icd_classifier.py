"""
ICD-11 interactive classification for extracted injuries.

Navigates the ICD-11 taxonomy tree top-down with the LLM, selecting the
most appropriate code for each injury description.

Example flow for "fracture of the left tibia":
  1. Select section: "Injuries to the knee or lower leg"
  2. Select category (children of that section):
     "NC92 - Fracture of lower leg, including ankle"
  3. Select subcategory (children of NC92):
     "NC92.2 - Fracture of shaft of tibia"
  4. That node has no children → we stop, returning NC92.2.

The tree walk also supports *re-classification*: starting from an arbitrary
node, forbidding a set of (coarse) codes, and forcing at least one more level
of specificity. This powers the Experiment 6 corpus clean-up.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..schemas import Injury
from .base import TokenCount
from .utils import extract_token_counts


class ICDClassificationError(Exception):
    """Raised when ICD-11 classification fails for any injury."""


def load_icd_tree(path: Path) -> list[dict[str, Any]]:
    """Load ICD-11 JSON taxonomy tree into memory.

    The file ``data/ICD-11.json`` contains top-level section objects, each
    with a nested ``children`` hierarchy of categories/subcategories/leaves.
    Returns the parsed list directly.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ICDClassificationError(
            f"ICD-11 JSON must be a list, got {type(data).__name__}"
        )
    return data


def build_icd_index(
    tree: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None]]:
    """Index an ICD tree as ``(by_code, parent_of)``.

    Section nodes have no ``code``; their children inherit the section as
    parent. ``parent_of`` maps a code to its nearest coded ancestor.
    """
    by_code: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str | None] = {}

    def walk(node: dict[str, Any], parent: str | None) -> None:
        code = node.get("code")
        if code:
            by_code[code] = node
            parent_of[code] = parent
        for child in node.get("children", []):
            walk(child, code or parent)

    for section in tree:
        walk(section, None)
    return by_code, parent_of


def _make_choice_model(num_options: int) -> type[BaseModel]:
    """Create a Pydantic model with a single ``choice`` int field.

    The field is constrained to ``0 <= choice < num_options``. The LLM
    returns JSON validated by the schema, so invalid indices are caught
    by Pydantic validation.
    """

    class ChoiceModel(BaseModel):
        choice: int = Field(
            ge=0,
            lt=num_options,
            description=f"Selected option index (0-{num_options - 1})",
        )

    return ChoiceModel


# System prompt for the ICD-11 classification agent
ICD_SYSTEM_PROMPT = (
    "You are a medical coding assistant specialising in ICD-11 injury codes. "
    "Your task is to match injury descriptions from legal judgments to the most "
    "appropriate ICD-11 category.\n\n"
    "Rules:\n"
    "- Read the injury description carefully.\n"
    "- Select the SINGLE option that best matches the injury.\n"
    "- If none of the specific options match, choose 'Other'.\n"
    "- If the current level is already specific enough and further detail is "
    "not needed, choose 'No further detail'.\n"
    "- Respond ONLY with a valid option index."
)

# Marker returned by the level classifier when the model chose to stop at the
# current node (via "Other" or "No further detail").
_STOP = object()


def _node_label(node: dict[str, Any]) -> str:
    return node.get("description") or node.get("section", "?")


def _build_level_prompt(
    injury_description: str,
    source: str,
    current_node: dict[str, Any],
    allowed_children: list[dict[str, Any]],
    allow_stop: bool,
) -> tuple[str, int, int | None, str]:
    """Build the option list and user prompt for one tree level.

    Only ``allowed_children`` are presented. Returns
    ``(user_prompt, other_index, no_detail_index, node_desc)`` where
    ``no_detail_index`` is ``None`` when stopping is disallowed.
    """
    node_desc = current_node.get("description") or current_node.get("section", "root")

    options: list[str] = []
    for child in allowed_children:
        code = child.get("code", "?")
        options.append(f"{code}: {_node_label(child)}")

    other_index = len(options)
    options.append("Other — none of the above match")

    no_detail_index: int | None = None
    if allow_stop:
        no_detail_index = len(options)
        options.append("No further detail — stop at current level")

    option_lines = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    user_prompt = (
        f"Injury: {injury_description}\n\n"
        f"Source from document: {source}\n\n"
        f"Current level: {node_desc}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Choose the SINGLE best match. Respond with only the option index."
    )
    return user_prompt, other_index, no_detail_index, node_desc


def _resolve_choice(
    choice: int,
    allowed_children: list[dict[str, Any]],
    other_index: int,
    no_detail_index: int | None,
) -> dict[str, Any] | object:
    """Map a raw model choice to a child node or the :data:`_STOP` marker."""
    if choice == other_index:
        return _STOP
    if no_detail_index is not None and choice == no_detail_index:
        return _STOP
    if 0 <= choice < len(allowed_children):
        return allowed_children[choice]
    raise ICDClassificationError(
        f"Invalid choice index {choice} (expected 0-{other_index})"
    )


def _allowed_children(
    node: dict[str, Any], forbidden: set[str]
) -> list[dict[str, Any]]:
    return [
        child
        for child in node.get("children", [])
        if child.get("code") not in forbidden
    ]


def _classify_level(
    model: BaseChatModel,
    injury_description: str,
    source: str,
    current_node: dict[str, Any],
    forbidden: set[str],
    allow_stop: bool,
) -> tuple[dict[str, Any] | object, TokenCount]:
    """Ask the LLM to pick the best allowed child at the current tree level."""
    allowed = _allowed_children(current_node, forbidden)
    if not allowed:
        return _STOP, TokenCount(input_tokens=0, output_tokens=0)

    user_prompt, other_index, no_detail_index, node_desc = _build_level_prompt(
        injury_description, source, current_node, allowed, allow_stop
    )

    num_options = len(allowed) + 1 + (1 if allow_stop else 0)
    choice_model_cls = _make_choice_model(num_options)
    structured_model = model.with_structured_output(choice_model_cls, include_raw=True)

    try:
        result = structured_model.invoke([
            SystemMessage(content=ICD_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        assert isinstance(result, dict)
    except Exception as e:
        raise ICDClassificationError(
            f"LLM call failed at node '{node_desc}': {e}"
        ) from e

    choice: int = result["parsed"].choice  # type: ignore[arg-type]
    tc = extract_token_counts(result)
    return _resolve_choice(choice, allowed, other_index, no_detail_index), tc


async def _aclassify_level(
    model: BaseChatModel,
    injury_description: str,
    source: str,
    current_node: dict[str, Any],
    forbidden: set[str],
    allow_stop: bool,
) -> tuple[dict[str, Any] | object, TokenCount]:
    """Async twin of :func:`_classify_level`."""
    allowed = _allowed_children(current_node, forbidden)
    if not allowed:
        return _STOP, TokenCount(input_tokens=0, output_tokens=0)

    user_prompt, other_index, no_detail_index, node_desc = _build_level_prompt(
        injury_description, source, current_node, allowed, allow_stop
    )

    num_options = len(allowed) + 1 + (1 if allow_stop else 0)
    choice_model_cls = _make_choice_model(num_options)
    structured_model = model.with_structured_output(choice_model_cls, include_raw=True)

    try:
        result = await structured_model.ainvoke([
            SystemMessage(content=ICD_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        assert isinstance(result, dict)
    except Exception as e:
        raise ICDClassificationError(
            f"LLM call failed at node '{node_desc}': {e}"
        ) from e

    choice: int = result["parsed"].choice  # type: ignore[arg-type]
    tc = extract_token_counts(result)
    return _resolve_choice(choice, allowed, other_index, no_detail_index), tc


def _walk_from(
    model: BaseChatModel,
    desc: str,
    source: str,
    start_node: dict[str, Any],
    forbidden: set[str],
    force_deep: bool,
) -> tuple[dict[str, Any], TokenCount]:
    """Synchronously walk down from ``start_node``.

    ``force_deep`` disallows the "No further detail" option for the first
    decision so at least one more level is traversed.
    """
    total_tc = TokenCount()
    current = start_node
    allow_stop = not force_deep

    while _allowed_children(current, forbidden):
        chosen, tc = _classify_level(model, desc, source, current, forbidden, allow_stop)
        total_tc = total_tc + tc
        if chosen is _STOP:
            break
        current = chosen  # type: ignore[assignment]
        allow_stop = True

    return current, total_tc


async def _awalk_from(
    model: BaseChatModel,
    desc: str,
    source: str,
    start_node: dict[str, Any],
    forbidden: set[str],
    force_deep: bool,
) -> tuple[dict[str, Any], TokenCount]:
    """Async twin of :func:`_walk_from`."""
    total_tc = TokenCount()
    current = start_node
    allow_stop = not force_deep

    while _allowed_children(current, forbidden):
        chosen, tc = await _aclassify_level(
            model, desc, source, current, forbidden, allow_stop
        )
        total_tc = total_tc + tc
        if chosen is _STOP:
            break
        current = chosen  # type: ignore[assignment]
        allow_stop = True

    return current, total_tc


def _copy_with_icd(injury: Injury, code: str | None, desc_icd: str) -> Injury:
    return Injury(
        injury_id=injury.injury_id,
        description=injury.description,
        injury_type=injury.injury_type,
        body_part=injury.body_part,
        laterality=injury.laterality,
        source=injury.source,
        icd_code=code,
        icd_description=desc_icd,
    )


def _classify_single_injury(
    model: BaseChatModel,
    injury: Injury,
    sections: list[dict[str, Any]],
) -> tuple[Injury, TokenCount]:
    """Classify a single injury by navigating the ICD-11 tree from the root."""
    desc = injury.description
    src_concat = "\n".join(injury.source)
    if not desc.strip():
        return injury, TokenCount()

    root: dict[str, Any] = {"children": sections, "section": "root"}
    current, total_tc = _walk_from(model, desc, src_concat, root, set(), False)
    if current is root or current.get("code") is None:
        return injury, total_tc

    return (
        _copy_with_icd(
            injury,
            current.get("code"),
            current.get("description") or current.get("section", ""),
        ),
        total_tc,
    )


async def _aclassify_single_injury(
    model: BaseChatModel,
    injury: Injury,
    sections: list[dict[str, Any]],
) -> tuple[Injury, TokenCount]:
    """Async twin of :func:`_classify_single_injury`."""
    desc = injury.description
    src_cat = "\n".join(injury.source)
    if not desc.strip():
        return injury, TokenCount()

    root: dict[str, Any] = {"children": sections, "section": "root"}
    current, total_tc = await _awalk_from(model, desc, src_cat, root, set(), False)
    if current is root or current.get("code") is None:
        return injury, total_tc

    return (
        _copy_with_icd(
            injury,
            current.get("code"),
            current.get("description") or current.get("section", ""),
        ),
        total_tc,
    )


async def areclassify_injury(
    model: BaseChatModel,
    injury: Injury,
    start_node: dict[str, Any],
    forbidden_codes: set[str] | None = None,
    force_deep: bool = True,
) -> tuple[Injury, TokenCount]:
    """Re-classify one injury starting from ``start_node``.

    Used by the Experiment 6 partial-edit agent to replace a coarse code with a
    more specific one. ``forbidden_codes`` are removed from every option list
    (typically the coarse-node denylist) and ``force_deep`` forces at least one
    further level of specificity.
    """
    desc = injury.description
    src = "\n".join(injury.source)
    if not desc.strip():
        return injury, TokenCount()

    current, total_tc = await _awalk_from(
        model, desc, src, start_node, set(forbidden_codes or ()), force_deep
    )
    if current.get("code") is None:
        return injury, total_tc

    return (
        _copy_with_icd(
            injury,
            current.get("code"),
            current.get("description") or current.get("section", ""),
        ),
        total_tc,
    )


def classify_injuries(
    model: BaseChatModel,
    injuries: list[Injury],
    icd_tree: list[dict[str, Any]],
) -> tuple[list[Injury], TokenCount]:
    """For each injury, navigate the ICD-11 tree to find the best matching code."""
    if not injuries:
        return injuries, TokenCount()

    sections = icd_tree
    classified: list[Injury] = []
    total_tc = TokenCount()
    for injury in injuries:
        result, tc = _classify_single_injury(model, injury, sections)
        classified.append(result)
        total_tc = total_tc + tc

    return classified, total_tc


async def aclassify_injuries(
    model: BaseChatModel,
    injuries: list[Injury],
    icd_tree: list[dict[str, Any]],
    max_concurrency: int = 5,
) -> tuple[list[Injury], TokenCount]:
    """Async twin of :func:`classify_injuries`.

    Injuries are independent, so they are classified concurrently behind a
    semaphore. Results are returned in the same order as ``injuries`` and
    token counts are aggregated across all classification calls.
    """
    if not injuries:
        return injuries, TokenCount()

    sections = icd_tree
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _classify_one(injury: Injury) -> tuple[Injury, TokenCount]:
        async with semaphore:
            return await _aclassify_single_injury(model, injury, sections)

    results = await asyncio.gather(*(_classify_one(inj) for inj in injuries))

    classified: list[Injury] = []
    total_tc = TokenCount()
    for result, tc in results:
        classified.append(result)
        total_tc = total_tc + tc

    return classified, total_tc
