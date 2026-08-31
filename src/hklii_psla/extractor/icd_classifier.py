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
"""

from __future__ import annotations

import json
from pathlib import Path
from pydoc import source_synopsis
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from hklii_psla.schemas import Injury
from hklii_psla.extractor.base import TokenCount
from hklii_psla.extractor.utils import extract_token_counts



# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ICDClassificationError(Exception):
    """Raised when ICD-11 classification fails for any injury."""


# ---------------------------------------------------------------------------
# Tree loading
# ---------------------------------------------------------------------------

def load_icd_tree(path: Path) -> list[dict]:
    """Load ICD-11 JSON taxonomy tree into memory.

    The file ``data/ICD-11.json`` contains 9 top-level section objects, each
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


# ---------------------------------------------------------------------------
# Dynamic Pydantic choice model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

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


def _classify_level(
    model: BaseChatModel,
    injury_description: str,
    source: str,
    current_node: dict[str, Any],
) -> tuple[int, TokenCount]:
    """Ask the LLM to pick the best child at the current tree level.

    Parameters
    ----------
    model : BaseChatModel
        The LLM to query.
    injury_description : str
        The injury text from the judgment.
    current_node : dict
        The current ICD-11 node (must have a ``children`` key).

    Returns
    -------
    tuple[int, TokenCount]
        Index into ``current_node["children"]`` (or -1 for "Other",
        or -2 for "No further detail"), and the token counts for this call.

    Raises
    ------
    ICDClassificationError
        If the LLM call fails or returns an invalid choice.
    """
    children: list[dict] = current_node.get("children", [])
    node_desc = current_node.get("description") or current_node.get("section", "root")

    if not children:
        return (-2, TokenCount(input_tokens=0, output_tokens=0))

    # Build option list
    options: list[str] = []
    for child in children:
        code = child.get("code", "?")
        # Non-section nodes have "description"; top-level section nodes use "section"
        desc = child.get("description") or child.get("section", "?")
        options.append(f"{code}: {desc}")

    other_index = len(options)
    no_detail_index = len(options) + 1

    options.append("Other — none of the above match")
    options.append("No further detail — stop at current level")

    # Build human prompt
    option_lines = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    user_prompt = (
        f"Injury: {injury_description}\n\n"
        f"Source from document: {source}\n\n"
        f"Current level: {node_desc}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Choose the SINGLE best match. Respond with only the option index."
    )

    # Create structured model for this specific level
    choice_model_cls = _make_choice_model(len(options))
    structured_model = model.with_structured_output(choice_model_cls, include_raw=True)

    try:
        result = structured_model.invoke([
            SystemMessage(content=ICD_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        assert(isinstance(result, dict))
    except Exception as e:
        raise ICDClassificationError(
            f"LLM call failed at node '{node_desc}': {e}"
        ) from e

    choice: int = result["parsed"].choice  # type: ignore[arg-type]
    tc = extract_token_counts(result)

    if choice == other_index:
        return -1, tc  # Other
    elif choice == no_detail_index:
        return -2, tc  # No further detail
    elif 0 <= choice < len(children):
        return choice, tc
    else:
        raise ICDClassificationError(
            f"Invalid choice index {choice} (expected 0-{len(options) - 1})"
        )


def _classify_single_injury(
    model: BaseChatModel,
    injury: Injury,
    sections: list[dict[str, Any]],
) -> tuple[Injury, TokenCount]:
    """Classify a single injury by navigating the ICD-11 tree.

    1. Select the best top-level section from the 9 sections.
    2. Recursively drill down through children until a leaf is reached
       or the LLM chooses "Other" / "No further detail".

    Returns
    -------
    tuple[Injury, TokenCount]
        The classified injury and the aggregate token counts for all
        LLM calls made during classification.
    """
    desc = injury.description
    source = injury.source
    if not desc.strip():
        # Nothing to classify — keep existing (None) fields
        return injury, TokenCount()

    total_tc = TokenCount()

    # Phase 1: pick top-level section
    section_idx, tc = _classify_level(
        model, desc, source, {"children": sections, "section": "root"}
    )
    total_tc = total_tc + tc

    if section_idx == -1 or section_idx == -2:
        # Shouldn't happen at root level, but handle gracefully by
        # returning the injury unchanged.
        return injury, total_tc

    current_node: dict[str, Any] = sections[section_idx]

    # Phase 2: recurse down
    while True:
        children = current_node.get("children", [])
        if not children:
            # Leaf node — use it
            break

        child_idx, tc = _classify_level(model, desc, source, current_node)
        total_tc = total_tc + tc

        if child_idx == -1:
            # Other — stop at current node
            break
        elif child_idx == -2:
            # No further detail — stop at current node
            break

        current_node = children[child_idx]

    # Populate ICD fields on a copy.
    # Top-level sections use key "section" instead of "description", and
    # have no ICD code (they are anatomical groupings, not codes).
    code: str | None = current_node.get("code")
    desc_icd: str = current_node.get("description") or current_node.get("section", "")

    classified = Injury(
        injury_id=injury.injury_id,
        description=injury.description,
        injury_type=injury.injury_type,
        body_part=injury.body_part,
        laterality=injury.laterality,
        source=injury.source,
        icd_code=code,
        icd_description=desc_icd,
    )
    return classified, total_tc


def classify_injuries(
    model: BaseChatModel,
    injuries: list[Injury],
    icd_tree: list[dict[str, Any]],
) -> tuple[list[Injury], TokenCount]:
    """For each injury, navigate the ICD-11 tree to find the best matching code.

    Parameters
    ----------
    model : BaseChatModel
        The LLM to use for classification decisions.
    injuries : list[Injury]
        Injuries extracted during Stage 1 (without ICD codes).
    icd_tree : list[dict]
        The parsed ICD-11 taxonomy tree (from ``load_icd_tree()``).

    Returns
    -------
    tuple[list[Injury], TokenCount]
        New injury list with ``icd_code`` and ``icd_description`` populated,
        and the aggregate token counts for all classification calls.

    Raises
    ------
    ICDClassificationError
        If any injury fails classification.
    """
    if not injuries:
        return injuries, TokenCount()

    # icd_tree is a list of top-level section dicts
    sections = icd_tree

    classified: list[Injury] = []
    total_tc = TokenCount()
    for injury in injuries:
        result, tc = _classify_single_injury(model, injury, sections)
        classified.append(result)
        total_tc = total_tc + tc

    return classified, total_tc
