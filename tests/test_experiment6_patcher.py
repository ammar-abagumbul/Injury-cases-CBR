"""Tests for the Experiment 6 partial-edit agent (patcher)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hklii_psla.experiments.experiment6.audit import Config
from hklii_psla.experiments.experiment6.patcher import (
    add_personality_change_loss,
    apply_patch,
    choose_rewalk_start,
    fix_citation,
    propose_patch,
    reclassify_coarse_injuries,
)
from hklii_psla.extractor.icd_classifier import build_icd_index, load_icd_tree

REPO_ROOT = Path(__file__).resolve().parents[1]
ICD_TREE = REPO_ROOT / "data/ICD-11.json"


# ---------------------------------------------------------------------------
# Deterministic ops
# ---------------------------------------------------------------------------


def test_fix_citation_rewrites_and_fills_action():
    case = {"metadata": {"neutral_citation": "[1989] HKCFI 191; HCAJ 209/1984", "action_number": ""}}
    changes = fix_citation(case, "##  FOO v. BAR [1989] HKCFI 191; HCAJ 209/1984\n")
    paths = {c.path for c in changes}
    assert case["metadata"]["neutral_citation"] == "[1989] HKCFI 191"
    assert case["metadata"]["action_number"] == "HCAJ 209/1984"
    assert paths == {"metadata.neutral_citation", "metadata.action_number"}


def test_fix_citation_noop_when_already_canonical():
    case = {"metadata": {"neutral_citation": "[1989] HKCFI 191", "action_number": "HCAJ 209/1984"}}
    assert fix_citation(case, "##  FOO v. BAR [1989] HKCFI 191; HCAJ 209/1984\n") == []


def test_fix_citation_recovers_from_extraction_without_source():
    # An unparseable / absent source must not prevent repair.
    case = {"metadata": {"neutral_citation": "[2001] HKDC 52 (stray trailing)", "action_number": "x"}}
    changes = fix_citation(case, "")
    assert case["metadata"]["neutral_citation"] == "[2001] HKDC 52"
    assert {c.path for c in changes} == {"metadata.neutral_citation"}
    assert changes[0].note == "recovered from extracted field"


def test_fix_citation_falls_back_to_source_for_missing_year():
    case = {"metadata": {"neutral_citation": "HKDC 394", "action_number": "DCPI 1981/2006"}}
    changes = fix_citation(case, "##  TSE v. ATLANTIC [2007] HKDC 394; DCPI 1981/2006\n")
    assert case["metadata"]["neutral_citation"] == "[2007] HKDC 394"
    assert {c.path for c in changes} == {"metadata.neutral_citation"}
    assert changes[0].note == "recovered from source header"


def test_fix_citation_ignores_law_report_series():
    case = {"metadata": {"neutral_citation": "[1988] HKC 795", "action_number": "HCA 303/1988"}}
    fix_citation(case, "##  TSANG v. FERRY [1988] HKCFI 461; [1988] HKC 795; HCA 303/1988\n")
    assert case["metadata"]["neutral_citation"] == "[1988] HKCFI 461"


def test_fix_citation_conflict_is_left_untouched():
    case = {"metadata": {"neutral_citation": "[2008] HKCFI 999", "action_number": "HCPI 1/2008"}}
    changes = fix_citation(case, "##  FOO v. BAR [2008] HKCFI 460; HCPI 1/2008\n")
    assert changes == []
    assert case["metadata"]["neutral_citation"] == "[2008] HKCFI 999"


def _psych_case() -> dict[str, Any]:
    return {
        "metadata": {"neutral_citation": "[2011] HKCFI 500"},
        "injuries": {
            "injuries": [
                {
                    "injury_id": "inj_001",
                    "description": "Post-traumatic stress disorder",
                    "injury_type": "Permanent injury",
                    "body_part": "psychiatric",
                    "laterality": None,
                    "source": ["He developed post-traumatic stress disorder."],
                    "icd_code": None,
                    "icd_description": None,
                }
            ],
            "manifestations": [],
        },
        "losses": {"losses": []},
    }


def test_add_personality_change_loss_uses_source_verbatim():
    config = Config.load()
    case = _psych_case()
    changes = add_personality_change_loss(case, config)
    assert len(changes) == 1
    loss = case["losses"]["losses"][0]
    assert loss["category"] == "Personality change"
    assert loss["description"] == "He developed post-traumatic stress disorder."
    assert loss["caused_by_injury_id"] == "inj_001"


def test_add_personality_change_loss_dedupes():
    config = Config.load()
    case = _psych_case()
    case["losses"]["losses"] = [
        {"category": "Personality change", "description": "x", "caused_by_injury_id": "inj_001"}
    ]
    assert add_personality_change_loss(case, config) == []
    assert len(case["losses"]["losses"]) == 1


def test_add_personality_change_loss_noop_without_psych():
    config = Config.load()
    case = _psych_case()
    case["injuries"]["injuries"][0]["description"] = "Fracture of left radius"
    case["injuries"]["injuries"][0]["source"] = ["Fracture of left radius."]
    assert add_personality_change_loss(case, config) == []


def test_add_personality_change_loss_manifestation_only():
    config = Config.load()
    case = _psych_case()
    case["injuries"]["injuries"] = []
    case["injuries"]["manifestations"] = [
        {
            "description": "Depressed mood and anxiety",
            "manifestation_type": "symptom",
            "caused_by_injury_id": "inj_009",
        }
    ]
    changes = add_personality_change_loss(case, config)
    assert len(changes) == 1
    loss = case["losses"]["losses"][0]
    assert loss["caused_by_injury_id"] == "inj_009"
    assert "Depressed mood and anxiety" in loss["description"]


# ---------------------------------------------------------------------------
# Coarse re-walk
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Minimal stand-in for a LangChain chat model returning scripted choices."""

    def __init__(self, choices: list[int]) -> None:
        self.choices = list(choices)
        self.pos = 0

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> _ScriptedModel:
        return self

    def _result(self) -> dict[str, Any]:
        choice = self.choices[self.pos]
        self.pos += 1
        return {
            "parsed": SimpleNamespace(choice=choice),
            "raw": SimpleNamespace(usage_metadata={}),
        }

    def invoke(self, messages: Any) -> dict[str, Any]:
        return self._result()

    async def ainvoke(self, messages: Any) -> dict[str, Any]:
        return self._result()


def test_choose_rewalk_start_for_coarse_leaf_goes_to_parent():
    by_code, parent_of = build_icd_index(load_icd_tree(ICD_TREE))
    config = Config.load()
    start = choose_rewalk_start("NB52.4", by_code, parent_of, set(config.coarse["nodes"]))
    assert start is not None
    assert start["code"] == "NB52"


def test_choose_rewalk_start_for_stopped_early_stays_at_node():
    by_code, parent_of = build_icd_index(load_icd_tree(ICD_TREE))
    config = Config.load()
    # NA07.0 (Concussion) has children and is not itself coarse
    start = choose_rewalk_start("NA07.0", by_code, parent_of, set(config.coarse["nodes"]))
    assert start is not None
    assert start["code"] == "NA07.0"


def test_reclassify_coarse_injuries_picks_specific_code():
    by_code, parent_of = build_icd_index(load_icd_tree(ICD_TREE))
    config = Config.load()
    case = {
        "injuries": {
            "injuries": [
                {
                    "injury_id": "inj_001",
                    "description": "Fracture of the lumbar vertebra",
                    "injury_type": "Permanent injury",
                    "body_part": "lumbar spine",
                    "laterality": None,
                    "source": ["There was a fracture of the lumbar vertebra."],
                    "icd_code": "NB52.4",
                    "icd_description": "Multiple fractures of lumbar spine or pelvis",
                }
            ]
        }
    }
    model = _ScriptedModel([0])  # first allowed child of NB52 is NB52.0
    changes = asyncio.run(
        reclassify_coarse_injuries(
            case, model, by_code, parent_of, set(config.coarse["nodes"])
        )
    )
    assert case["injuries"]["injuries"][0]["icd_code"] == "NB52.0"
    assert {c.path for c in changes} == {
        "injuries.injuries[0].icd_code",
        "injuries.injuries[0].icd_description",
    }


def test_reclassify_skips_non_qualifying_injuries():
    by_code, parent_of = build_icd_index(load_icd_tree(ICD_TREE))
    config = Config.load()
    case = {
        "injuries": {
            "injuries": [
                {
                    "injury_id": "inj_001",
                    "description": "Fracture of scaphoid bone",
                    "injury_type": "Permanent injury",
                    "body_part": "left wrist",
                    "laterality": "left",
                    "source": [],
                    "icd_code": "NC53.0",
                    "icd_description": "Fracture of scaphoid bone of hand",
                }
            ]
        }
    }
    model = _ScriptedModel([0])
    changes = asyncio.run(
        reclassify_coarse_injuries(
            case, model, by_code, parent_of, set(config.coarse["nodes"])
        )
    )
    assert changes == []


# ---------------------------------------------------------------------------
# Propose / apply
# ---------------------------------------------------------------------------


def test_propose_and_apply_patch(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    case_path = cases_dir / "case.json"
    case = _psych_case()
    case["metadata"]["neutral_citation"] = "[2011] HKCFI 500; HCPI 285/2008"
    case["metadata"]["action_number"] = ""
    case["plaintiff"] = {}
    case["treatment"] = {}
    case["psla"] = {}
    case_path.write_text(json.dumps(case), encoding="utf-8")

    source_text = "##  X v. Y [2011] HKCFI 500; HCPI 285/2008\n"
    patch = propose_patch(
        "case.json", case_path, source_text, ["citation", "psych"], Config.load()
    )
    assert patch.valid, patch.error
    assert not patch.is_empty

    # dry run leaves the file untouched
    assert json.loads(case_path.read_text())["metadata"]["neutral_citation"].endswith(
        "HCPI 285/2008"
    )

    backup_dir = tmp_path / "backup"
    assert apply_patch(patch, backup_dir)
    written = json.loads(case_path.read_text())
    assert written["metadata"]["neutral_citation"] == "[2011] HKCFI 500"
    assert written["losses"]["losses"][0]["category"] == "Personality change"
    assert (backup_dir / "case.json").exists()


# ---------------------------------------------------------------------------
# Root navigation regression (classifier refactor)
# ---------------------------------------------------------------------------


def test_aclassify_injuries_navigates_from_root():
    from hklii_psla.extractor.icd_classifier import aclassify_injuries
    from hklii_psla.schemas import Injury

    sections = [
        {
            "section": "Section A",
            "children": [
                {
                    "code": "AA",
                    "description": "Category A",
                    "children": [
                        {"code": "AA.0", "description": "Leaf 0"},
                        {"code": "AA.1", "description": "Leaf 1"},
                    ],
                },
                {"code": "AB", "description": "Category B", "children": []},
            ],
        }
    ]
    injury = Injury(
        injury_id="inj_001",
        description="something",
        injury_type="Temporary injury",
        source=["text"],
    )
    model = _ScriptedModel([0, 0, 0])
    classified, _ = asyncio.run(aclassify_injuries(model, [injury], sections))
    assert classified[0].icd_code == "AA.0"


def test_aclassify_injuries_returns_unchanged_when_root_stops():
    from hklii_psla.extractor.icd_classifier import aclassify_injuries
    from hklii_psla.schemas import Injury

    sections = [{"section": "Section A", "children": [{"code": "AA", "description": "A"}]}]
    injury = Injury(
        injury_id="inj_001",
        description="something",
        injury_type="Temporary injury",
        source=["text"],
    )
    # index 1 = "Other" at root, i.e. stop before choosing a section
    model = _ScriptedModel([1])
    classified, _ = asyncio.run(aclassify_injuries(model, [injury], sections))
    assert classified[0].icd_code is None
