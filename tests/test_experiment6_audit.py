"""Tests for Experiment 6 QC configs and the deterministic audit tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hklii_psla.experiments.experiment6 import audit as audit_mod
from hklii_psla.experiments.experiment6.audit import (
    DEFAULT_COURTS,
    Config,
    CorpusAuditor,
    _make_matcher,
    make_citation_recovery_re,
    parse_citation_from_source,
    recover_citation_from_extracted,
    resolve_citation,
    source_header_block,
)

CONFIG_DIR = Path(audit_mod.__file__).parent
REPO_ROOT = CONFIG_DIR.parents[3]
ICD_TREE = REPO_ROOT / "data/ICD-11.json"


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------


def test_configs_load():
    config = Config.load()
    assert config.coarse["nodes"]
    assert config.red_flags["fields"]
    assert config.psych["keywords"]
    assert config.qc["null_ratio"]["scalar_fields"]


def test_all_coarse_nodes_exist_in_icd_tree():
    config = Config.load()
    tree = json.loads(ICD_TREE.read_text(encoding="utf-8"))
    codes: set[str] = set()

    def walk(node: dict) -> None:
        if node.get("code"):
            codes.add(node["code"])
        for child in node.get("children", []):
            walk(child)

    for section in tree:
        walk(section)

    missing = set(config.coarse["nodes"]) - codes
    assert not missing, f"coarse node codes not in ICD tree: {sorted(missing)}"


def test_psych_target_category_is_canonical():
    from hklii_psla.schemas import ALL_LOSS_CATEGORIES

    config = Config.load()
    assert config.psych["target_loss_category"] in ALL_LOSS_CATEGORIES


# ---------------------------------------------------------------------------
# Citation recovery
# ---------------------------------------------------------------------------


def test_source_header_block_reflows_wrapped_heading():
    text = (
        "* * *\n\n"
        "##  CHEUNG YUK CHUN AND OTHERS v. MITSUI CONSTRUCION CO LTD AND ANOTHER [1984]\n"
        "HKCFI 403; HCA 12597/1982 (8 November 1984)\n\n"
        "HCA012597/1982\n"
    )
    block = source_header_block(text)
    assert "[1984] HKCFI 403" in block
    assert "HCA 12597/1982" in block


def test_parse_citation_from_source_normalises_action():
    text = "##  FOO v. BAR [2015] HKDC\n256; DCPI 2013/2014 (24 March 2015)\n"
    neutral, action = parse_citation_from_source(text)
    assert neutral == "[2015] HKDC 256"
    assert action == "DCPI 2013/2014"


def test_parse_citation_strips_leading_zeros():
    text = "##  FOO v. BAR [1984] HKCFI 403; HCA012597/1982\n"
    _, action = parse_citation_from_source(text)
    assert action == "HCA 12597/1982"


def test_parse_citation_missing_header_returns_none():
    neutral, action = parse_citation_from_source("no heading here\n")
    assert neutral is None and action is None


# --- extraction-first recovery ----------------------------------------------


CITATION_RE = make_citation_recovery_re(list(DEFAULT_COURTS))


def test_recover_citation_strips_suffix_and_prefix():
    assert (
        recover_citation_from_extracted(
            "[1989] HKCFI 191; HCAJ 209/1984 (10 November 1989)", CITATION_RE
        )
        == "[1989] HKCFI 191"
    )
    assert (
        recover_citation_from_extracted(
            "SIT KA YEE v. LAI WAI HO [2001] HKDC 52", CITATION_RE
        )
        == "[2001] HKDC 52"
    )


def test_recover_citation_rejects_law_report_series():
    # HKC is a law-report series, not a neutral citation.
    assert recover_citation_from_extracted("[1988] HKC 795", CITATION_RE) is None
    assert (
        recover_citation_from_extracted("[1997] 3 HKC 655; HKCFI 766", CITATION_RE)
        is None
    )
    assert recover_citation_from_extracted("HKDC 394", CITATION_RE) is None
    assert recover_citation_from_extracted(None, CITATION_RE) is None


def test_resolve_citation_outcomes():
    extra = resolve_citation(
        "[2009] HKCFI 2062; [2009] 3 HKLRD 44; HCAL 11/2007 (2009-03-03)",
        "##  FOO v. BAR [2009] HKCFI 2062; [2009] 3 HKLRD 44; HCAL 11/2007\n",
        CITATION_RE,
    )
    assert extra.outcome == "extra_text"
    assert extra.resolved == "[2009] HKCFI 2062"

    clean = resolve_citation(
        "[2009] HKCFI 2062", "##  FOO v. BAR [2009] HKCFI 2062\n", CITATION_RE
    )
    assert clean.outcome == "clean"
    assert clean.resolved == "[2009] HKCFI 2062"

    from_source = resolve_citation(
        "HKCFI 460; HCPI 894/2006 (28 April 2008)",
        "##  FOO v. BAR [2008] HKCFI 460; HCPI 894/2006\n",
        CITATION_RE,
    )
    assert from_source.outcome == "from_source"
    assert from_source.resolved == "[2008] HKCFI 460"

    conflict = resolve_citation(
        "[2008] HKCFI 999", "##  FOO v. BAR [2008] HKCFI 460\n", CITATION_RE
    )
    assert conflict.outcome == "conflict"
    assert conflict.resolved is None

    unrecoverable = resolve_citation("", "no heading here\n", CITATION_RE)
    assert unrecoverable.outcome == "unrecoverable"


def test_audit_citation_recovers_from_extraction(auditor, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "case.txt").write_text(
        "##  FOO v. BAR [2011] HKCFI 500; HCPI 285/2008\n", encoding="utf-8"
    )
    case_path = _write_case(tmp_path, "case.json", _base_case())
    result = auditor.audit_case("case.txt", "case.json", case_path)
    assert result.citation_outcome == "extra_text"
    assert result.recovered_citation == "[2011] HKCFI 500"
    assert "CITATION_MALFORMED" in result.flags
    assert "CITATION_MISMATCH" in result.flags


def test_audit_citation_from_source_for_law_report(auditor, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "case.txt").write_text(
        "##  TSANG v. FERRY [1988] HKCFI 461; [1988] HKC 795; HCA 303/1988\n",
        encoding="utf-8",
    )
    case = _base_case()
    case["metadata"]["neutral_citation"] = "[1988] HKC 795"
    case["metadata"]["action_number"] = "HCA 303/1988"
    case_path = _write_case(tmp_path, "case.json", case)
    result = auditor.audit_case("case.txt", "case.json", case_path)
    assert result.citation_outcome == "from_source"
    assert result.recovered_citation is None
    assert "CITATION_FROM_SOURCE" in result.flags
    assert "CITATION_MISMATCH" in result.flags


def test_audit_citation_conflict_is_flagged(auditor, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "case.txt").write_text("##  FOO v. BAR [2008] HKCFI 460\n", encoding="utf-8")
    case = _base_case()
    case["metadata"]["neutral_citation"] = "[2008] HKCFI 999"
    case_path = _write_case(tmp_path, "case.json", case)
    result = auditor.audit_case("case.txt", "case.json", case_path)
    assert result.citation_outcome == "conflict"
    assert "CITATION_CONFLICT" in result.flags


# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def test_psych_matcher_excludes_depressed_skull_fracture():
    config = Config.load()
    matcher = _make_matcher(
        config.psych["keywords"], config.psych.get("exclusions", [])
    )
    assert matcher("depressed skull fracture") is None
    assert matcher("Post-traumatic stress disorder (PTSD)") is not None
    assert matcher("major depressive episode") is not None


def test_symptom_bucket_matches_pain_but_not_fracture():
    config = Config.load()
    matcher = _make_matcher(config.qc["null_icd_buckets"]["symptom"]["keywords"])
    assert matcher("severe pain in right eye")
    assert matcher("fracture of the left distal radius") is None


# ---------------------------------------------------------------------------
# End-to-end audit on a synthetic case
# ---------------------------------------------------------------------------


@pytest.fixture()
def auditor(tmp_path: Path) -> CorpusAuditor:
    config = Config.load()
    return CorpusAuditor(
        config=config,
        cases_dir=tmp_path / "cases",
        progress_path=tmp_path / "progress.csv",
        source_dir=tmp_path / "src",
        icd_tree_path=ICD_TREE,
    )


def _write_case(tmp_path: Path, name: str, case: dict) -> Path:
    path = tmp_path / "cases" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case), encoding="utf-8")
    return path


def _base_case() -> dict:
    return {
        "metadata": {
            "neutral_citation": "[2011] HKCFI 500; HCPI 285/2008",
            "action_number": "HCPI 285/2008",
            "case_name": "X v. Y",
            "judgment_date": "2011-07-28",
            "plaintiff_count": 1,
            "has_pre_existing_injuries": False,
        },
        "plaintiff": {"gender": "Male", "age_at_accident": 30},
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
            "manifestations": [
                {
                    "description": "Recurring nightmares",
                    "manifestation_type": "symptom",
                    "caused_by_injury_id": None,
                }
            ],
            "overall_category": "Serious injury",
        },
        "treatment": {},
        "losses": {"losses": []},
        "psla": {"amount": 100000, "comparable_cases": [{"case_name": "A"}]},
        "death": None,
    }


def test_audit_flags_psych_missing_personality_change(auditor, tmp_path):
    case_path = _write_case(tmp_path, "case.json", _base_case())
    result = auditor.audit_case("src.txt", "case.json", case_path)
    assert "PSYCH_DETECTED" in result.flags
    assert "PSYCH_NO_PERSONALITY_CHANGE_LOSS" in result.flags
    assert "NULL_ICD_PSYCH" in result.flags
    assert "CITATION_MALFORMED" in result.flags  # extra text appended
    assert "UNLINKED_MANIFESTATION" in result.flags


def test_audit_does_not_flag_psych_when_loss_present(auditor, tmp_path):
    case = _base_case()
    case["losses"]["losses"] = [
        {
            "category": "Personality change",
            "description": "He developed post-traumatic stress disorder.",
            "caused_by_injury_id": "inj_001",
        }
    ]
    case_path = _write_case(tmp_path, "case.json", case)
    result = auditor.audit_case("src.txt", "case.json", case_path)
    assert "PSYCH_DETECTED" in result.flags
    assert "PSYCH_NO_PERSONALITY_CHANGE_LOSS" not in result.flags


def test_audit_flags_coarse_and_not_in_tree(auditor, tmp_path):
    case = _base_case()
    case["injuries"]["injuries"] = [
        {
            "injury_id": "inj_001",
            "description": "Multiple fractures of lower leg",
            "injury_type": "Permanent injury",
            "body_part": "left knee, right knee",
            "laterality": None,
            "source": [],
            "icd_code": "NC92.8",
            "icd_description": "Multiple fractures of lower leg",
        },
        {
            "injury_id": "inj_002",
            "description": "made up",
            "injury_type": "Temporary injury",
            "body_part": None,
            "laterality": None,
            "source": [],
            "icd_code": "309.0",
            "icd_description": "made up",
        },
    ]
    case_path = _write_case(tmp_path, "case.json", case)
    result = auditor.audit_case("src.txt", "case.json", case_path)
    assert "COARSE_ICD_NODE" in result.flags
    assert "ICD_CODE_NOT_IN_TREE" in result.flags
    assert "INJURY_MULTI_BODY_PART" in result.flags


def test_audit_clean_case_has_only_source_missing(auditor, tmp_path):
    case = _base_case()
    case["metadata"]["neutral_citation"] = "[2011] HKCFI 500"
    case["plaintiff"].update(
        {
            "age_at_trial": 35,
            "occupation_before": "worker",
            "occupation_after": "clerk",
            "expected_occupation_after": "clerk",
            "salary_before": 10000.0,
            "salary_after": 8000.0,
            "expected_salary_after": 9000.0,
            "education_level": "Form 5",
        }
    )
    case["injuries"]["injuries"] = [
        {
            "injury_id": "inj_001",
            "description": "Fracture of the left distal radius",
            "injury_type": "Permanent injury",
            "body_part": "left wrist",
            "laterality": "left",
            "source": ["An X-ray showed a fracture of the left distal radius."],
            "icd_code": "NC53.0",
            "icd_description": "Fracture of scaphoid bone of hand",
        }
    ]
    case["injuries"]["manifestations"] = []
    case["treatment"] = {
        "treatments_received": ["physiotherapy"],
        "treatments_future": [],
        "hospitalisation_days": 3,
        "expected_hospitalisation_days": 0,
        "operations_count": 1,
        "future_operations_count": 0,
        "sick_leave_days_actual": 30,
        "sick_leave_days_expected": 0,
    }
    case["losses"]["losses"] = [
        {
            "category": "Loss of mobility",
            "description": "Reduced ability to walk long distances.",
            "caused_by_injury_id": "inj_001",
        }
    ]
    case_path = _write_case(tmp_path, "case.json", case)
    result = auditor.audit_case("src.txt", "case.json", case_path)
    # source file does not exist -> SOURCE_MISSING is expected
    assert result.flags == ["SOURCE_MISSING"], result.flags
