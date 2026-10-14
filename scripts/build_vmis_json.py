"""Convert the VMIS Schedule of Surgical Procedures PDF into data/VMIS.json.

The PDF lists, per body system, sub-regions (e.g. "Liver") followed by
procedures that each end with a tier: Minor / Intermediate / Major / Complex.
Output mirrors the ICD-11 tree layout used by the extractor:

    [{"section": "ABDOMINAL AND DIGESTIVE SYSTEM",
      "children": [{"region": "Liver",
                    "children": [{"procedure": "...", "category": "Major"}]}]}]

Usage:
    uv run --with pypdf python scripts/build_vmis_json.py <vmis.pdf> [out.json]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CATEGORIES = ("Minor", "Intermediate", "Major", "Complex")
EXPECTED_COUNTS = {"Minor": 112, "Intermediate": 155, "Major": 149, "Complex": 59}

# Systems in PDF order, each with its sub-regions in PDF order. Regions are only
# matched in order, so a procedure that merely starts with a region word
# ("Spine osteotomy", "Liver transplantation") is not mistaken for a heading.
# Wrapped labels in the PDF are joined with a space.
SYSTEM_REGIONS: dict[str, list[str]] = {
    "ABDOMINAL AND DIGESTIVE SYSTEM": [
        "Oesophageal / stomach / duodenum", "Jejunum, ileum and large intestine",
        "Biliary tract", "Liver", "Pancreas", "Abdominal wall",
    ],
    "BRAIN AND NERVOUS SYSTEM": ["Brain", "Spine"],
    "CARDIOVASCULAR SYSTEM": ["Heart", "Vessels"],
    "ENDOCRINE SYSTEM": ["Adrenal Gland", "Pineal gland", "Pituitary Gland", "Thyroid Gland"],
    "EAR / NOSE / THROAT / RESPIRATORY SYSTEM": [
        "Ear", "Nose, mouth and pharynx", "Respiratory system",
    ],
    "EYE": ["Eye"],
    "FEMALE GENITAL SYSTEM": [
        "Cervix", "Fallopian tubes and ovaries^", "Uterus", "Vagina", "Vulva and introitus",
    ],
    "HEMIC AND LYMPHATIC SYSTEM": ["Lymph Nodes", "Spleen"],
    "MALE GENITAL SYSTEM": ["Prostate", "Penis", "Testicles^", "Spermatic cord"],
    "MUSCULOSKELETAL SYSTEM": [
        "Bone", "Joint", "Muscle / Tendon", "Fracture / dislocation", "Spine", "Others",
    ],
    "SKIN AND BREAST": ["Skin", "Breast"],
    "URINARY SYSTEM": ["Kidney", "Bladder, ureter and urethra"],
    "DENTAL": [],
}

_NOISE = {"SPECIMEN", "Schedule of Surgical Procedures", "Procedure / Surgery Category"}
_FOOTNOTE = re.compile(r"^\^ ")
_PROCEDURE_END = re.compile(rf"^(.*?)\s*\b({'|'.join(CATEGORIES)})$")
_MAX_LABEL_LINES = 3


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _match_region(lines: list[str], i: int, label: str) -> tuple[int, str] | None:
    """If ``label`` starts at ``lines[i]`` (possibly wrapped), return (next_i, remainder)."""
    for k in range(_MAX_LABEL_LINES):
        joined = " ".join(lines[i:i + k + 1])
        if joined == label:
            return i + k + 1, ""
        if joined.startswith(label + " "):
            return i + k + 1, joined[len(label) + 1:]
    return None


def parse(pdf_path: Path) -> list[dict]:
    from pypdf import PdfReader

    raw: list[str] = []
    for page in PdfReader(str(pdf_path)).pages:
        raw.extend(_clean(line) for line in page.extract_text().splitlines())
    lines = [ln for ln in raw if ln and ln not in _NOISE and not _FOOTNOTE.match(ln)]

    tree: list[dict] = []
    section: dict | None = None
    region: dict | None = None
    expected: list[str] = []
    buf: list[str] = []

    def open_region(name: str) -> None:
        nonlocal region
        assert section is not None
        region = {"region": name.rstrip("^"), "children": []}
        section["children"].append(region)

    i = 0
    while i < len(lines):
        line = lines[i]

        if line in SYSTEM_REGIONS:
            section = {"section": line, "children": []}
            tree.append(section)
            expected = list(SYSTEM_REGIONS[line])
            region, buf = None, []
            if line == "DENTAL":
                open_region("Dental")
            i += 1
            continue

        if not buf and expected:
            hit = _match_region(lines, i, expected[0])
            if hit:
                i, rest = hit
                open_region(expected.pop(0))
                if rest:
                    lines.insert(i, rest)  # procedure text that shared the heading's line
                continue

        buf.append(line)
        m = _PROCEDURE_END.match(" ".join(buf))
        if m and (m.group(1) or len(buf) > 1):
            assert region is not None, f"procedure outside region: {buf}"
            region["children"].append({"procedure": _clean(m.group(1)), "category": m.group(2)})
            buf = []
        i += 1

    assert not buf, f"unterminated procedure: {buf}"
    for sec in tree:
        assert not any(not r["children"] for r in sec["children"]), sec["section"]
    return tree


def main() -> None:
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/VMIS.json")
    tree = parse(pdf)

    counts = {c: 0 for c in CATEGORIES}
    for sec in tree:
        for reg in sec["children"]:
            for proc in reg["children"]:
                counts[proc["category"]] += 1
    print("tier counts:", counts, "total:", sum(counts.values()))
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"tier counts differ from the PDF's expected {EXPECTED_COUNTS}")

    out.write_text(json.dumps(tree, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
