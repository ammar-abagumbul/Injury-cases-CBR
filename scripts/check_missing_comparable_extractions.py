#!/usr/bin/env python3
"""
Check which comparable cases (identified by 'action_number_from_judiciary_hk')
in the output experiment JSON files do NOT have a corresponding extracted judgment
file in the judgement_extractions directory.

Source format:  "HCPI 668/2005"           (space, slash)
Target format:  "HCPI_668_2005_gpt_single-pass.json"  (underscores, suffix)
"""

import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

JSON_DIR = os.path.join(PROJECT_ROOT, "output/experiments/1.1-model-comparison_cases")
EXTRACTION_DIR = os.path.join(PROJECT_ROOT, "judgement_extractions/experiments/1.1-model-comparison_cases")


def action_number_to_filename(an: str) -> str:
    """Convert 'HCPI 668/2005' → 'HCPI_668_2005_gpt_single-pass.json'"""
    return an.replace(" ", "_").replace("/", "_") + "_gpt_single-pass.json"


def main() -> int:
    # Build set of existing extraction files
    existing_files: set[str] = set()
    for fname in os.listdir(EXTRACTION_DIR):
        if fname.endswith(".json"):
            existing_files.add(fname)

    print(f"Extraction directory has {len(existing_files)} .json files.\n")

    # Collect all action numbers from output JSONs
    # source_file → list of (comparable_case_name, action_number, expected_filename, exists)
    missing: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    found: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)
    total_action_numbers = 0

    json_files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith(".json"))

    for src_filename in json_files:
        src_path = os.path.join(JSON_DIR, src_filename)
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        comparable = data.get("psla", {}).get("comparable_cases")
        if not isinstance(comparable, list):
            continue

        for case in comparable:
            an = case.get("action_number_from_judiciary_hk")
            if not an:
                continue
            total_action_numbers += 1
            expected = action_number_to_filename(an)
            case_name = case.get("case_name", "(unknown)")
            entry = (case_name, an, expected)

            if expected in existing_files:
                found[src_filename].append(entry)
            else:
                missing[src_filename].append(entry)

    # Report
    print(f"Scanned {len(json_files)} output JSON files.")
    print(f"Found {total_action_numbers} comparable case action numbers.\n")

    if missing:
        print("=" * 70)
        print("MISSING EXTRACTION FILES")
        print("=" * 70)
        for src_file, entries in missing.items():
            print(f"\n[{src_file}]")
            for case_name, an, expected in entries:
                print(f"  • {case_name}")
                print(f"    action_number: {an}")
                print(f"    expected file: {expected}")
        print()
    else:
        print("All comparable cases have matching extraction files.\n")

    # Summary counts
    missing_count = sum(len(v) for v in missing.values())
    found_count = sum(len(v) for v in found.values())

    print(f"Summary: {found_count} matched, {missing_count} missing.")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())