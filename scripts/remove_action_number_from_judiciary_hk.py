#!/usr/bin/env python3
"""Remove all 'action_number_from_judiciary_hk' fields from comparable_cases in each JSON file."""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # scripts/ is inside the project root
TARGET_DIR = os.path.join(PROJECT_ROOT, "output/experiments/1.1-model-comparison_cases")

FIELD = "action_number_from_judiciary_hk"


def clean_file(filepath: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    psla = data.get("psla")
    if psla is None:
        return False

    comparable_cases = psla.get("comparable_cases")
    if not isinstance(comparable_cases, list):
        return False

    removed = 0
    for case in comparable_cases:
        if FIELD in case:
            del case[FIELD]
            removed += 1

    if removed == 0:
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return True


def main() -> int:
    if not os.path.isdir(TARGET_DIR):
        print(f"ERROR: directory not found: {TARGET_DIR}", file=sys.stderr)
        return 1

    json_files = sorted(
        f for f in os.listdir(TARGET_DIR) if f.endswith(".json")
    )

    files_changed = 0

    for filename in json_files:
        filepath = os.path.join(TARGET_DIR, filename)
        try:
            changed = clean_file(filepath)
            if changed:
                files_changed += 1
                print(f"Cleaned: {filename}")
        except Exception as e:
            print(f"ERROR processing {filename}: {e}", file=sys.stderr)
            return 1

    # Count remaining occurrences to verify
    remaining = 0
    for filename in json_files:
        filepath = os.path.join(TARGET_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            remaining += f.read().count(f'"{FIELD}"')

    print(f"\nDone. {files_changed} file(s) modified.")
    print(f"Remaining occurrences of '{FIELD}': {remaining}")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())