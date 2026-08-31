import json
from pathlib import Path


DATA_DIR = Path("output/experiments/1.1-model-comparison_cases/")
FILE_SUFFIX = "html_gpt_single-pass.json"

REPROCESS_DIR = Path("reprocess/")

def main():

    if not REPROCESS_DIR.exists():
        REPROCESS_DIR.mkdir(parents=True)


    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist.")
        return

    files = DATA_DIR.rglob(f"*{FILE_SUFFIX}")

    count = 0
    files_count = 0

    for file_path in files:
        files_count += 1
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR reading {file_path}: {e}")
            continue

        psla = data.get("psla")

        if not isinstance(psla, dict):
            continue

        comparable_cases = psla.get("comparable_cases")

        if not comparable_cases:
            metadata = data.get("metadata", {})
            neutral_citation = metadata.get("neutral_citation")

            print(neutral_citation or f"[NO CITATION] {file_path}")
            count += 1

    print(f"\nFound {count} cases with empty comparable_cases.")
    print(f"Total files processed: {files_count}")


if __name__ == "__main__":
    main()
