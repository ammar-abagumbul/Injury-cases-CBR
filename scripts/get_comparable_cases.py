import json
import subprocess
from pathlib import Path

DATA_DIR = Path("output/experiments/1.1-model-comparison_cases/")
FILE_SUFFIX = "html_gpt_single-pass.json"


def main():


    if not DATA_DIR.exists():
        print(f"ERROR: {DATA_DIR} does not exist.")
        return

    files = DATA_DIR.rglob(f"*{FILE_SUFFIX}")

    total_cases = 0
    total_comparables = 0
    successful = 0
    failed = 0
    failed_files = []

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR reading {file_path}: {e}")
            continue

        metadata = data.get("metadata", {})
        neutral_citation = metadata.get("neutral_citation")
        action_number = metadata.get("action_number")
        psla = data.get("psla")

        if not isinstance(psla, dict):
            continue

        comparable_cases = psla.get("comparable_cases")

        if not isinstance(comparable_cases, list) or not comparable_cases:
            continue

        total_cases += 1

        print(f"\n{'=' * 80}")
        print(f"Source case: {file_path.name} | Comparable cases: {len(comparable_cases)}")
        print(f"{'=' * 80}")

        for i, comparable in enumerate(comparable_cases, start=1):
            if not isinstance(comparable, dict):
                continue

            # Build command using only available values.
            command = ["uv", "run", "judgement_retriever"]

            nc = comparable.get("neutral_citation")
            an = comparable.get("action_number")
            cn = comparable.get("case_name")

            if nc:
                command.extend(["--neutral-citation", str(nc)])

            if an:
                command.extend(["--action-number", str(an)])

            if cn:
                command.extend(["--case-name", str(cn)])

            command.extend(["--prefix", str(neutral_citation or "")])

            # Nothing useful to search with.
            if len(command) == 5:
                print(f"\n[{i}/{len(comparable_cases)}] SKIPPED: no identifiers")
                continue

            total_comparables += 1

            print(f"\n[{i}/{len(comparable_cases)}] Running:")
            print(" ".join(command))

            try:

                if not neutral_citation:
                    raise ValueError(f"No neutral citation for {file_path.name}")

                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    successful += 1
                    action_number = result.stdout.split("Action number: ")[1].strip()
                    court, number, year = action_number.split("_")
                    comparable["action_number_from_judiciary_hk"] = f"{court} {number}/{year}"
                else:
                    failed += 1
                    failed_files.append(f"{file_path} - {an or nc or cn}")
                    print(
                        f"FAILED with exit code {result.returncode}: "
                        f"{an or nc or cn}"
                    )



            except OSError as e:
                failed += 1
                print(f"ERROR running retriever: {e}")

        # write back json to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Source cases processed: {total_cases}")
    print(f"Comparable cases attempted: {total_comparables}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_files:
        print(f"Failed files:\n    {"\n    ".join(failed_files)}")



if __name__ == "__main__":
    main()
