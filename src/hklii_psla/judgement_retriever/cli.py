from __future__ import annotations

import argparse
import logging
import sys

from .models import CaseQuery
from .sources.judiciary import JudiciaryClient
from .storage import save_judgment
from .normalize import identifiers_match


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Hong Kong Judiciary judgments "
            "from the Legal Reference System."
        )
    )

    parser.add_argument(
        "--case-name",
        help="Case name / parties of judgment.",
    )

    parser.add_argument(
        "--neutral-citation",
        help="Neutral citation, e.g. '[2007] HKCFI 101'.",
    )

    parser.add_argument(
        "--action-number",
        help="Action number, e.g. 'HCPI 668/2005'.",
    )

    parser.add_argument(
        "--prefix",
        help="An associated suffix (like filename) for grouping purposes."
    )

    parser.add_argument(
        "--output-dir",
        default="judgments",
        help="Output directory.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s: %(message)s"
        ),
    )

    try:
        query = CaseQuery(
            case_name=args.case_name,
            neutral_citation=args.neutral_citation,
            action_number=args.action_number,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        with JudiciaryClient() as judiciary:
            judgment = judiciary.search_and_retrieve(query)

            if judgment is None:
                print(
                    "No judgments found.",
                    file=sys.stderr,
                )
                return 1

            path = save_judgment(
                judgment,
                args.output_dir,
                # prefix=args.prefix,
            )

            print(f"Saved: {path}")
            print(
                "Action number: "
                f"{judgment.action_number}"
            )

            return 0

    except KeyboardInterrupt:
        return 130

    except Exception as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
