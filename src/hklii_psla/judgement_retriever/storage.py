from __future__ import annotations

import re
from pathlib import Path

from .models import Judgment


_CITATION_RE = re.compile(
    r"\[(\d{4})\]\s*([A-Za-z]+)\s*(\d+)"
)

def citation_to_filename(neutral_citation: str) -> str:
    match = _CITATION_RE.search(neutral_citation)

    if not match:
        raise ValueError(
            f"Cannot generate filename from neutral citation: "
            f"{neutral_citation!r}"
        )

    year, court, number = match.groups()

    return f"{court.upper()}_{year}_{number}.txt"


def save_judgment(
    judgment: Judgment,
    output_dir: str | Path = "judgments",
    prefix: str | None = None,
) -> Path:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if judgment.action_number:
        filename = f"{prefix + "-" if prefix else ""}{judgment.action_number}.txt"
    elif judgment.neutral_citation:
        filename = f"{prefix + "-" if prefix else ""}{citation_to_filename(judgment.neutral_citation)}"
    else:
        raise ValueError(
            "Cannot save judgment without an action number or neutral citation."
        )

    path = output_path / filename

    path.write_text(
        judgment.text,
        encoding="utf-8",
    )

    return path
