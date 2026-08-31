from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CaseQuery:
    case_name: str | None = None
    neutral_citation: str | None = None
    action_number: str | None = None

    def __post_init__(self) -> None:
        if not any(
            value and value.strip()
            for value in (
                self.case_name,
                self.neutral_citation,
                self.action_number,
            )
        ):
            raise ValueError(
                "At least one of case name, neutral citation, "
                "or action number must be supplied."
            )


@dataclass
class JudgmentCandidate:
    """
    A result returned by the Judiciary search page.

    `dis` and `qs` are intentionally kept as raw values because they
    are part of the Judiciary's site-specific retrieval mechanism.
    """

    case_name: str | None
    neutral_citation: str | None
    action_number: str | None

    dis: str
    qs: str

    detail_url: str | None = None
    raw_text: str | None = None


@dataclass
class Judgment:
    source_url: str
    text: str
    case_name: str | None = None
    neutral_citation: str | None = None
    action_number: str | None = None
    judgment_date: str | None = None
    source_type: Literal["html"] = "html"
