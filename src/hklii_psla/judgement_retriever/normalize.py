from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s/.\-\[\]]", re.UNICODE)


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""

    value = value.replace("\xa0", " ")
    return _WHITESPACE_RE.sub(" ", value).strip()


def normalize_identifier(value: str | None) -> str:
    """
    Normalize identifiers such as:

        HCPI 668/2005
        HCPI668/2005
        HCPI No. 668 of 2005

    into a comparable representation.
    """

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.upper().strip()

    value = re.sub(r"\bNO\.?\b", " ", value)
    value = re.sub(r"\bOF\b", "/", value)

    value = re.sub(r"\s+", "", value)

    return value


def normalize_neutral_citation(value: str | None) -> str:
    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.upper().strip()

    # [2007] HKCFI 101
    match = re.search(
        r"\[(\d{4})\]\s*([A-Z]+)\s*(\d+)",
        value,
    )

    if match:
        year, court, number = match.groups()
        return f"{year}{court}{number}"

    return re.sub(r"\W", "", value)


def normalize_case_name(value: str | None) -> str:
    """
    Conservative normalization for party names.

    We normalize:
      - Unicode
      - case
      - line breaks
      - repeated whitespace
      - common punctuation

    We do not perform broad fuzzy matching.
    """

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.upper()

    value = value.replace("\n", " ")
    value = value.replace("\r", " ")

    value = re.sub(r"\s+V\s+", " v ", value)
    value = re.sub(r"\s+VS\.?\s+", " v ", value)

    value = _PUNCTUATION_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value)

    return value.strip()


def identifiers_match(
    requested: str | None,
    candidate: str | None,
    *,
    kind: str,
) -> bool:
    if not requested:
        return True

    if not candidate:
        return False

    if kind == "case_name":
        return normalize_case_name(requested) == normalize_case_name(candidate)

    if kind == "neutral_citation":
        return (
            normalize_neutral_citation(requested)
            == normalize_neutral_citation(candidate)
        )

    if kind == "action_number":
        return (
            normalize_identifier(requested)
            == normalize_identifier(candidate)
        )

    raise ValueError(f"Unknown identifier kind: {kind}")
