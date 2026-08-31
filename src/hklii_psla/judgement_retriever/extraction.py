"""
HTML judgment extraction for the Hong Kong Judiciary Legal Reference System.

The Judiciary's current HTML judgment representation has a highly regular
structure:

    <table>
        ...
        <p>...</p>
        <p class="heading">...</p>
        <blockquote>
            <p>...</p>
        </blockquote>
        ...
    </table>

Paragraphs are represented by <p> elements, and numbered paragraphs have
an <a class="para" id="pN">...</a> marker containing the paragraph number.

This module intentionally relies on that structure rather than attempting
generic webpage/article extraction.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag


# Tags belonging to the website itself rather than the judgment.
# NOTE: <form> is intentionally NOT included here because the Judiciary
# body page wraps the entire judgment table inside a <form name="search_body">.
_SITE_TAGS = {
    "script",
    "style",
    "noscript",
    "iframe",
    "object",
}


def _clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving meaningful paragraph content.

    The Judiciary HTML frequently uses:
        &nbsp;
        multiple spaces
        line breaks
        spacing around inline elements

    These should not survive into the extracted text.
    """

    text = text.replace("\xa0", " ")

    # Normalize all whitespace runs to one space.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _extract_paragraph(paragraph: Tag) -> str:
    """
    Extract one <p> element.

    The paragraph number is already represented in the HTML and therefore
    remains part of the extracted text.

    Inline elements such as <i>, <b>, and <em> are deliberately flattened
    into ordinary text. Their formatting is not needed for the text corpus.
    """

    text = paragraph.get_text(
        " ",
        strip=True,
    )

    return _clean_text(text)


def _extract_blockquote(blockquote: Tag) -> list[str]:
    """
    Extract a quotation.

    A blockquote may contain one or more paragraphs:

        <blockquote>
            <p>...</p>
        </blockquote>

    We preserve each paragraph separately.
    """

    paragraphs = blockquote.find_all(
        "p",
        recursive=False,
    )

    if not paragraphs:
        text = _clean_text(
            blockquote.get_text(
                " ",
                strip=True,
            )
        )

        return [text] if text else []

    return [
        text
        for paragraph in paragraphs
        if (text := _extract_paragraph(paragraph))
    ]


def _find_judgment_table(soup: BeautifulSoup) -> Tag:
    """
    Locate the table containing the judgment.

    The Judiciary page has an outer table containing the complete judgment,
    with nested tables used for parties and the signature block.

    We identify the judgment table by looking for the characteristic
    judgment content rather than simply selecting the first table.
    """

    tables = soup.find_all("table")

    if not tables:
        raise ValueError(
            "No table found in Judiciary HTML."
        )

    best_table: Tag | None = None
    best_score = -1

    for table in tables:
        text = table.get_text(
            " ",
            strip=True,
        )

        score = 0

        # Strong indicators that this is the judgment table.
        if "J U D G M E N T" in text.upper():
            score += 100

        if "IN THE HIGH COURT OF THE" in text.upper():
            score += 50

        if "Date of Judgment" in text:
            score += 50

        # Numbered judgment paragraphs are an especially strong signal.
        paragraph_count = len(
            table.find_all(
                "a",
                class_="para",
            )
        )

        score += min(
            paragraph_count,
            100,
        )

        # Prefer larger tables when scores are otherwise similar.
        score += min(
            len(text) // 10_000,
            20,
        )

        if score > best_score:
            best_score = score
            best_table = table

    if best_table is None or best_score <= 0:
        raise ValueError(
            "Could not identify the Judiciary judgment table."
        )

    return best_table


def extract_judgment_text(html: str) -> str:
    """
    Extract the judgment text from a Judiciary HTML representation.

    Output is plain UTF-8 text with:

      - headings on their own lines
      - numbered paragraphs on their own lines
      - block quotes preserved as separate paragraphs
      - parties/coram/date information preserved
      - website navigation/javascript removed
      - inline HTML formatting flattened

    No attempt is made to summarize, rewrite, or otherwise modify the
    judgment's wording.
    """

    if not html or not html.strip():
        raise ValueError(
            "Cannot extract judgment from empty HTML."
        )

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove website implementation details.
    for tag_name in _SITE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    judgment_table = _find_judgment_table(soup)

    output: list[str] = []

    # ------------------------------------------------------------------
    # We only process direct children of the judgment table's main <td>.
    #
    # This prevents nested tables (parties/signature) from being
    # accidentally extracted a second time.
    # ------------------------------------------------------------------

    content_cell = judgment_table.find("td")

    if content_cell is None:
        raise ValueError(
            "Judgment table does not contain a content cell."
        )

    for child in content_cell.children:
        if not isinstance(child, Tag):
            continue

        # --------------------------------------------------------------
        # Ordinary paragraphs and headings.
        # --------------------------------------------------------------

        if child.name == "p":
            text = _extract_paragraph(child)

            if text:
                output.append(text)

            continue

        # --------------------------------------------------------------
        # Parties section.
        #
        # The parties are contained in a custom <parties> element,
        # which itself contains a nested table.
        # --------------------------------------------------------------

        if child.name == "parties":
            parties_table = child.find("table")

            if parties_table is not None:
                rows = parties_table.find_all(
                    "tr",
                    recursive=False,
                )

                for row in rows:
                    cells = row.find_all(
                        "td",
                        recursive=False,
                    )

                    values = [
                        _clean_text(
                            cell.get_text(
                                " ",
                                strip=True,
                            )
                        )
                        for cell in cells
                    ]

                    values = [
                        value
                        for value in values
                        if value
                    ]

                    if values:
                        output.append(
                            "    ".join(values)
                        )

            continue

        # --------------------------------------------------------------
        # Coram and date sections.
        # --------------------------------------------------------------

        if child.name in {
            "coram",
            "date",
        }:
            for paragraph in child.find_all(
                "p",
                recursive=False,
            ):
                text = _extract_paragraph(paragraph)

                if text:
                    output.append(text)

            continue

        # --------------------------------------------------------------
        # Quotations.
        # --------------------------------------------------------------

        if child.name == "blockquote":
            output.extend(
                _extract_blockquote(child)
            )

            continue

        # --------------------------------------------------------------
        # Representation section.
        # --------------------------------------------------------------

        if child.name == "representation":
            for paragraph in child.find_all(
                "p",
                recursive=False,
            ):
                text = _extract_paragraph(paragraph)

                if text:
                    output.append(text)

            continue

        # --------------------------------------------------------------
        # Nested tables outside the known semantic sections.
        #
        # In particular, this catches the judge's signature block.
        # We flatten its rows without recursively processing the same
        # nested table as part of the outer document.
        # --------------------------------------------------------------

        if child.name == "table":
            for row in child.find_all(
                "tr",
                recursive=False,
            ):
                cells = row.find_all(
                    "td",
                    recursive=False,
                )

                values = [
                    _clean_text(
                        cell.get_text(
                            " ",
                            strip=True,
                        )
                    )
                    for cell in cells
                ]

                values = [
                    value
                    for value in values
                    if value
                ]

                if values:
                    output.append(
                        "    ".join(values)
                    )

    # Remove accidental empty lines.
    output = [
        line.strip()
        for line in output
        if line.strip()
    ]

    if not output:
        raise ValueError(
            "Judiciary judgment table contained no extractable text."
        )

    return "\n\n".join(output)
