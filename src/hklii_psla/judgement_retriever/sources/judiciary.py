from __future__ import annotations

import logging
import re

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlencode, urljoin

import httpx
from bs4 import BeautifulSoup

from ..extraction import extract_judgment_text
from ..models import CaseQuery, Judgment, JudgmentCandidate

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search mode identifiers and their corresponding txtselectopt values.
#
# Derived from the Judiciary site's SUBMIT_SEARCH() JavaScript:
#   - Action Number   → txtselectopt = 4
#   - Neutral Citation → txtselectopt = 12
#   - Case Name/Party → txtselectopt = 11  (formatted as (name)@party)
# ---------------------------------------------------------------------------

SearchMode = Literal["action_number", "neutral_citation", "case_name"]

_TXTSELECTOPT: dict[SearchMode, str] = {
    "action_number": "4",
    "neutral_citation": "12",
    "case_name": "11",
}

# Priority order for fallback: most-precise → least-precise.
_MODE_PRIORITY: list[SearchMode] = [
    "action_number",
    "neutral_citation",
    "case_name",
]


BASE_URL = "https://legalref.judiciary.hk"

SEARCH_URL = (
    f"{BASE_URL}/lrs/common/ju/judgment.jsp"
)
ACTION_URL = "../search/searchbox_result.jsp"

DETAIL_URL = (
    f"{BASE_URL}/lrs/common/search/"
    "search_result_detail_frame.jsp"
)

JUDGEMENT_FRAME_URL = (
    f"{BASE_URL}/lrs/common/search/search_result_detail_frame.jsp"
)

COOKIES = {
    "LrsLan": "en",
    "ispopup": "0",
    "jbudes": "",
}


@dataclass
class JudiciaryClient:
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.client = httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            cookies=COOKIES
        )
        self._ensure_session()

    def _ensure_session(self) -> None:
        """
        Visit the search page to obtain a fresh JSESSIONID and other
        session cookies from the server.

        The Judiciary site issues new session cookies on each visit;
        hardcoded cookies will be stale and the server will return
        empty responses.
        """
        logger.debug("Obtaining fresh session cookies from search page")
        response = self.client.get(SEARCH_URL)
        response.raise_for_status()
        logger.debug("Session cookies obtained: %s", dict(self.client.cookies))

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "JudiciaryClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_search_page(self) -> str:
        response = self.client.get(SEARCH_URL)
        response.raise_for_status()

        logger.debug(
            "Search page: status=%s url=%s",
            response.status_code,
            response.url,
        )

        return response.text

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: CaseQuery,
        mode: SearchMode | None = None,
    ) -> list[JudgmentCandidate]:
        """
        Search the Judiciary LRS.

        Parameters
        ----------
        query:
            The case query containing one or more of case_name,
            neutral_citation, and action_number.
        mode:
            Which search mode to use.  When ``None`` (the default) the
            most-precise available field is chosen automatically:
            action_number > neutral_citation > case_name.
        """
        if mode is None:
            mode = self._auto_detect_mode(query)

        txtselectopt = _TXTSELECTOPT[mode]
        search_value = self._format_search_value(query, mode)

        action = urljoin(SEARCH_URL, ACTION_URL)

        payload = {
            "txtSearch": search_value,
            "txtselectopt": txtselectopt,
            "isadvsearch": "0",
            "stem": "1",
            "selall": "1",
            "ncnLanguage": "en",
            "selallct": "1",
            "selDatabase": ["ALL", "JU", "RV", "RS", "PD"],
            "selSchct": ["FA", "CA", "HC", "CT", "DC", "FC", "LD", "OT"],
            "ncnValue": "",
            "ncnParagraph": "",
            "query": "",
        }

        logger.debug(
            "Search request: mode=%s txtselectopt=%s url=%s payload=%s",
            mode,
            txtselectopt,
            action,
            payload,
        )

        response = self.client.get(
            action,
            params=payload,
        )

        response.raise_for_status()

        logger.debug(
            "Search response: status=%s url=%s",
            response.status_code,
            response.url,
        )

        return self._parse_candidates(response.text)

    # ------------------------------------------------------------------
    # search_and_retrieve  (high-level: fallback + validation)
    # ------------------------------------------------------------------

    def search_and_retrieve(
        self,
        query: CaseQuery,
    ) -> Judgment | None:
        """
        Search with automatic fallback across available query fields.

        When *query* supplies more than one field (e.g. both an action
        number and a case name), each mode is tried in priority order.
        If the first candidate from a mode is rejected by the first-30-
        lines heuristic, the next mode is attempted.

        Returns the first validated :class:`Judgment`, or ``None`` when
        no mode produces an acceptable candidate.
        """
        modes = self._available_modes(query)

        for mode in modes:
            logger.debug("Trying search mode: %s", mode)

            candidates = self.search(query, mode=mode)

            if not candidates:
                logger.debug("No candidates for mode: %s", mode)
                continue

            # We only consider the first candidate per mode.
            candidate = candidates[0]

            try:
                judgment = self.retrieve(candidate)
            except Exception:
                logger.debug(
                    "Retrieve failed for mode %s, candidate DIS=%s",
                    mode,
                    candidate.dis,
                    exc_info=True,
                )
                continue

            if self._validate_candidate(judgment.text, query, mode):
                logger.debug(
                    "Candidate accepted (mode=%s, DIS=%s)",
                    mode,
                    candidate.dis,
                )
                return judgment

            logger.debug(
                "Candidate rejected by heuristic (mode=%s, DIS=%s)",
                mode,
                candidate.dis,
            )

        return None

    # ------------------------------------------------------------------
    # Helpers: mode detection & search-value formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_mode(query: CaseQuery) -> SearchMode:
        """Pick the most-precise available search mode."""
        for mode in _MODE_PRIORITY:
            value = getattr(query, mode, None)
            if value and value.strip():
                return mode
        raise ValueError("No searchable query value supplied.")

    @staticmethod
    def _available_modes(query: CaseQuery) -> list[SearchMode]:
        """Return every mode for which *query* has a non-empty value."""
        return [
            mode
            for mode in _MODE_PRIORITY
            if getattr(query, mode, None)
            and str(getattr(query, mode)).strip()
        ]

    @staticmethod
    def _format_search_value(
        query: CaseQuery,
        mode: SearchMode,
    ) -> str:
        """
        Format the search string for the Judiciary LRS.

        Party-name searches use the ``(name)@party`` syntax recognised
        by the Judiciary's quick-search box.  When the case name contains
        "v" / "VS", only the first party is used.
        """
        if mode == "action_number":
            return (query.action_number or "").strip()

        if mode == "neutral_citation":
            return (query.neutral_citation or "").strip()

        if mode == "case_name":
            raw = (query.case_name or "").strip()
            # Extract the first party when the name contains "v" / "VS".
            party = re.split(r"\s+[vV]\s+|\s+VS\.?\s+", raw, maxsplit=1)[0].strip()
            party = re.sub(r"\s+", " ", party)
            return f"({party})@party" if party else raw

        raise ValueError(f"Unknown search mode: {mode}")

    # ------------------------------------------------------------------
    # Candidate validation (first-30-lines heuristic)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_candidate(
        judgment_text: str,
        query: CaseQuery,
        mode: SearchMode,
    ) -> bool:
        """
        Return ``True`` when the judgment's opening lines contain the
        search terms, indicating the retrieved document is likely correct.

        The first 30 lines are inspected:

        * **action_number / neutral_citation** – every digit sequence
          extracted from the query value must appear somewhere in those
          lines.
        * **case_name** – at least half of the significant (length > 1)
          tokens from the parties names must appear.
        """
        lines = judgment_text.split("\n")[:30]
        window = "\n".join(lines).upper()

        if mode in ("action_number", "neutral_citation"):
            value = (
                query.action_number
                if mode == "action_number"
                else query.neutral_citation
            ) or ""
            numbers = re.findall(r"\d+", value)
            if not numbers:
                return True
            return any(num in window for num in numbers)

        if mode == "case_name":
            raw = (query.case_name or "").strip()
            parties = re.split(
                r"\s+[vV]\s+|\s+VS\.?\s+", raw, maxsplit=1
            )
            tokens = []
            for party in parties:
                tokens.extend(t for t in party.strip().upper() if len(t) > 1)
            if not tokens:
                return True
            matches = sum(1 for t in tokens if t in window)
            return (matches / len(tokens)) >= 0.5

        return False

    def _parse_candidates(
        self,
        html: str,
    ) -> list[JudgmentCandidate]:
        """
        Parse search results.

        The Judiciary result page is legacy HTML. Rather than assuming
        a particular table schema, inspect links for DIS/QS parameters.
        """

        soup = BeautifulSoup(html, "html.parser")

        a_tags = soup.find_all("a", href=True)

        # filter tags which have classes: searchfont and result-caseno
        a_tags = [
            tag for tag in a_tags
            if tag.has_attr("class")
            and "searchfont" in tag["class"]
            and "result-caseno" in tag["class"]
        ]

        if not a_tags:
            return []

        link = a_tags[0]

        # we only select the first a tag
        # example href: "javascript:judpop1('search_result_detail_frame.jsp?%27+temp67562);"
        # we parse the number temp<number> with regex matching
        pattern = r"temp(\d+)"
        match = re.search(pattern, str(link["href"]))
        if not match:
            return []

        dis = match.group(1)
        qs = "+"
        absolute_url = JUDGEMENT_FRAME_URL

        logger.debug(
            "Parsed %d Judiciary candidates but chose the first one",
            len(a_tags),
        )

        return [
                JudgmentCandidate(
                case_name=None,
                neutral_citation=None,
                action_number=None,
                dis=dis,
                qs=qs,
                detail_url=(
                    f"{absolute_url}?" + urlencode(
                        {
                            "DIS": dis,
                            "QS": qs,
                        }
                    )
                ),
                raw_text=link.get_text(),
            )
        ]

    def retrieve(
        self,
        candidate: JudgmentCandidate,
    ) -> Judgment:
        if not candidate.detail_url:
            candidate.detail_url = (
                f"{DETAIL_URL}?"
                + urlencode(
                    {
                        "DIS": candidate.dis,
                        "QS": candidate.qs,
                    }
                )
            )

        logger.debug(
            "Retrieving detail page: %s",
            candidate.detail_url,
        )

        # Step 1: Fetch the frameset page.
        # The Judiciary site returns a <frameset> page where the actual
        # judgment content lives inside the mainFrame, which loads
        # search_result_detail_body.jsp.  Browsers automatically load
        # all frames, but httpx only fetches the outer frameset.
        frameset_response = self.client.get(candidate.detail_url)
        frameset_response.raise_for_status()

        # Step 2: Extract the mainFrame src URL from the frameset.
        # The src is relative to the frameset page, so we resolve it
        # against the frameset URL (not BASE_URL).
        body_url = self._extract_mainframe_url(
            frameset_response.text,
            base_url=str(frameset_response.url),
        )

        if body_url is None:
            raise RuntimeError(
                "Could not find mainFrame URL in frameset response."
            )

        logger.debug("Fetching judgment body: %s", body_url)

        # Step 3: Fetch the actual judgment body.
        body_response = self.client.get(body_url)
        body_response.raise_for_status()

        text = extract_judgment_text(body_response.text)

        action_number = self._extract_action_number(body_response.text)

        return Judgment(
            action_number=action_number,
            source_url=str(body_response.url),
            source_type="html",
            text=text,
        )

    @staticmethod
    def _extract_mainframe_url(
        frameset_html: str,
        base_url: str,
    ) -> str | None:
        """
        Extract the mainFrame src URL from the Judiciary frameset page.

        The frameset contains:

            <frame name="mainFrame"
                   src="search_result_detail_body.jsp?DIS=...&QS=...">

        The src is relative to the frameset page, so we resolve it
        against the frameset URL (e.g. .../lrs/common/search/...).
        """
        soup = BeautifulSoup(frameset_html, "html.parser")

        main_frame = soup.find("frame", attrs={"name": "mainFrame"})

        if main_frame is None or not main_frame.get("src"):
            return None

        src = main_frame["src"]
        return urljoin(base_url, str(src))

    @staticmethod
    def _find_html_judgment_url(
        detail_html: str,
        base_url: str,
    ) -> str | None:
        soup = BeautifulSoup(
            detail_html,
            "html.parser",
        )

        for link in soup.find_all(
            "a",
            href=True,
        ):
            href = link["href"]
            label = link.get_text(
                " ",
                strip=True,
            ).lower()

            # Look for explicit HTML/judgment links.
            if (
                "html" in label
                or "judgment" in label
                or "judgement" in label
            ):
                return urljoin(
                    base_url,
                    str(href),
                )

        # Also inspect iframes/frames because the Judiciary site
        # may expose the actual document through a frame.
        for frame in soup.find_all(
            ["iframe", "frame"],
            src=True,
        ):
            src = str(frame["src"])

            if (
                "judgment" in src.lower()
                or "judgement" in src.lower()
                or "document" in src.lower()
            ):
                return urljoin(
                    base_url,
                    src,
                )

        return None

    @staticmethod
    def _extract_action_number(
        html: str,
    ) -> str | None:
        """
        Extract the action number from the HTML.

        This will be tightened once the real judgment HTML has been
        inspected.
        """

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        text = soup.get_text(
            "\n",
            strip=True,
        )[:50] # action number appears early in the text

        action_number = None

        pattern = r"([A-Z]+).*?(\d+)/(\d+)"
        action_match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if action_match:
            court, number, volume = action_match.groups()

            # this is used for file naming
            action_number = (
                f"{court}_{int(number)}_{int(volume)}"
            )

        return action_number
