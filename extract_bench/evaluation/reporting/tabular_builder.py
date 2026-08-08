"""Compact tabular builder for multi-document evaluation views.

Builds a compact tabular representation where:
- Each row = one extraction-CSV pair (identified by neutral_citation)
- Each column = one schema field
- Each cell = {"status": "match"|"miss"|"skipped", "score": float|null}
"""

from typing import Any


class CompactTabularBuilder:
    """Builds compact tabular view from pair evaluation results."""

    def build(
        self,
        pair_results: list[dict[str, Any]],
        field_names: list[str],
    ) -> dict[str, Any]:
        """Build tabular report with rows=pairs, columns=fields.

        Args:
            pair_results: List of pair result dicts, each containing:
                - neutral_citation: str
                - judgment_id: str
                - row_index: int
                - overall_score: float
                - field_results: dict[str, {"status": str, "score": float|null,
                  "reason": str|null}]
                - skipped_fields: list[str]
            field_names: Ordered list of schema field names (columns).

        Returns:
            Dict with "columns" and "rows" keys suitable for JSON serialization.
        """
        rows: list[dict[str, Any]] = []
        for pair in pair_results:
            row: dict[str, Any] = {
                "neutral_citation": pair.get("neutral_citation", ""),
                "judgment_id": pair.get("judgment_id", ""),
                "row_index": pair.get("row_index", -1),
                "overall_score": pair.get("overall_score", 0.0),
            }

            field_results = pair.get("field_results", {})
            skipped_fields = set(pair.get("skipped_fields", []))

            for col in field_names:
                if col in field_results:
                    cell = field_results[col]
                    row[col] = {
                        "status": cell.get("status", "skipped"),
                        "score": cell.get("score"),
                        "reason": cell.get("reason"),
                    }
                elif col in skipped_fields:
                    row[col] = {
                        "status": "skipped",
                        "score": None,
                        "reason": "field_not_in_both",
                    }
                else:
                    row[col] = {
                        "status": "skipped",
                        "score": None,
                        "reason": "not_evaluated",
                    }

            rows.append(row)

        return {
            "columns": field_names,
            "rows": rows,
        }

    def to_json(self, tabular: dict[str, Any]) -> str:
        """Serialize tabular report to JSON string."""
        import json

        return json.dumps(tabular, indent=2, default=str)
