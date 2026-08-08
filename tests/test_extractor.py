"""Tests for extraction module."""

import pytest

from hklii_psla.extractor.base import BaseExtractor


class TestParseJSON:
    def test_valid_json(self):
        raw = '{"name": "test", "value": 42}'
        d, err = BaseExtractor._parse_json_response(raw)
        assert err is None
        assert d == {"name": "test", "value": 42}

    def test_json_in_code_block(self):
        raw = '```json\n{"name": "test"}\n```'
        d, err = BaseExtractor._parse_json_response(raw)
        assert err is None
        assert d == {"name": "test"}

    def test_json_in_bare_block(self):
        raw = '```\n{"name": "test"}\n```'
        d, err = BaseExtractor._parse_json_response(raw)
        assert err is None
        assert d == {"name": "test"}

    def test_invalid_json(self):
        raw = "not json at all"
        d, err = BaseExtractor._parse_json_response(raw)
        assert d is None
        assert err is not None


class TestDictToCase:
    def test_valid_dict(self):
        data = {
            "metadata": {
                "neutral_citation": "[2020] HKDC 1745",
                "case_name": "TEST v TEST",
            },
            "plaintiff": {"gender": "Male"},
            "injuries": {"injuries": []},
            "treatment": {},
            "losses": {"losses": []},
            "psla": {"comparable_cases": []},
            "injury_loss_relations": [],
        }
        case = BaseExtractor._dict_to_case(data)
        assert case is not None
        assert case.metadata.case_name == "TEST v TEST"

    def test_invalid_dict(self):
        data = {"not": "a case"}
        case = BaseExtractor._dict_to_case(data)
        assert case is None
