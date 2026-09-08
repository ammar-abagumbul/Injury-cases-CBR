from enum import Enum


class StringConfig(str, Enum):
    EQUALITY = "EQUALITY"
    NGRAM = "NGRAM"
    LEVENSHTEIN = "LEVENSHTEIN"
