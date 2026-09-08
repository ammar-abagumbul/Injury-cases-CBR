from enum import Enum


class DistanceConfig(str, Enum):
    """DIFFERENCE: distance given by case - query.
    QUOTIENT: similarity based on case / query (only when 0 is not in range)."""

    DIFFERENCE = "DIFFERENCE"
    QUOTIENT = "QUOTIENT"
