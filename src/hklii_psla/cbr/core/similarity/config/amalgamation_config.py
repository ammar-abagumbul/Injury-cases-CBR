from enum import Enum


class AmalgamationConfig(str, Enum):
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"
    WEIGHTED_SUM = "WEIGHTED_SUM"
    EUCLIDEAN = "EUCLIDEAN"
    SIM_DEF = "SIM_DEF"
