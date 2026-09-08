from enum import Enum


class NumberConfig(str, Enum):
    CONSTANT = "CONSTANT"
    STEP_AT = "STEP_AT"
    POLYNOMIAL_WITH = "POLYNOMIAL_WITH"
    SMOOTH_STEP_AT = "SMOOTH_STEP_AT"
