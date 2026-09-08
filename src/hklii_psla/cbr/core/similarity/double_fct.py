from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.distance_config import DistanceConfig
from hklii_psla.cbr.core.similarity.config.number_config import NumberConfig
from hklii_psla.cbr.core.similarity.number_fct import NumberFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.double_desc import DoubleDesc
    from hklii_psla.cbr.core.project import Project


class DoubleFct(NumberFct):
    """Configurable similarity function for doubles: CONSTANT, STEP_AT,
    POLYNOMIAL_WITH or SMOOTH_STEP_AT, separately for query<case and case<query."""

    _CONSTANT_VALUE = 1.0
    _STEP_AT_VALUE = 0.0
    _POLYNOMIAL_WITH_VALUE = 1.0
    _SMOOTH_STEP_AT_VALUE = 0.0

    def __init__(self, prj: Project, desc: DoubleDesc, name: str):
        super().__init__(prj, desc, name)
        self.sub_desc = desc
        self.range = desc.double_range
        self.max = desc.max
        self.min = desc.min
        self.diff = self.max - self.min

        self.function_type_r = NumberConfig.CONSTANT
        self.function_parameter_r = self._CONSTANT_VALUE
        self.function_type_l = NumberConfig.CONSTANT
        self.function_parameter_l = self._CONSTANT_VALUE
        self.max_for_quotient = 10.0

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.double_attribute import DoubleAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        result = Similarity.INVALID_SIM
        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)
        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)
        if isinstance(a1, DoubleAttribute) and isinstance(a2, DoubleAttribute):
            q, c = a1.value, a2.value
            if a1.attribute_desc is not self.desc or a2.attribute_desc is not self.desc:
                return result

            d = (c / q) if self.distance_function == DistanceConfig.QUOTIENT else (c - q)

            if (d < 0 and self.distance_function == DistanceConfig.DIFFERENCE) or (
                d < 1 and self.distance_function == DistanceConfig.QUOTIENT
            ):
                result = self._eval(self.function_type_l, d, self.function_parameter_l, self.diff)
            elif (d > 0 and self.distance_function == DistanceConfig.DIFFERENCE) or (
                d > 1 and self.distance_function == DistanceConfig.QUOTIENT
            ):
                result = self._eval(self.function_type_r, d, self.function_parameter_r, -self.diff)
            else:
                return Similarity.get(1.00)
        return result

    def _eval(self, function_type: NumberConfig, d: float, param: float, diff: float) -> Similarity:
        if function_type == NumberConfig.STEP_AT:
            left = diff > 0
            return self._step_at(d, param, left)
        if function_type == NumberConfig.POLYNOMIAL_WITH:
            return self._polynomial_with(d, param, diff)
        if function_type == NumberConfig.SMOOTH_STEP_AT:
            if diff > 0:
                return Similarity.get(1 / (1 + math.exp((-d + param) * (100 / diff))))
            return Similarity.get(1 / (1 + math.exp((d - param) * (100 / -diff))))
        return Similarity.get(param)

    def _step_at(self, value: float, step: float, left: bool) -> Similarity:
        if left:
            return Similarity.get(0.00) if value < step else Similarity.get(1.00)
        return Similarity.get(0.00) if value > step else Similarity.get(1.00)

    def _polynomial_with(self, value: float, exponent: float, diff: float) -> Similarity:
        return Similarity.get(math.pow(value / diff + 1, exponent))

    def set_function_parameter_r(self, value: float) -> bool:
        updated = self._set_function_parameter(True, value)
        if updated and self.is_symmetric:
            self._mirror_parameter(source_is_r=True)
        return updated

    def set_function_parameter_l(self, value: float) -> bool:
        if self.function_type_l in (NumberConfig.SMOOTH_STEP_AT, NumberConfig.STEP_AT) and value > 0:
            value = -value
        updated = self._set_function_parameter(False, value)
        if updated and self.is_symmetric:
            self._mirror_parameter(source_is_r=False)
        return updated

    def _set_function_parameter(self, is_r: bool, value: float) -> bool:
        function_type = self.function_type_r if is_r else self.function_type_l
        updated = False
        if function_type in (NumberConfig.SMOOTH_STEP_AT, NumberConfig.STEP_AT):
            if self.distance_function == DistanceConfig.DIFFERENCE and (
                (is_r and 0 < value <= self.diff) or (not is_r and value >= -self.diff)
            ):
                updated = True
            elif self.distance_function == DistanceConfig.QUOTIENT and (
                (is_r and 1 < value <= self.max / self.min)
                or (not is_r and value < 1 and value >= self.min / self.max)
            ):
                updated = True
        elif function_type == NumberConfig.CONSTANT:
            updated = 0 <= value <= 1
        elif function_type == NumberConfig.POLYNOMIAL_WITH:
            updated = value >= 0

        if updated:
            if is_r:
                self.function_parameter_r = value
            else:
                self.function_parameter_l = value
        return updated

    def _mirror_parameter(self, source_is_r: bool) -> None:
        source_type = self.function_type_r if source_is_r else self.function_type_l
        source_value = self.function_parameter_r if source_is_r else self.function_parameter_l
        if source_type in (NumberConfig.SMOOTH_STEP_AT, NumberConfig.STEP_AT):
            if self.distance_function == DistanceConfig.DIFFERENCE:
                mirrored = -source_value
            else:
                mirrored = 1 / source_value
        else:
            mirrored = source_value
        if source_is_r:
            self.function_type_l = source_type
            self.function_parameter_l = mirrored
        else:
            self.function_type_r = source_type
            self.function_parameter_r = mirrored

    def get_function_parameter_r(self) -> float:
        return self.function_parameter_r

    def get_function_parameter_l(self) -> float:
        return self.function_parameter_l

    def set_function_type_r(self, function_type: NumberConfig) -> None:
        self.function_parameter_r = self._reset_param(self.function_type_r, function_type)
        self.function_type_r = function_type
        if self.is_symmetric:
            self._mirror_parameter(source_is_r=True)

    def set_function_type_l(self, function_type: NumberConfig) -> None:
        self.function_parameter_l = self._reset_param(self.function_type_l, function_type)
        self.function_type_l = function_type
        if self.is_symmetric:
            self._mirror_parameter(source_is_r=False)

    def _reset_param(self, old_type: NumberConfig, new_type: NumberConfig) -> float:
        if old_type in (NumberConfig.POLYNOMIAL_WITH, NumberConfig.CONSTANT):
            if new_type == NumberConfig.STEP_AT:
                return self._STEP_AT_VALUE
            if new_type == NumberConfig.SMOOTH_STEP_AT:
                return self._SMOOTH_STEP_AT_VALUE
        elif old_type in (NumberConfig.STEP_AT, NumberConfig.SMOOTH_STEP_AT):
            if new_type == NumberConfig.POLYNOMIAL_WITH:
                return self._POLYNOMIAL_WITH_VALUE
            if new_type == NumberConfig.CONSTANT:
                return self._CONSTANT_VALUE
        return self.function_parameter_r if old_type == self.function_type_r else self.function_parameter_l

    def get_function_type_l(self) -> NumberConfig:
        return self.function_type_l

    def get_function_type_r(self) -> NumberConfig:
        return self.function_type_r

    def fits_distance(self, x: float) -> bool:
        if self.distance_function == DistanceConfig.DIFFERENCE:
            return (self.min - self.max) < x < (self.max - self.min)
        return x > 0

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.double_desc import DoubleDesc
        from hklii_psla.cbr.core.project import Project

        if isinstance(new_desc, DoubleDesc) and self.name != Project.DEFAULT_FCT_NAME:
            f = new_desc.add_double_fct(self.name, is_active)
            f.function_parameter_l = self.function_parameter_l
            f.function_parameter_r = self.function_parameter_r
            f.function_type_l = self.function_type_l
            f.function_type_r = self.function_type_r
            f.is_symmetric = self.is_symmetric
            f.mc = self.mc
            f.distance_function = self.distance_function
            f.max_for_quotient = self.max_for_quotient
