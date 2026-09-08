from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, override

from hklii_psla.cbr.core.similarity.config.distance_config import DistanceConfig
from hklii_psla.cbr.core.similarity.number_fct import NumberFct
from hklii_psla.cbr.core.similarity.similarity import Similarity

if TYPE_CHECKING:
    from hklii_psla.cbr.core.casebase.attribute import Attribute
    from hklii_psla.cbr.core.model.attribute_desc import AttributeDesc
    from hklii_psla.cbr.core.model.double_desc import DoubleDesc
    from hklii_psla.cbr.core.project import Project


class AdvancedDoubleFct(NumberFct):
    """Similarity is the piecewise-linear interpolation of a set of
    (distance, similarity) points, sorted by distance. Also usable for
    ordered-symbol similarity via the internal integer distance."""

    def __init__(self, prj: Project, desc: DoubleDesc, name: str):
        super().__init__(prj, desc, name)
        self.sub_desc = desc
        self.min_point = Similarity.get(0.00)
        self.zero_point = Similarity.get(1.00)
        self.max_point = Similarity.get(0.00)
        self.max = desc.max
        self.min = desc.min
        self.diff = self.max - self.min
        self.max_for_quotient = 10.0

        self._points: dict[float, Similarity] = {
            -self.diff: self.min_point,
            0.0: self.zero_point,
            self.diff: self.max_point,
        }

    def get_additional_points(self) -> dict[float, Similarity]:
        return self._points

    def add_additional_point(self, key: float, value: Similarity) -> None:
        if self.fits_distance(key):
            if self.distance_function == DistanceConfig.QUOTIENT and key >= self.max_for_quotient:
                return
            self._points[key] = value

    @override
    def calculate_similarity(self, a1: Attribute, a2: Attribute) -> Similarity:
        from hklii_psla.cbr.core.casebase.double_attribute import DoubleAttribute
        from hklii_psla.cbr.core.casebase.multiple_attribute import MultipleAttribute
        from hklii_psla.cbr.core.casebase.special_attribute import SpecialAttribute

        if isinstance(a1, SpecialAttribute) or isinstance(a2, SpecialAttribute):
            return self.prj.calculate_special_similarity(a1, a2)
        if isinstance(a1, MultipleAttribute) and isinstance(a2, MultipleAttribute):
            return self.prj.calculate_multiple_attribute_similarity(self, a1, a2)
        if isinstance(a1, DoubleAttribute) and isinstance(a2, DoubleAttribute):
            if a1.attribute_desc is not self.desc or a2.attribute_desc is not self.desc:
                return Similarity.INVALID_SIM
            return self.calculate_similarity_values(a1.value, a2.value)
        return Similarity.INVALID_SIM

    def calculate_similarity_values(self, q: float, c: float) -> Similarity:
        d = (c / q) if self.distance_function == DistanceConfig.QUOTIENT else (c - q)

        if self.distance_function == DistanceConfig.QUOTIENT and c / q >= self.max_for_quotient:
            return Similarity.get(0.0)

        exact = self._points.get(d)
        if exact is not None:
            return exact

        keys = sorted(self._points)
        idx = bisect.bisect_left(keys, d)
        if idx == 0:
            right_key = keys[0]
            left_key = right_key
        elif idx >= len(keys):
            left_key = keys[-1]
            right_key = left_key
        else:
            left_key = keys[idx - 1]
            right_key = keys[idx]

        left_sim = self._points[left_key].value
        right_sim = self._points[right_key].value

        if left_key == right_key:
            return Similarity.get(left_sim)

        x1, x2 = left_key, right_key
        y1, y2 = left_sim, right_sim
        result = (y2 - y1) / (x2 - x1) * d + (x2 * y1 - x1 * y2) / (x2 - x1)
        return Similarity.get(result)

    def set_max_for_quotient(self, max_value: float) -> None:
        if max_value > 0:
            self.max_for_quotient = max_value
            for p in [p for p in self._points if p > max_value]:
                del self._points[p]

    def fits_distance(self, x: float) -> bool:
        if self.distance_function == DistanceConfig.DIFFERENCE:
            return (self.min - self.max) < x < (self.max - self.min)
        return x > 0

    def _update_points(self) -> None:
        self._points = {k: v for k, v in self._points.items() if self.fits_distance(k)}

    @override
    def clone(self, new_desc: AttributeDesc, is_active: bool) -> None:
        from hklii_psla.cbr.core.model.double_desc import DoubleDesc
        from hklii_psla.cbr.core.project import Project

        if isinstance(new_desc, DoubleDesc) and self.name != Project.DEFAULT_FCT_NAME:
            f = new_desc.add_advanced_double_fct(self.name, is_active)
            f.distance_function = self.distance_function
            f._points = dict(self._points)
            f.zero_point = self.zero_point
            f.is_symmetric = self.is_symmetric
            f.max_for_quotient = self.max_for_quotient
