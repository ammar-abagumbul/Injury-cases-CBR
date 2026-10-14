"""Experiment 8 — compact, anonymised text "cards" of extracted cases for the LLM.

A card is everything a person would need to judge severity: injuries, treatment, hospital and
sick-leave figures, manifestations and losses. Deliberately left out, because they would leak
the answer or the court: PSLA awards, comparable cases, the LLM's own ``overall_category``,
citation, case name, court and date. Cards carry a neutral id (``C017``) instead of the filename.
"""

from __future__ import annotations

from hklii_psla.schemas import Case, TreatmentItem

_MAX_INJURIES = 15
_MAX_MANIFESTATIONS = 10
_MAX_LOSSES = 10


def _v(x) -> str:
    return str(getattr(x, "value", x))


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _treatment(t: TreatmentItem) -> str:
    if not t.is_surgical:
        return _clip(t.treatment, 110)
    if t.vmis_category is not None:
        return f"{_clip(t.treatment, 90)} [surgery, VMIS tier {_v(t.vmis_category)}]"
    return f"{_clip(t.treatment, 90)} [surgery, not in VMIS schedule]"


def _days(label: str, value: int | None) -> str:
    return f"{label}: {value}" if value is not None else f"{label}: not stated"


def build_card(case: Case, card_id: str) -> str:
    p, tr = case.plaintiff, case.treatment
    lines = [f"CASE {card_id}"]

    who = [x for x in (_v(p.gender) if p.gender else None,
                       f"age {p.age_at_accident} at accident" if p.age_at_accident is not None else None,
                       _clip(p.occupation_before, 50) if p.occupation_before else None) if x]
    if who:
        lines.append("Plaintiff: " + ", ".join(who))
    if p.salary_before is not None or p.salary_after is not None:
        lines.append(f"Monthly salary before / after: {p.salary_before if p.salary_before is not None else '?'}"
                     f" / {p.salary_after if p.salary_after is not None else '?'}")
    if case.metadata.has_pre_existing_injuries:
        lines.append("Pre-existing injuries: yes")

    injuries = case.injuries.injuries
    lines.append(f"Injuries ({len(injuries)}):")
    for i in injuries[:_MAX_INJURIES]:
        icd = f" [ICD-11: {_clip(i.icd_description, 60)}]" if i.icd_description else ""
        side = f", {i.laterality}" if i.laterality else ""
        lines.append(f"- {_clip(i.description, 110)} ({_v(i.injury_type)}{side}){icd}")
    if len(injuries) > _MAX_INJURIES:
        lines.append(f"- ... and {len(injuries) - _MAX_INJURIES} more")

    if case.injuries.manifestations:
        lines.append("Symptoms and functional effects:")
        for m in case.injuries.manifestations[:_MAX_MANIFESTATIONS]:
            lines.append(f"- {_v(m.manifestation_type).replace('_', ' ')}: {_clip(m.description, 90)}")

    if tr.treatments_received:
        lines.append("Treatment received: " + "; ".join(_treatment(t) for t in tr.treatments_received))
    if tr.treatments_future:
        lines.append("Treatment expected in future: " + "; ".join(_treatment(t) for t in tr.treatments_future))
    lines.append(" | ".join([
        _days("Hospital days", tr.hospitalisation_days), _days("Operations", tr.operations_count),
        _days("Sick leave days", tr.sick_leave_days_actual),
    ]))
    if tr.expected_hospitalisation_days is not None or tr.future_operations_count is not None:
        lines.append(f"Expected future hospital days: {tr.expected_hospitalisation_days}; "
                     f"future operations: {tr.future_operations_count}")

    losses = case.losses.losses
    if losses:
        lines.append("Losses claimed:")
        for loss in losses[:_MAX_LOSSES]:
            lines.append(f"- {_v(loss.category)}: {_clip(loss.description, 80)}")
    return "\n".join(lines)
