"""Treatment-complexity-as-severity-proxy scoring.

Severity comes from ICHI-classified treatment items (see
``treatment_narrative.py``), plus each injury's own ``injury_type`` flag.
There is deliberately no numeric-field (``hospitalisation_days``,
``operations_count``, ``sick_leave_days``...) or keyword-based scoring here
any more — that path was unconvincing: numeric fields are sparse in real
extracted cases, and keyword matching can't distinguish "no surgery was
needed" from "surgery was performed". ``injury_type`` stays because it's a
different kind of signal — a categorical, already-extracted field, not a
day-count heuristic — and dropping it would lose a clean signal for no
reason.

Without a narrative assessment (``llm`` not supplied to ``assess_case``, or
nothing in ``treatments_received``/``treatments_future`` to read),
``acute_intervention``/``recovery_burden`` are simply unscored (0.0) — that
means "not assessed", not "confirmed mild". Check
``TreatmentComplexityScore.data_completeness`` / ``.notes`` before trusting
a low composite.
"""

from __future__ import annotations

from hklii_psla.schemas import Injury, InjuryCategory
from hklii_psla.severity.ichi_actions import ICHI_ACTIONS
from hklii_psla.severity.models import SeverityConfig, TreatmentComplexityScore
from hklii_psla.severity.treatment_narrative import TreatmentNarrativeAssessment

_PERMANENT_INJURY_TYPES = {"Permanent injury", "Residual disability"}


def _narrative_axis_score(narrative: TreatmentNarrativeAssessment, axis: str) -> float:
    """Aggregate the classified items' weights for one axis by taking the
    single most severe item, not an average.

    Averaging was tried first and produces the wrong answer: a case with
    one near-maximal item (e.g. "Artificial ventilation", weight 1.00)
    alongside several routine follow-up items (e.g. "Assessment", weight
    0.25) got its acute score diluted to ~0.66 — below the point where it
    should read as severe at all. A patient who had one catastrophic
    emergency procedure plus routine follow-ups is not less severe than one
    who had only the procedure; the follow-ups shouldn't drag the score
    down. Verified against a real case where switching mean -> max moved
    the composite from 0.65 to 0.76, correctly crossing into
    ``Gross disability`` and matching the judgment's own stated category.

    Each item's weight/axis comes from its classified ICHI action code
    (see ``ichi_actions.py``), not a freeform LLM number. For permanence
    specifically, two independent signals can push the score up even if
    the item's primary axis is something else: the LLM's own
    ``implies_ongoing_or_permanent`` judgment, and the action code's own
    ``implies_permanence`` hint (e.g. "Implantation"/"Replacement" imply
    lasting impact regardless of which axis the action itself sits in).
    """
    axis_weights = [item.weight for item in narrative.items if item.axis == axis]
    score = max(axis_weights) if axis_weights else 0.0

    if axis == "permanence":
        permanence_weights = [
            item.weight
            for item in narrative.items
            if item.implies_ongoing_or_permanent or item.action_info.implies_permanence
        ]
        if permanence_weights:
            score = max(score, max(permanence_weights))

    return score


def score_treatment_complexity(
    injuries: list[Injury],
    config: SeverityConfig | None = None,
    narrative: TreatmentNarrativeAssessment | None = None,
) -> TreatmentComplexityScore:
    """Compute a decomposed, [0,1]-normalised treatment-complexity score.

    ``injuries`` is only used for the permanence component (checking
    ``Injury.injury_type``). ``narrative`` (see ``treatment_narrative.py``)
    is the LLM's classification of each free-text treatment item into an
    ICHI action code, and is the sole source of ``acute_intervention``/
    ``recovery_burden``.
    """
    config = config or SeverityConfig()
    notes: list[str] = []

    has_permanent_injury = any(inj.injury_type in _PERMANENT_INJURY_TYPES for inj in injuries)
    injury_type_signal = 1.0 if has_permanent_injury else 0.0
    if has_permanent_injury:
        notes.append("permanence: at least one injury marked Permanent/Residual disability")

    if narrative is None or not narrative.items:
        acute = 0.0
        recovery = 0.0
        permanence = injury_type_signal
        completeness = 0.0
        notes.append(
            "no ICHI treatment classification available (llm not supplied to assess_case, "
            "or nothing in treatments_received/treatments_future to read) — "
            "acute_intervention/recovery_burden are UNSCORED, not confirmed low"
        )
    else:
        acute = _narrative_axis_score(narrative, "acute")
        recovery = _narrative_axis_score(narrative, "recovery")
        permanence = max(_narrative_axis_score(narrative, "permanence"), injury_type_signal)

        matched = sum(1 for item in narrative.items if item.ichi_action_code in ICHI_ACTIONS)
        completeness = matched / len(narrative.items)
        if matched < len(narrative.items):
            notes.append(
                f"{len(narrative.items) - matched}/{len(narrative.items)} treatment item(s) "
                "did not resolve to a known ICHI action code"
            )

    weights = (
        config.weight_acute_intervention,
        config.weight_recovery_burden,
        config.weight_permanence,
    )
    values = (acute, recovery, permanence)
    weight_sum = sum(weights)
    composite = sum(w * v for w, v in zip(weights, values)) / weight_sum if weight_sum else 0.0

    return TreatmentComplexityScore(
        acute_intervention=acute,
        recovery_burden=recovery,
        permanence=permanence,
        composite=composite,
        data_completeness=completeness,
        notes=notes,
    )


def classify_severity(
    composite: float,
    config: SeverityConfig | None = None,
    llm_category: InjuryCategory | None = None,
) -> InjuryCategory:
    """Map a composite complexity score to the HK PI severity categories.

    The composite is derived from actual classified ICHI items — trust it
    for everything except the top tier. ``llm_category`` (the holistic
    judgment from ``TreatmentNarrativeAssessment.overall_severity_category``)
    is a separate, unconstrained LLM read of the whole narrative, not
    derived from the composite at all, so it is used *only* to escalate to
    ``InjuryCategory.DISASTER`` — never to override any other tier. Letting
    it override the full scale would make the composite non-monotonic with
    the final category (a lower composite could outrank a higher one purely
    because the LLM's separate holistic call happened to be more dramatic
    for that case) — verified this actually happening on real output before
    narrowing the override to Disaster-only.
    """
    config = config or SeverityConfig()
    if composite < config.threshold_serious:
        rule_based = InjuryCategory.NON_SERIOUS
    elif composite < config.threshold_substantial:
        rule_based = InjuryCategory.SERIOUS
    elif composite < config.threshold_gross_disability:
        rule_based = InjuryCategory.SUBSTANTIAL
    else:
        rule_based = InjuryCategory.GROSS_DISABILITY

    if llm_category is InjuryCategory.DISASTER:
        return InjuryCategory.DISASTER
    return rule_based
