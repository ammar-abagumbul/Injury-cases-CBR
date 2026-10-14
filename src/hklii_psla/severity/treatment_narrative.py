"""LLM-based reading of free-text treatment narrative.

This is now the primary source of treatment-complexity severity signal —
not a fallback. Numeric treatment fields (hospitalisation_days,
operations_count, ...) and keyword matching were dropped from
``treatment_complexity.py`` because they were unconvincing: numeric fields
are sparse in real extracted cases, and keyword matching can't distinguish
"no surgery was needed" from "surgery was performed".

Each free-text treatment item is classified into one of ICHI's 120 official
ACTION codes (see ``ichi_actions.py``) — a constrained, auditable task
("which of these ~120 known actions is this?" rather than "invent a
number"), with the severity weight coming from a fixed, hand-reviewed
lookup table instead of the LLM deciding a number itself.

Callers that don't pass an ``llm`` to ``assess_case`` never trigger this —
but without it, ``score_treatment_complexity`` has almost nothing left to
work with (only each injury's ``injury_type`` flag). ``llm`` is optional in
the type signature, but effectively required for a meaningful result.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from hklii_psla.schemas import InjuryCategory, Treatment
from hklii_psla.severity.ichi_actions import ICHI_ACTIONS, NO_MATCH, ActionTier, Axis, IchiActionInfo

logger = logging.getLogger(__name__)

_DEFAULT_UNMATCHED = IchiActionInfo(
    code=NO_MATCH, label="No matching action", tier=ActionTier.UNKNOWN, weight=0.0, axis="acute"
)


def _action_catalog_text() -> str:
    lines = [f"{code}: {info.label}" for code, info in sorted(ICHI_ACTIONS.items())]
    return "\n".join(lines)


_SYSTEM_PROMPT_TEMPLATE = (
    "You are classifying treatments described in a Hong Kong personal injury judgment "
    "using WHO's ICHI (International Classification of Health Interventions) ACTION codes.\n\n"
    "Please do not treat loss of earnings, or the money lost in treatment as a treatment."
    "Only consider these treatments that required hospital intervention or home care support."
    "For EACH treatment item listed, choose the single best-matching ACTION code from this "
    "list (do not invent codes; use '{no_match}' only if truly nothing fits):\n\n"
    "{catalog}\n\n"
    "Also assign implies_ongoing_or_permanent: true if the text implies the plaintiff will "
    "need this treatment, medication, or monitoring indefinitely / for the rest of their life "
    "(e.g. 'continues to take', 'lifelong', 'permanent', 'followed up every N months').\n\n"
    "Then give an overall_severity_category: your holistic judgment of the injury's "
    "severity tier based on reading ALL the treatments together, using EXACTLY one of:\n"
    "'Non-serious injury', 'Serious injury', 'Substantial injury', 'Gross disability', 'Disaster'.\n"
    "Only assign 'Disaster' when the treatment narrative clearly describes a catastrophic, "
    "life-threatening, or completely life-altering outcome (e.g. permanent ventilation, "
    "persistent vegetative state, quadriplegia, death). Leave overall_severity_category null "
    "if you are not confident enough to make that call from the treatment text alone.\n\n"
    "Use the exact verbatim text of each item for the 'text' field. Do not invent items."
)

_USER_TEMPLATE = (
    "Treatments received:\n{received}\n\n"
    "Treatments planned/expected in future:\n{future}\n"
)


class TreatmentSeverityItem(BaseModel):
    text: str = Field(description="Verbatim treatment text")
    ichi_action_code: str = Field(
        description="Best-matching ICHI ACTION code from the supplied catalog, or 'NO_MATCH'"
    )
    implies_ongoing_or_permanent: bool = Field(
        default=False,
        description="True if this implies an indefinite/lifelong ongoing condition",
    )

    @property
    def action_info(self) -> IchiActionInfo:
        """Resolved severity info for this item's action code.

        Falls back to a zero-weight placeholder for 'NO_MATCH' or any code
        the LLM hallucinated that isn't in the real catalog — never raises.
        """
        return ICHI_ACTIONS.get(self.ichi_action_code, _DEFAULT_UNMATCHED)

    @property
    def weight(self) -> float:
        return self.action_info.weight

    @property
    def axis(self) -> Axis:
        return self.action_info.axis


class TreatmentNarrativeAssessment(BaseModel):
    items: list[TreatmentSeverityItem] = Field(default_factory=list)
    overall_severity_category: InjuryCategory | None = Field(
        default=None,
        description=(
            "Holistic severity judgment from the treatment narrative alone; "
            "null if not confident enough to assign one."
        ),
    )
    reasoning: str | None = Field(
        default=None, description="Brief justification, mainly for audit/debugging"
    )


def assess_treatment_narrative(
    llm: BaseChatModel, treatment: Treatment
) -> TreatmentNarrativeAssessment | None:
    """Read ``treatment``'s free-text lists and classify each item into an
    ICHI ACTION code.

    Returns ``None`` if there's nothing to read (both lists empty) or if the
    LLM call fails — callers treat that the same as "no narrative signal
    available" rather than propagate the error.
    """
    if not treatment.treatments_received and not treatment.treatments_future:
        return None

    received = "\n".join(f"- {t.treatment}" for t in treatment.treatments_received) or "(none)"
    future = "\n".join(f"- {t.treatment}" for t in treatment.treatments_future) or "(none)"

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        no_match=NO_MATCH, catalog=_action_catalog_text()
    )

    try:
        structured_llm = llm.with_structured_output(TreatmentNarrativeAssessment)
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=_USER_TEMPLATE.format(received=received, future=future)),
        ])
        if isinstance(result, TreatmentNarrativeAssessment):
            return result
        return TreatmentNarrativeAssessment.model_validate(result)
    except Exception as e:
        # logger.warning alone isn't reliable here: importing create_model()
        # pulls in litellm, which reconfigures logging on import and can
        # swallow this. Print directly so a failure is never silent.
        message = f"Treatment narrative assessment failed: {type(e).__name__}: {e}"
        logger.warning(message)
        print(f"WARNING: {message}", file=sys.stderr)
        return None
