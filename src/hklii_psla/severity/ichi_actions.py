"""Severity/invasiveness tiers for ICHI's 120 ACTION codes.

WHO defines what these 120 actions *are*; it does not rank how invasive
one is relative to another. This table is that ranking, and it is a
judgment call — reviewed against each code's real WHO title (see the
reconnaissance in this module's git history / conversation), but a
starting point to correct, not a fact pulled from the ICHI data itself.

``weight`` is what a classified item contributes to ``acute_intervention``
(or ``recovery_burden``/``permanence`` per ``axis``) before the usual
``max(structured, this)`` blending in ``treatment_complexity.py``.
``implies_permanence`` additionally feeds the permanence axis regardless
of the action's own primary axis — e.g. "Implantation" is coded axis=
"acute" (it's an acute intervention event) but also flags permanence,
since implanted devices mean ongoing/lifelong impact.

Codes that never realistically appear in a personal-injury treatment
narrative (policy/public-health/advocacy actions) are included for
completeness with weight 0.0 rather than omitted, so an unrecognised
action never crashes the lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ActionTier(str, Enum):
    CRITICAL = "critical"  # life-support / catastrophic-adjacent
    MAJOR_SURGICAL = "major_surgical"
    PROCEDURAL = "procedural"  # minor/percutaneous procedures, injections, drainage
    DIAGNOSTIC_MONITORING = "diagnostic_monitoring"
    CONSERVATIVE_SUPPORTIVE = "conservative_supportive"  # rehab, education, counselling
    NON_CLINICAL = "non_clinical"  # policy/public-health; not expected in these personal injury judgments
    UNKNOWN = "unknown"


Axis = Literal["acute", "recovery", "permanence"]


@dataclass(frozen=True)
class IchiActionInfo:
    code: str
    label: str
    tier: ActionTier
    weight: float
    axis: Axis
    implies_permanence: bool = False


def _a(code: str, label: str, tier: ActionTier, weight: float, axis: Axis = "acute", implies_permanence: bool = False) -> IchiActionInfo:
    return IchiActionInfo(code=code, label=label, tier=tier, weight=weight, axis=axis, implies_permanence=implies_permanence)


_T = ActionTier

ICHI_ACTIONS: dict[str, IchiActionInfo] = {
    a.code: a for a in [
        # -- CRITICAL: life-support / catastrophic-adjacent --------------
        _a("KD", "Transplantation", _T.CRITICAL, 1.00, implies_permanence=True),
        _a("MC", "Open chest cardiac massage", _T.CRITICAL, 1.00),
        _a("DE", "Artificial ventilation", _T.CRITICAL, 1.00, implies_permanence=True),
        _a("TH", "Post mortem care", _T.CRITICAL, 1.00),
        _a("DN", "Implantation of device", _T.CRITICAL, 0.95, implies_permanence=True),
        _a("JH", "Procurement (organ, for transplant)", _T.CRITICAL, 0.90),
        _a("JL", "Radical excision / exenteration", _T.CRITICAL, 0.90),
        _a("MA", "Construction (major reconstructive procedure)", _T.CRITICAL, 0.85),
        _a("JN", "Disarticulation / radical resection", _T.CRITICAL, 0.85),
        _a("JF", "Delivery by caesarean section", _T.CRITICAL, 0.85),
        _a("JC", "Therapeutic apheresis", _T.CRITICAL, 0.80),
        _a("DG", "Transfusion of blood or blood products", _T.CRITICAL, 0.75),
        _a("SH", "Medical induction of labour", _T.CRITICAL, 0.70),
        _a("SI", "Failed instrumental delivery", _T.CRITICAL, 0.70),
        _a("DD", "Super saturated oxygen therapy", _T.CRITICAL, 0.70),

        # -- MAJOR_SURGICAL ------------------------------------------------
        _a("ML", "Reconstruction", _T.MAJOR_SURGICAL, 0.80),
        _a("KA", "Replacement (device/graft)", _T.MAJOR_SURGICAL, 0.80, implies_permanence=True),
        _a("JJ", "Partial excision", _T.MAJOR_SURGICAL, 0.75),
        _a("JK", "Total excision", _T.MAJOR_SURGICAL, 0.80),
        _a("GA", "Destruction / ablation", _T.MAJOR_SURGICAL, 0.75),
        _a("MK", "Repair", _T.MAJOR_SURGICAL, 0.70),
        _a("LE", "Arthrodesis / fusion", _T.MAJOR_SURGICAL, 0.75, implies_permanence=True),
        _a("LB", "Anastomosis", _T.MAJOR_SURGICAL, 0.75),
        _a("JD", "Removal (of implanted device/organ)", _T.MAJOR_SURGICAL, 0.70),
        _a("JI", "Local excision", _T.MAJOR_SURGICAL, 0.65),
        _a("FA", "Incision", _T.MAJOR_SURGICAL, 0.65),
        _a("JE", "Extraction (e.g. endarterectomy)", _T.MAJOR_SURGICAL, 0.65),
        _a("LI", "Percutaneous transluminal bypass", _T.MAJOR_SURGICAL, 0.65),
        _a("LJ", "Transfer (surgical, of tissue/organ)", _T.MAJOR_SURGICAL, 0.65),
        _a("FB", "Division", _T.MAJOR_SURGICAL, 0.60),
        _a("LL", "Reduction / plication", _T.MAJOR_SURGICAL, 0.60),
        _a("LK", "Augmentation / enlargement", _T.MAJOR_SURGICAL, 0.60, implies_permanence=True),
        _a("AE", "Surgical exploration", _T.MAJOR_SURGICAL, 0.55),
        _a("DH", "Reproductive transfer procedure", _T.MAJOR_SURGICAL, 0.55),

        # -- PROCEDURAL: minor/percutaneous procedures ---------------------
        _a("JB", "Drainage", _T.PROCEDURAL, 0.55),
        _a("JG", "Debridement", _T.PROCEDURAL, 0.50),
        _a("LG", "Percutaneous transluminal dilatation", _T.PROCEDURAL, 0.50),
        _a("LH", "Percutaneous dilatation with device insertion", _T.PROCEDURAL, 0.55, implies_permanence=True),
        _a("DL", "Insertion (catheter/line/temporary device)", _T.PROCEDURAL, 0.50, implies_permanence=True),
        _a("LA", "Occlusion", _T.PROCEDURAL, 0.50),
        _a("LC", "Application / fixation", _T.PROCEDURAL, 0.45),
        _a("LD", "Repositioning", _T.PROCEDURAL, 0.40),
        _a("FC", "Release (e.g. of adhesions)", _T.PROCEDURAL, 0.50),
        _a("AD", "Biopsy", _T.PROCEDURAL, 0.45),
        _a("DB", "Injection (therapeutic, e.g. thrombolysis)", _T.PROCEDURAL, 0.45),
        _a("SC", "Phototherapy / cardioversion", _T.PROCEDURAL, 0.45),
        _a("SN", "Management of implanted device", _T.PROCEDURAL, 0.40, implies_permanence=True),
        _a("SM", "Management of prosthetics", _T.PROCEDURAL, 0.30, implies_permanence=True),
        _a("SO", "Removal (of external prosthesis)", _T.PROCEDURAL, 0.30),
        _a("KB", "Change/replacement of external immobilisation device", _T.PROCEDURAL, 0.25),
        _a("JA", "Irrigation", _T.PROCEDURAL, 0.30),
        _a("DA", "Intracavity administration", _T.PROCEDURAL, 0.35),
        _a("DC", "Injection of anaesthetic", _T.PROCEDURAL, 0.25),
        _a("TE", "Detoxification", _T.PROCEDURAL, 0.40),
        _a("SJ", "Manufacturing of assistive products/technology", _T.PROCEDURAL, 0.20),

        # -- DIAGNOSTIC_MONITORING ------------------------------------------
        _a("BA", "Imaging (radiography/angiography/MRI/CT/ultrasound)", _T.DIAGNOSTIC_MONITORING, 0.30),
        _a("AA", "Clinical assessment", _T.DIAGNOSTIC_MONITORING, 0.25),
        _a("AC", "Diagnostic test", _T.DIAGNOSTIC_MONITORING, 0.25),
        _a("AM", "Clinical observation", _T.DIAGNOSTIC_MONITORING, 0.25),
        _a("AI", "Continuous monitoring", _T.DIAGNOSTIC_MONITORING, 0.30),
        _a("AB", "Measurement", _T.DIAGNOSTIC_MONITORING, 0.20),
        _a("AH", "Specimen collection", _T.DIAGNOSTIC_MONITORING, 0.15),
        _a("AF", "Exercise/stress test", _T.DIAGNOSTIC_MONITORING, 0.20),
        _a("AN", "Clinical interview", _T.DIAGNOSTIC_MONITORING, 0.15),

        # -- CONSERVATIVE_SUPPORTIVE: rehab, education, counselling ---------
        _a("PH", "Training", _T.CONSERVATIVE_SUPPORTIVE, 0.15, axis="recovery"),
        _a("PM", "Health education", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("PN", "Advising", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("PP", "Counselling", _T.CONSERVATIVE_SUPPORTIVE, 0.15, axis="recovery"),
        _a("PQ", "Psychotherapy", _T.CONSERVATIVE_SUPPORTIVE, 0.20, axis="recovery"),
        _a("PG", "Assisted exercise / physiotherapy", _T.CONSERVATIVE_SUPPORTIVE, 0.20, axis="recovery"),
        _a("PA", "Manipulation", _T.CONSERVATIVE_SUPPORTIVE, 0.20, axis="recovery"),
        _a("PB", "Mobilisation", _T.CONSERVATIVE_SUPPORTIVE, 0.20, axis="recovery"),
        _a("PC", "Massage", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("PD", "Stretch therapy", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("PJ", "Deconditioning from assistive products", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("RA", "Task-performance assistance", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("RB", "Practical support", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("RC", "Emotional support", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("RD", "Providing assistive goods", _T.CONSERVATIVE_SUPPORTIVE, 0.15, axis="recovery", implies_permanence=True),
        _a("RE", "Peer support", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("RF", "Facilitating participation", _T.CONSERVATIVE_SUPPORTIVE, 0.05, axis="recovery"),
        _a("RG", "Providing support services", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("DK", "Wound dressing", _T.CONSERVATIVE_SUPPORTIVE, 0.15, axis="recovery"),
        _a("DP", "Strapping / taping", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("SB", "Acupressure", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("SD", "Warming / hyperthermy for pain", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("SE", "Cooling / hypothermy for pain", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("SP", "Acupuncture", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("SQ", "Moxibustion", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("SR", "Cupping", _T.CONSERVATIVE_SUPPORTIVE, 0.10, axis="recovery"),
        _a("TI", "Prescription", _T.CONSERVATIVE_SUPPORTIVE, 0.15, axis="recovery"),
        _a("VD", "Vaccination", _T.CONSERVATIVE_SUPPORTIVE, 0.05, axis="recovery"),
        _a("TJ", "Transporting person", _T.CONSERVATIVE_SUPPORTIVE, 0.05, axis="recovery"),

        # -- NON_CLINICAL: policy/public-health, not expected in PI cases ---
        *[
            _a(code, label, _T.NON_CLINICAL, 0.0, axis="recovery")
            for code, label in [
                ("WI", "Restrictions (policy)"), ("WG", "Economic incentives (policy)"),
                ("WJ", "Policy change"), ("WC", "Restrictions (policy)"),
                ("WD", "Restrictions (policy)"), ("WE", "Restrictions (policy)"),
                ("WF", "Restrictions (policy)"), ("WB", "Restricting movement of people"),
                ("VC", "Public health surveillance"), ("VB", "Awareness raising"),
                ("VA", "Capacity building"), ("VO", "Marshalling health services"),
                ("VP", "Improving access to services"), ("VN", "Establishing health services"),
                ("VL", "Environmental remediation"), ("VR", "Reducing emissions (policy)"),
                ("VE", "Infection control measures (population-level)"),
                ("VI", "Fortification (public health)"),
                ("TA", "Advocacy"), ("TD", "Collaboration / partnerships"),
                ("TM", "Environment modification (population-level)"),
                ("TB", "Planning (public health)"), ("TK", "Public infrastructure development"),
                ("TC", "Navigating the service system"),
                ("VF", "Alert (public health)"),
            ]
        ],

        # -- residual / catch-all ---------------------------------------
        _a("ZZ", "Other/unspecified intervention", _T.UNKNOWN, 0.30),
        _a("ZY", "Other specified intervention", _T.UNKNOWN, 0.30),
    ]
}

# Sentinel the LLM can return when no action code genuinely fits.
NO_MATCH = "NO_MATCH"


def get_action(code: str) -> IchiActionInfo | None:
    return ICHI_ACTIONS.get(code)
