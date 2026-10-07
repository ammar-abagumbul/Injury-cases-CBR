"""Experiment 7 — feature-query CBR retrieval over the extracted corpus.

A judge supplies a set of case features (injuries, losses, demographics); the
system retrieves the most similar cases from the corpus via case-based
reasoning, with similarity weights tuned so the retrieved cases also have PSLA
compensations close to one another.  See ``PLAN.md`` in this folder.
"""

from .cbr_model import (
    CorpusCBRModel,
    build_corpus_model,
    clinical_gate_mask,
    composite_similarities,
)
from .evaluation import (
    ClinicalRelevance,
    QueryEvaluation,
    clinical_relevance,
    evaluate_query,
    summarise_evaluations,
)
from .exp7 import Exp7Config, FeatureQueryRetrievalExperiment
from .features import (
    CorpusCase,
    load_corpus,
    severity_from_award,
)
from .price_index import PriceIndex
from .query import FeatureQuery, load_query
from .retrieval import CBRFeatureRetriever, RankedCase, RetrievalResult

__all__ = [
    "CBRFeatureRetriever",
    "ClinicalRelevance",
    "CorpusCBRModel",
    "CorpusCase",
    "Exp7Config",
    "FeatureQuery",
    "FeatureQueryRetrievalExperiment",
    "PriceIndex",
    "QueryEvaluation",
    "RankedCase",
    "RetrievalResult",
    "build_corpus_model",
    "clinical_gate_mask",
    "clinical_relevance",
    "composite_similarities",
    "evaluate_query",
    "load_corpus",
    "load_query",
    "severity_from_award",
    "summarise_evaluations",
]
