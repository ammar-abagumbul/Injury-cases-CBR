"""Tests for Experiment 7 — feature-query CBR retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from hklii_psla.experiments.experiment4.icd_similarity import IcdTaxonomy
from hklii_psla.experiments.experiment7.cbr_model import (
    AUXILIARY_ATTRIBUTES,
    CLINICAL_ATTRIBUTES,
    CORE_ATTRIBUTES,
    DEFAULT_ATTRIBUTES,
    available_case_similarities,
    build_corpus_model,
    clinical_gate_mask,
    composite_similarities,
)
from hklii_psla.experiments.experiment7.evaluation import (
    QueryEvaluation,
    aggregate_metrics,
    clinical_relevance,
    evaluate_query,
    spearman,
    summarise_evaluations,
)
from hklii_psla.experiments.experiment7.exp7 import FeatureQueryRetrievalExperiment
from hklii_psla.experiments.experiment7.features import (
    CorpusCase,
    extract_year,
    graded_relevance,
    is_relevant,
    log_ratio_error,
    severity_from_award,
)
from hklii_psla.experiments.experiment7.price_index import PriceIndex
from hklii_psla.experiments.experiment7.query import (
    FeatureQuery,
    load_query,
    resolve_injury_text,
)
from hklii_psla.experiments.experiment7.retrieval import CBRFeatureRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def price_index(tmp_path_factory) -> PriceIndex:
    path = tmp_path_factory.mktemp("price") / "price_index.csv"
    path.write_text(
        "Year,Price Index\n"
        "1969,8.31\n"
        "2000,78.30\n"
        "2010,82.10\n"
        "2020,111.00\n",
        encoding="utf-8",
    )
    return PriceIndex.from_csv(path)


@pytest.fixture(scope="module")
def taxonomy() -> IcdTaxonomy:
    return IcdTaxonomy.from_json()


def _case(
    cid: str,
    *,
    gender: str | None = "Male",
    age_acc: int | None = 40,
    age_trial: int | None = 45,
    injuries: tuple[str, ...] = ("NA02.01",),
    losses: tuple[str, ...] = ("Loss of mobility",),
    psla_real: float | None = 200_000.0,
    year: int | None = 2010,
    severity: str | None = "Serious injury",
    ops: int | None = 1,
    hosp: int | None = 10,
) -> CorpusCase:
    return CorpusCase(
        case_id=cid,
        gender=gender,
        age_at_accident=age_acc,
        age_at_trial=age_trial,
        injuries=injuries,
        losses=losses,
        psla_nominal=psla_real,
        year=year,
        psla_real=psla_real,
        overall_category=severity,
        operations_count=ops,
        hospitalisation_days=hosp,
        n_injuries=len(injuries),
        n_losses=len(losses),
    )


@pytest.fixture()
def small_cases() -> list[CorpusCase]:
    return [
        _case("A", injuries=("NA02.01",), losses=("Loss of mobility",), psla_real=100_000),
        _case(
            "B",
            injuries=("NA02.01",),
            losses=("Loss of mobility", "Loss of mental ability"),
            psla_real=110_000,
        ),
        _case(
            "C",
            injuries=("NC92.1",),
            losses=("Loss of ability to engage in sports and hobbies",),
            psla_real=800_000,
            severity="Disaster",
        ),
        _case(
            "D",
            injuries=("NB53.5",),
            losses=("Loss from scarring or disfigurement",),
            psla_real=90_000,
            gender="Female",
            severity="Non-serious injury",
        ),
    ]


# ---------------------------------------------------------------------------
# Price index
# ---------------------------------------------------------------------------
def test_price_index_deflates_to_base_year(price_index: PriceIndex):
    assert price_index.base_year == 2020
    # 1969 -> 2020 multiplies by 111/8.31
    real = price_index.to_real(100_000, 1969)
    assert real == pytest.approx(100_000 * 111.0 / 8.31)
    assert price_index.to_real(100_000, 2020) == pytest.approx(100_000)


def test_price_index_roundtrip_and_missing(price_index: PriceIndex):
    real = price_index.to_real(50_000, 2000)
    assert price_index.to_nominal(real, 2000) == pytest.approx(50_000)
    assert price_index.to_real(50_000, 1800) is None
    assert price_index.to_real(None, 2000) is None
    assert price_index.to_real(0, 2000) == 0.0


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_extract_year_prefers_judgment_date():
    assert extract_year({"judgment_date": "2011-03-04", "neutral_citation": "[2009] HKCFI 1"}) == 2011
    assert extract_year({"neutral_citation": "[2009] HKCFI 1"}) == 2009
    assert extract_year({}) is None


def test_relevance_and_graded_relevance():
    assert is_relevant(100_000, 140_000, 0.5)
    assert not is_relevant(100_000, 200_000, 0.5)
    assert graded_relevance(100_000, 100_000, 0.5) == pytest.approx(1.0)
    assert graded_relevance(100_000, 150_000, 0.5) == pytest.approx(0.0, abs=1e-9)
    assert log_ratio_error(100_000, 100_000) == pytest.approx(0.0)


def test_severity_from_award_bands():
    assert severity_from_award(None) is None
    assert severity_from_award(0) is None
    assert severity_from_award(100_000) == "Non-serious injury"
    # Boundaries are half-open: lower bound belongs to the higher band.
    assert severity_from_award(564_000) == "Serious injury"
    assert severity_from_award(760_999) == "Serious injury"
    assert severity_from_award(761_000) == "Substantial injury"
    assert severity_from_award(931_000) == "Gross disability"
    assert severity_from_award(1_410_000) == "Disaster"
    assert severity_from_award(2_000_000) == "Disaster"


def test_award_severity_overrides_extracted_label(price_index: PriceIndex):
    case = CorpusCase(
        case_id="X",
        gender="Male",
        age_at_accident=40,
        age_at_trial=45,
        injuries=("NA02.01",),
        losses=("Loss of mobility",),
        psla_real=200_000.0,
        overall_category="Disaster",  # noisy extractor label
    )
    # The award band is authoritative for a decided case.
    assert case.award_severity == "Non-serious injury"
    assert case.effective_severity == "Non-serious injury"
    no_award = CorpusCase(
        case_id="Y",
        gender="Male",
        age_at_accident=40,
        age_at_trial=45,
        injuries=(),
        losses=(),
        overall_category="Serious injury",
    )
    assert no_award.award_severity is None
    assert no_award.effective_severity == "Serious injury"


def test_corpus_case_from_dict(price_index: PriceIndex):
    data = {
        "metadata": {"neutral_citation": "[2010] HKDC 1", "judgment_date": "2010-05-05"},
        "plaintiff": {"gender": "Female", "age_at_accident": 30, "age_at_trial": 35},
        "injuries": {
            "injuries": [
                {"icd_code": "NA02.01"},
                {"icd_code": None},
                {"icd_code": "NA02.01"},
            ],
            "overall_category": "Serious injury",
        },
        "losses": {"losses": [{"category": "Loss of mobility"}]},
        "treatment": {"operations_count": 2, "hospitalisation_days": 7},
        "psla": {"amount": 100_000},
    }
    case = CorpusCase.from_dict(data, price_index)
    assert case.case_id == "[2010] HKDC 1"
    assert case.injuries == ("NA02.01",)
    assert case.losses == ("Loss of mobility",)
    assert case.year == 2010
    assert case.n_injuries == 3
    assert case.n_losses == 1
    assert case.overall_category == "Serious injury"
    assert case.psla_real == pytest.approx(100_000 * 111.0 / 82.10)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def test_feature_query_validates_losses():
    with pytest.raises(ValueError):
        FeatureQuery(losses=["Not a real loss category"])


def test_feature_query_resolves_injury_text(taxonomy: IcdTaxonomy):
    query = FeatureQuery(
        query_id="q",
        injury_descriptions=["fracture of the left tibia"],
        losses=["Loss of mobility"],
    )
    case = query.to_corpus_case(taxonomy)
    assert case.injuries, "expected at least one resolved ICD code"
    assert taxonomy.is_known(case.injuries[0])
    assert case.n_losses == 1


def test_load_query_roundtrip(tmp_path: Path):
    path = tmp_path / "q.yaml"
    path.write_text(
        yaml.safe_dump({"query_id": "x", "gender": "Male", "k": 5}),
        encoding="utf-8",
    )
    query = load_query(path)
    assert query.query_id == "x"
    assert query.k == 5


def test_resolve_injury_text_unknown(taxonomy: IcdTaxonomy):
    assert resolve_injury_text("", taxonomy) is None
    assert resolve_injury_text("zzz qqq", taxonomy) is None


# ---------------------------------------------------------------------------
# CBR model
# ---------------------------------------------------------------------------
def test_model_attributes_and_retrieval(small_cases: list[CorpusCase]):
    model = build_corpus_model(small_cases)
    assert model.attribute_names == DEFAULT_ATTRIBUTES
    ranked = model.retrieve(small_cases[0], exclude="A")
    ids = [cid for cid, _ in ranked]
    assert "A" not in ids
    # B shares injuries + losses with A, so should rank above C.
    assert ids.index("B") < ids.index("C")


def test_available_case_similarity_ignores_missing():
    S = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    known = np.array([[True, False], [False, False]])
    w = np.array([1.0, 1.0])
    out = available_case_similarities(S, known, w)
    # First pair has only the first attribute available -> similarity 1.0.
    assert out[0] == pytest.approx(1.0)
    # Second pair has no co-present attribute -> 0.0.
    assert out[1] == pytest.approx(0.0)


def test_per_attribute_shapes_and_known_mask(small_cases: list[CorpusCase]):
    model = build_corpus_model(small_cases)
    S, known = model.per_attribute_similarities(small_cases[:2])
    assert S.shape == (2, len(small_cases), len(DEFAULT_ATTRIBUTES))
    assert known.shape == S.shape
    assert S.dtype == np.float32
    assert float(S.min()) >= -1.0 and float(S.max()) <= 1.0


def test_available_case_matches_sequential_retrieval(small_cases: list[CorpusCase]):
    """With every attribute present, the linear available-case sum equals myCBR."""
    model = build_corpus_model(small_cases, missing_policy="available")
    S, known = model.per_attribute_similarities(
        [small_cases[0]], model.candidate_ids
    )
    weights = np.ones(len(model.attribute_names))
    linear = available_case_similarities(S[0], known[0], weights)
    by_id = {cid: float(linear[i]) for i, cid in enumerate(model.candidate_ids)}
    cbr = dict(model.retrieve_cbr(small_cases[0]))
    for cid, value in by_id.items():
        if cid == "A":
            continue
        assert value == pytest.approx(cbr[cid], abs=1e-6)


def test_default_attributes_are_clinical_first():
    assert set(CLINICAL_ATTRIBUTES) <= set(DEFAULT_ATTRIBUTES)
    assert set(AUXILIARY_ATTRIBUTES) <= set(DEFAULT_ATTRIBUTES)
    assert "gender" not in DEFAULT_ATTRIBUTES
    assert "overall_category" not in DEFAULT_ATTRIBUTES


def test_composite_is_clinical_first_and_penalises_missing():
    names = ("injuries", "losses", "age_at_trial")
    weights = np.ones(3)
    # P: strong clinical, no aux.  Q: no clinical, perfect aux.
    S = np.array([[0.9, 0.9, 0.0], [0.0, 0.0, 1.0]])
    known = np.array([[True, True, False], [True, True, True]])
    out = composite_similarities(S, known, weights, names, clinical_alpha=0.7)
    assert out[0] > out[1]
    assert out[0] == pytest.approx(0.7 * 0.9)
    assert out[1] == pytest.approx(0.3 * 1.0)

    # A clinical attribute missing on either side is penalised, not dropped.
    S2 = np.array([[0.0, 0.9, 0.0]])
    known2 = np.array([[False, True, False]])
    out2 = composite_similarities(S2, known2, weights, names, clinical_alpha=0.7)
    assert out2[0] == pytest.approx(0.7 * 0.9 / 2.0)


def test_clinical_gate_excludes_unrelated_candidates():
    names = ("injuries", "losses")
    S = np.array([[0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
    known = np.array([[True, True], [True, True], [False, False]])
    gate = clinical_gate_mask(
        S, known, names, min_injury_similarity=0.1, query_has_injuries=True
    )
    assert gate.tolist() == [True, False, False]
    # A query without injuries passes everyone.
    assert clinical_gate_mask(
        S, known, names, min_injury_similarity=0.1, query_has_injuries=False
    ).all()
    # A disabled threshold passes everyone.
    assert clinical_gate_mask(
        S, known, names, min_injury_similarity=None, query_has_injuries=True
    ).all()


def test_clinical_relevance_is_loose_and_graded():
    names = ("injuries", "losses")
    S = np.array([[0.8, 1.0], [0.05, 0.0]])
    known = np.ones((2, 2), dtype=bool)
    clinical = clinical_relevance(
        S, known, names, threshold=0.15, injury_weight=0.75
    )
    assert clinical.gains[0] == pytest.approx(0.85)
    assert bool(clinical.relevant[0])
    assert not bool(clinical.relevant[1])
    # A low injury match is still relevant when losses agree strongly.
    reinforced = clinical_relevance(
        np.array([[0.05, 1.0]]), np.ones((1, 2), dtype=bool), names
    )
    assert bool(reinforced.relevant[0])


def test_special_policy_runs(small_cases: list[CorpusCase]):
    model = build_corpus_model(small_cases, missing_policy="special")
    ranked = model.retrieve(small_cases[0], exclude="A")
    assert ranked
    assert all(0.0 <= s <= 1.0 for _, s in ranked)


def test_core_attributes_only(small_cases: list[CorpusCase]):
    model = build_corpus_model(small_cases, attributes=CORE_ATTRIBUTES)
    assert model.attribute_names == CORE_ATTRIBUTES


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def test_spearman_perfect_and_constant():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_evaluate_query_rewards_relevant_first(taxonomy: IcdTaxonomy):
    query = _case("Q", psla_real=100_000)
    pool = [
        _case("R1", psla_real=105_000),
        _case("R2", psla_real=95_000),
        _case("N1", psla_real=500_000),
        _case("N2", psla_real=600_000),
    ]
    similarities = [0.9, 0.8, 0.2, 0.1]
    evaluation = evaluate_query(
        query, pool, similarities, k=2, tolerance=0.5, taxonomy=taxonomy
    )
    assert evaluation.precision_at_k == pytest.approx(1.0)
    assert evaluation.recall_at_k == pytest.approx(1.0)
    assert evaluation.ndcg_at_k == pytest.approx(1.0)
    assert evaluation.within_tolerance_rate == pytest.approx(1.0)
    assert evaluation.n_relevant == 2


def test_evaluate_query_random_baseline(taxonomy: IcdTaxonomy):
    query = _case("Q", psla_real=100_000)
    pool = [_case(f"C{i}", psla_real=100_000 + i) for i in range(1, 11)]
    similarities = [1.0 / (i + 1) for i in range(10)]
    evaluation = evaluate_query(
        query, pool, similarities, k=3, tolerance=0.5, taxonomy=taxonomy
    )
    assert 0.0 <= evaluation.ndcg_at_k <= 1.0
    assert 0.0 <= evaluation.random_ndcg_at_k <= 1.0


def test_aggregate_metrics_matches_shape(taxonomy: IcdTaxonomy):
    q = 3
    c = 5
    G = np.random.default_rng(0).random((q, c))
    gains = np.zeros((q, c))
    relevant = np.zeros((q, c), dtype=bool)
    gains[:, 0] = 1.0
    relevant[:, 0] = True
    # put the relevant column first -> perfect NDCG
    out = aggregate_metrics(G, gains, relevant, k=2)
    assert "mean_ndcg_at_k" in out
    assert out["mean_n_relevant"] == pytest.approx(1.0)


def test_evaluate_query_reports_clinical_metrics(taxonomy: IcdTaxonomy):
    query = _case("Q", injuries=("NA02.01",), losses=("Loss of mobility",))
    pool = [
        _case("R", injuries=("NA02.01",), losses=("Loss of mobility",)),
        _case("N", injuries=("NC92.1",), losses=("Loss of independence",)),
    ]
    names = ("injuries", "losses")
    S = np.array([[1.0, 1.0], [0.0, 0.0]])
    known = np.ones((2, 2), dtype=bool)
    clinical = clinical_relevance(S, known, names)
    evaluation = evaluate_query(
        query, pool, [0.9, 0.1], k=1, tolerance=0.5, taxonomy=taxonomy,
        clinical=clinical,
    )
    assert evaluation.clinical_precision_at_k == pytest.approx(1.0)
    assert evaluation.clinical_ndcg_at_k == pytest.approx(1.0)
    assert evaluation.min_injury_similarity_at_k == pytest.approx(1.0)
    assert evaluation.loss_similarity_at_k == pytest.approx(1.0)


def test_aggregate_metrics_clinical(taxonomy: IcdTaxonomy):
    G = np.array([[0.9, 0.1], [0.1, 0.9]])
    gains = np.zeros((2, 2))
    relevant = np.zeros((2, 2), dtype=bool)
    names = ("injuries", "losses")
    S = np.array(
        [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]
    )
    known = np.ones_like(S, dtype=bool)
    clinical = clinical_relevance(S, known, names)
    out = aggregate_metrics(G, gains, relevant, k=1, clinical=clinical)
    assert out["mean_clinical_ndcg_at_k"] == pytest.approx(1.0)
    assert out["mean_min_injury_similarity_at_k"] == pytest.approx(1.0)


def test_summarise_evaluations():
    evals = [
        QueryEvaluation(
            query_id="a", k=10, n_candidates=100, n_relevant=5,
            psla_mae=1.0, psla_median_abs_error=1.0, psla_mape=0.1,
            psla_median_log_ratio=0.1, within_tolerance_rate=0.5,
            precision_at_k=0.5, recall_at_k=0.5, ndcg_at_k=0.5,
            random_precision_at_k=0.1, random_recall_at_k=0.1,
            random_ndcg_at_k=0.1, clinical_precision_at_k=0.6,
            clinical_recall_at_k=0.4, clinical_ndcg_at_k=0.7,
            injury_similarity_at_k=0.3, min_injury_similarity_at_k=0.1,
            loss_similarity_at_k=0.2, injury_overlap_at_k=0.2,
            loss_overlap_at_k=0.3, spearman_similarity_psla=0.1,
        )
    ]
    summary = summarise_evaluations(evals)
    assert summary["n_queries"] == 1
    assert summary["mean_ndcg_at_k"] == pytest.approx(0.5)
    assert summary["mean_clinical_ndcg_at_k"] == pytest.approx(0.7)
    assert summary["median_psla_mae"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Retrieval wrapper
# ---------------------------------------------------------------------------
def test_band_filtering_and_require_psla(small_cases: list[CorpusCase]):
    model = build_corpus_model(small_cases)
    retriever = CBRFeatureRetriever(model)
    band = (80_000, 120_000)
    ids = retriever.candidate_ids(band)
    assert set(ids) == {"A", "B", "D"}
    assert retriever.candidate_ids(require_psla=True) == ["A", "B", "C", "D"]


def test_retrieve_returns_ranked_cases(small_cases: list[CorpusCase]):
    model = build_corpus_model(small_cases)
    retriever = CBRFeatureRetriever(model)
    result = retriever.retrieve(small_cases[0], k=2, exclude="A")
    assert len(result.results) == 2
    assert result.case_ids == [r.case_id for r in result.results]


# ---------------------------------------------------------------------------
# End-to-end experiment
# ---------------------------------------------------------------------------
def _write_case(path: Path, cid: str, psla: float, *, injuries: list[str], losses: list[str], year: int = 2010):
    data = {
        "metadata": {"neutral_citation": cid, "judgment_date": f"{year}-01-01"},
        "plaintiff": {"gender": "Male", "age_at_accident": 40, "age_at_trial": 45},
        "injuries": {
            "injuries": [{"icd_code": c} for c in injuries],
            "overall_category": "Serious injury",
        },
        "losses": {"losses": [{"category": c} for c in losses]},
        "treatment": {"operations_count": 1, "hospitalisation_days": 5},
        "psla": {"amount": psla},
    }
    (path / f"{cid.replace(' ', '_').replace('[', '').replace(']', '')}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def test_experiment_end_to_end(tmp_path: Path, price_index_path: Path):
    corpus = tmp_path / "cases"
    corpus.mkdir()
    _write_case(corpus, "[2010] HKDC 1", 100_000, injuries=["NA02.01"], losses=["Loss of mobility"])
    _write_case(corpus, "[2010] HKDC 2", 110_000, injuries=["NA02.01"], losses=["Loss of mobility"])
    _write_case(corpus, "[2011] HKDC 3", 120_000, injuries=["NA02.01"], losses=["Loss of mobility"])
    _write_case(corpus, "[2012] HKDC 4", 800_000, injuries=["NC92.1"], losses=["Loss of independence"])
    _write_case(corpus, "[2013] HKDC 5", 90_000, injuries=["NB53.5"], losses=["Loss from scarring or disfigurement"])
    _write_case(corpus, "[2014] HKDC 6", 95_000, injuries=["NB53.5"], losses=["Loss from scarring or disfigurement"])

    config = {
        "id": "feature-query-retrieval",
        "output_dir": str(tmp_path / "out"),
        "case_base_dir": str(corpus),
        "price_index_path": str(price_index_path),
        "k": 2,
        "psla_tolerance": 0.5,
        "tune": True,
        "tune_trials": 3,
        "seed": 1,
        "max_queries": 4,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    experiment = FeatureQueryRetrievalExperiment(config_path)
    experiment.run()
    experiment.save_results()

    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "leave_one_out_benchmark"
    assert metrics["benchmark"]["n_queries"] >= 2
    assert "mean_ndcg_at_k" in metrics["benchmark"]["default"]
    assert (tmp_path / "out" / "per_query.csv").exists()
    assert (tmp_path / "out" / "tuned_weights.json").exists()


@pytest.fixture(scope="module")
def price_index_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("price2") / "price_index.csv"
    path.write_text(
        "Year,Price Index\n1969,8.31\n2000,78.30\n2010,82.10\n2020,111.00\n",
        encoding="utf-8",
    )
    return path
