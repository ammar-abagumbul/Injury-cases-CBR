"""Generate a worked example of the Experiment 7 fetch for documentation.

Picks a real corpus extraction as the query, retrieves its nearest cases with
the tuned clinical-first weights, copies the query and returned documents into
``demo/query`` / ``demo/results`` and writes ``demo/DEMO.md`` + ``demo/retrieval.json``.

Run:
    .venv/bin/python scripts/exp7_demo.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from hklii_psla.experiments.experiment7.cbr_model import (
    AUXILIARY_ATTRIBUTES,
    CLINICAL_ATTRIBUTES,
    build_corpus_model,
)
from hklii_psla.experiments.experiment7.features import (
    CorpusCase,
    load_corpus,
    log_ratio_error,
    relative_error,
)
from hklii_psla.experiments.experiment7.price_index import PriceIndex
from hklii_psla.experiments.experiment7.retrieval import CBRFeatureRetriever

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "output" / "experiments" / "corpus_extraction" / "cases"
BENCH_DIR = REPO / "output" / "experiments" / "feature_query_retrieval"
DEMO_DIR = BENCH_DIR / "demo"
QUERY_ID = "[2016] HKCFI 601"
K = 8
TOLERANCE = 0.5
CLINICAL_ALPHA = 0.7
MIN_INJURY_SIMILARITY = 0.1
# Cases singled out in the original critique as better comparables.
WATCH = ("[2003] HKCFI 270", "[1980] HKCFI 66", "[2000] HKCFI 1090")


def safe(name: str) -> str:
    return name.replace("[", "").replace("]", "").replace(" ", "_").replace("/", "_")


def money(value: float | None) -> str:
    return "—" if value is None else f"HK${value:,.0f}"


def raw_documents() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(CORPUS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cid = (data.get("metadata") or {}).get("neutral_citation")
        if cid:
            out[cid] = data
    return out


def local_similarities(
    model, query: CorpusCase, case_ids: list[str]
) -> dict[str, dict[str, dict]]:
    """Per-attribute local similarities of ``query`` against ``case_ids``."""
    names = [name for name, _desc, _fct in model.active_fcts()]
    S, known = model.per_attribute_similarities([query], case_ids)
    S, known = S[0], known[0]
    return {
        cid: {
            name: {"local": float(S[ci, ai]), "known": bool(known[ci, ai])}
            for ai, name in enumerate(names)
        }
        for ci, cid in enumerate(case_ids)
    }


def local(sims: dict[str, dict[str, dict]], case_id: str, attribute: str) -> float | None:
    entry = sims[case_id][attribute]
    return entry["local"] if entry["known"] else None


def case_card(case: CorpusCase, raw: dict, similarity: float | None, rank: int | None = None) -> str:
    plaintiff = raw.get("plaintiff") or {}
    lines = []
    header = case.case_id if rank is None else f"{rank}. {case.case_id}"
    lines.append(f"### {header}")
    if similarity is not None:
        lines.append(f"- **CBR similarity to query:** `{similarity:.3f}`")
    lines.append(f"- **Case name:** {raw.get('metadata', {}).get('case_name', '—')}")
    lines.append(
        f"- **Judgment:** {raw.get('metadata', {}).get('judgment_date', '—')} "
        f"({case.year})"
    )
    lines.append(
        f"- **Plaintiff:** {case.gender or '—'}, "
        f"age at accident {case.age_at_accident if case.age_at_accident is not None else '—'}, "
        f"age at trial {case.age_at_trial if case.age_at_trial is not None else '—'}"
        + (f", occupation: {plaintiff.get('occupation_before')}" if plaintiff.get("occupation_before") else "")
    )
    lines.append(
        f"- **Severity:** extracted `{case.overall_category or '—'}`, "
        f"reconstructed from award `{case.award_severity or '—'}`"
    )
    lines.append(
        f"- **Treatment:** {case.operations_count if case.operations_count is not None else '—'} operations, "
        f"{case.hospitalisation_days if case.hospitalisation_days is not None else '—'} hospital days"
    )
    lines.append(f"- **Injuries ({len(case.injuries)} ICD codes):** " + (", ".join(f"`{c}`" for c in case.injuries) or "—"))
    lines.append(f"- **Losses ({len(case.losses)}):** " + (", ".join(case.losses) or "—"))
    lines.append(
        f"- **PSLA:** nominal {money(case.psla_nominal)}, "
        f"real (2020) **{money(case.psla_real)}**"
    )
    return "\n".join(lines)


def main() -> None:
    price_index = PriceIndex.from_csv()
    cases = load_corpus(price_index=price_index)
    by_id = {c.case_id: c for c in cases}
    raw = raw_documents()
    tuned = json.loads((BENCH_DIR / "tuned_weights.json").read_text(encoding="utf-8"))
    weights = tuned["weights"]

    model = build_corpus_model(
        cases, attributes=CLINICAL_ATTRIBUTES + AUXILIARY_ATTRIBUTES,
        missing_policy="available",
    )
    retriever = CBRFeatureRetriever(model)

    query = by_id[QUERY_ID]
    result = retriever.retrieve(
        query, k=K, weights=weights, exclude=QUERY_ID, require_psla=True,
        clinical_alpha=CLINICAL_ALPHA, min_injury_similarity=MIN_INJURY_SIMILARITY,
    )
    deeper = retriever.retrieve(
        query, k=40, weights=weights, exclude=QUERY_ID, require_psla=True,
        clinical_alpha=CLINICAL_ALPHA, min_injury_similarity=MIN_INJURY_SIMILARITY,
    )
    rank_of = {r.case_id: i for i, r in enumerate(deeper.results, start=1)}

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "query").mkdir(exist_ok=True)
    (DEMO_DIR / "results").mkdir(exist_ok=True)
    for stale in list((DEMO_DIR / "results").glob("*")) + list(
        (DEMO_DIR / "query").glob("*")
    ):
        if stale.is_file():
            stale.unlink()

    # -- copy the query document -----------------------------------------
    query_doc = DEMO_DIR / "query" / f"{safe(QUERY_ID)}.json"
    query_doc.write_text(json.dumps(raw[QUERY_ID], indent=2, ensure_ascii=False), encoding="utf-8")

    judge_query = {
        "query_id": "demo-" + safe(QUERY_ID),
        "gender": query.gender,
        "age_at_accident": query.age_at_accident,
        "age_at_trial": query.age_at_trial,
        "injuries": list(query.injuries),
        "losses": list(query.losses),
        "overall_category": query.overall_category,
        "operations_count": query.operations_count,
        "hospitalisation_days": query.hospitalisation_days,
        "k": K,
    }
    judge_path = DEMO_DIR / "query" / "query_features.yaml"
    judge_path.write_text(yaml.safe_dump(judge_query, sort_keys=False), encoding="utf-8")

    # -- copy the returned documents -------------------------------------
    copied: list[tuple[int, object, Path]] = []
    for i, r in enumerate(result.results, start=1):
        path = DEMO_DIR / "results" / f"{i:02d}_{safe(r.case_id)}.json"
        path.write_text(json.dumps(raw[r.case_id], indent=2, ensure_ascii=False), encoding="utf-8")
        copied.append((i, r, path))

    groups: dict[tuple[str, float | None], list[int]] = {}
    for i, r, _path in copied:
        name = (raw[r.case_id].get("metadata") or {}).get("case_name")
        groups.setdefault((name, r.psla_nominal), []).append(i)
    duplicates = {k: v for k, v in groups.items() if len(v) > 1}

    # -- component similarities ------------------------------------------
    res_ids = [r.case_id for _i, r, _path in copied]
    top_sims = local_similarities(model, query, res_ids)

    # -- machine-readable result -----------------------------------------
    payload = {
        "query": query.case_id,
        "query_features": judge_query,
        "method": "clinical-first composite (injuries+losses dominant) + loose injury gate",
        "attributes": list(model.attribute_names),
        "clinical_alpha": CLINICAL_ALPHA,
        "min_injury_similarity": MIN_INJURY_SIMILARITY,
        "weights": weights,
        "k": K,
        "psla_tolerance": TOLERANCE,
        "candidate_pool": "cases with a known real PSLA (leave-one-out)",
        "query_psla_real": query.psla_real,
        "query_severity": {
            "extracted": query.overall_category,
            "from_award": query.award_severity,
        },
        "results": [
            {
                "rank": i,
                "case_id": r.case_id,
                "similarity": r.similarity,
                "injury_similarity": local(top_sims, r.case_id, "injuries"),
                "loss_similarity": local(top_sims, r.case_id, "losses"),
                "extracted_severity": r.extracted_severity,
                "award_severity": r.award_severity,
                "year": r.year,
                "psla_nominal": r.psla_nominal,
                "psla_real": r.psla_real,
                "log_ratio_error": log_ratio_error(query.psla_real, r.psla_real),
                "relative_error": relative_error(query.psla_real, r.psla_real),
                "within_tolerance": (
                    log_ratio_error(query.psla_real, r.psla_real) is not None
                    and log_ratio_error(query.psla_real, r.psla_real) <= math.log1p(TOLERANCE)
                ),
                "document": str(path.relative_to(REPO)),
            }
            for i, r, path in copied
        ],
    }
    (DEMO_DIR / "retrieval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # -- report -----------------------------------------------------------
    errs = [log_ratio_error(query.psla_real, r.psla_real) for _, r, _ in copied]
    errs = [e for e in errs if e is not None]
    within = sum(1 for e in errs if e <= math.log1p(TOLERANCE))
    median = sorted(errs)[len(errs) // 2]
    reals = [r.psla_real for _, r, _ in copied if r.psla_real]
    inj_sims = [local(top_sims, r.case_id, "injuries") for _, r, _ in copied]
    min_inj = min(v for v in inj_sims if v is not None)

    lines: list[str] = []
    lines.append("# Experiment 7 — worked example of the fetch")
    lines.append("")
    lines.append(
        "One end-to-end retrieval: a real corpus extraction is used as the **query "
        "document**, the CBR retriever fetches its nearest cases with the tuned "
        "clinical-first weights, and the **returned documents** are listed and copied "
        "next to this file."
    )
    lines.append("")
    lines.append("## 1. Query selection")
    lines.append("")
    lines.append(
        f"Query case **{query.case_id}** was chosen from the corpus because it has a rich, "
        "fully-populated feature vector (eight ICD-coded injuries, three loss categories, "
        "treatment intensity, demographics) and a mid-range award, so there are genuinely "
        "comparable precedents to find. Nothing about its PSLA is used to retrieve — that "
        "is the held-out target we check afterwards."
    )
    lines.append("")
    lines.append("## 2. The query document")
    lines.append("")
    lines.append(case_card(query, raw[QUERY_ID], similarity=None))
    lines.append("")
    lines.append(
        f"Full extracted document: [`query/{query_doc.name}`](query/{query_doc.name})  "
        "(the corpus extraction verbatim). A judge would express the same case as the "
        f"feature query [`query/{judge_path.name}`](query/{judge_path.name}):"
    )
    lines.append("")
    lines.append("```yaml")
    lines.append(judge_path.read_text(encoding="utf-8").rstrip())
    lines.append("```")
    lines.append("")
    lines.append("<details><summary>Full query document JSON</summary>")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(raw[QUERY_ID], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append("## 3. Retrieval setup")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append("| Method | myCBR local similarities + clinical-first composite amalgamation |")
    lines.append("| Clinical group | `injuries`, `losses` — missing-penalised |")
    lines.append(f"| Clinical share (`clinical_alpha`) | {CLINICAL_ALPHA} |")
    lines.append(f"| Loose injury gate (`min_injury_similarity`) | {MIN_INJURY_SIMILARITY} |")
    lines.append("| Auxiliary group | " + ", ".join(f"`{a}`" for a in AUXILIARY_ATTRIBUTES) + " — available-case |")
    lines.append("| Candidate pool | cases with a known real PSLA, query itself excluded (leave-one-out) |")
    lines.append(f"| k | {K} |")
    lines.append(f"| PSLA tolerance | ±{int(TOLERANCE * 100)}% (ratio in [2/3, 1.5]) |")
    lines.append("| Base year | 2020 (awards deflated with `data/price_index.csv`) |")
    lines.append("")
    lines.append(
        "Severity (`overall_category`) and `gender` are **not** retrieval attributes: "
        "gender is a near-constant, and severity is the award band (a target-derived "
        "encoding of the compensation), so it is reconstructed from the award for "
        "diagnostics only. Similarity weights (tuned against clinical NDCG@k on a "
        "held-out train split):"
    )
    lines.append("")
    lines.append("| Attribute | Weight | Group |")
    lines.append("|---|---|---|")
    for name, value in weights.items():
        group = "clinical" if name in CLINICAL_ATTRIBUTES else "auxiliary"
        lines.append(f"| `{name}` | {value:.3f} | {group} |")
    lines.append("")
    lines.append("## 4. Returned documents")
    lines.append("")
    lines.append(
        f"The query's own real PSLA is **{money(query.psla_real)}** (2020 HKD), while the "
        f"severity reconstructed from its award is **{query.award_severity}** (its extracted "
        f"label was `{query.overall_category}`). Across the top {K}: {within}/{len(errs)} are "
        f"within ±{int(TOLERANCE * 100)}%, median |ln(candidate/query)| = **{median:.3f}**, "
        f"the awards span {money(min(reals))} – {money(max(reals))}, and the minimum injury "
        f"similarity is **{min_inj:.3f}**."
    )
    lines.append("")
    lines.append("| Rank | Case | Similarity | Inj sim | Loss sim | Year | Severity (extracted) | Severity (from award) | PSLA real (2020) | abs(ln(c/q)) | Within ±50% |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r, _path in copied:
        e = log_ratio_error(query.psla_real, r.psla_real)
        inj = local(top_sims, r.case_id, "injuries")
        loss = local(top_sims, r.case_id, "losses")
        lines.append(
            f"| {i} | {r.case_id} | {r.similarity:.3f} | "
            f"{'—' if inj is None else f'{inj:.3f}'} | "
            f"{'—' if loss is None else f'{loss:.3f}'} | {r.year} | "
            f"{r.extracted_severity or '—'} | {r.award_severity or '—'} | "
            f"{money(r.psla_real)} | {'—' if e is None else f'{e:.3f}'} | "
            f"{'yes' if (e is not None and e <= math.log1p(TOLERANCE)) else 'no'} |"
        )
    lines.append("")
    if duplicates:
        notes = "; ".join(
            f"ranks {', '.join(str(r) for r in ranks)} are the same judgment "
            f"(*{name}*) indexed under different neutral citations"
            for (name, _amount), ranks in duplicates.items()
        )
        lines.append(
            f"> **Corpus note.** {notes}. This is a duplication in the extracted corpus, "
            "not a retrieval error — it also explains why those rows have identical awards."
        )
        lines.append("")
    lines.append("### Returned document cards")
    lines.append("")
    for i, r, path in copied:
        lines.append(case_card(by_id[r.case_id], raw[r.case_id], similarity=r.similarity, rank=i))
        lines.append("")
        lines.append(f"Full document: [`{path.relative_to(DEMO_DIR)}`]({path.relative_to(DEMO_DIR)})")
        lines.append("")
    lines.append("## 5. How the fetch works")
    lines.append("")
    lines.append(
        "1. The query is reduced to its clinical features (ICD codes, loss categories) "
        "plus the auxiliary attributes (ages, treatment intensity, counts)."
    )
    lines.append(
        "2. The **clinical group** is compared with the myCBR local similarity functions — "
        "ICD-11 taxonomy max-overlap for injuries, Jaccard for losses — and averaged with a "
        "*missing penalty*, so a case that cannot be compared clinically is pushed down "
        "rather than let off."
    )
    lines.append(
        "3. The **auxiliary group** is averaged available-case (missing attributes dropped) "
        "and only adjusts within the clinical neighbourhood. The two groups are combined as "
        f"`{CLINICAL_ALPHA} * clinical + {1 - CLINICAL_ALPHA:.1f} * auxiliary`."
    )
    lines.append(
        f"4. A **loose gate** on graded injury similarity (≥ {MIN_INJURY_SIMILARITY}) drops "
        "cases with no demonstrable clinical overlap before ranking, so treatment and counts "
        "cannot buy a clinically unrelated case into the top-k. The floor is deliberately low "
        "— an exact-code test would be defeated by extraction noise."
    )
    lines.append(
        "5. The weights were tuned to maximise clinical NDCG@k (injury/loss relevance), not "
        "PSLA-band NDCG; PSLA closeness is reported separately as the compensation signal."
    )
    lines.append("")
    lines.append("## 6. The diagnosis that motivated this method")
    lines.append("")
    lines.append(
        "The earlier headline benchmark was only marginally better than chance, and the "
        "worked example's 8th hit was a single-region hand-crush with no exact ICD overlap "
        "while a clinically richer case sat at rank 25. The cause was **objective "
        "misalignment**: weights were tuned against PSLA-band relevance, so treatment "
        "intensity, loss counts and severity were upweighted and the injury and loss sets "
        "were almost tuned away. Severity was also dropping out as a *free pass* for cases "
        "that lacked it. The full write-up is in "
        "`src/hklii_psla/experiments/experiment7/notes.md`."
    )
    lines.append("")
    lines.append("## 7. What changed, and the acceptance check")
    lines.append("")
    lines.append("| Before | After |")
    lines.append("|---|---|")
    lines.append("| Available-case weighted sum over 10 attributes | Clinical-first composite: clinical group (injury+loss) missing-penalised, auxiliary available-case |")
    lines.append("| Tuned against PSLA-band NDCG | Tuned against clinical NDCG@k |")
    lines.append("| `injuries` weight 0.16, `losses` 0.26 | Clinical group carries 0.7 of the score |")
    lines.append("| `gender`, `overall_category` in the score | Both removed; severity reconstructed from the award |")
    lines.append("| No clinical floor; rank 8 had injury sim 0.000 | Loose injury gate (≥ 0.1) before ranking |")
    lines.append("")
    lines.append(
        "The new top-8 all clear the clinical gate and are injuries-first: every returned "
        "case has injury similarity ≥ "
        f"{min_inj:.3f}, whereas the previous top-8 included two cases with injury "
        "similarity 0.000. The two cases the old critique flagged as unfairly buried now "
        "rank honestly on their clinical merits:"
    )
    lines.append("")
    lines.append("| Case | Injury sim | Loss sim | Old rank | New rank |")
    lines.append("|---|---|---|---|---|")
    deep_sims = local_similarities(model, query, [r.case_id for r in deeper.results])
    old_rank = {"[2003] HKCFI 270": 25, "[1980] HKCFI 66": 18, "[2000] HKCFI 1090": 8}
    for cid in WATCH:
        inj = local(deep_sims, cid, "injuries") if cid in deep_sims else None
        loss = local(deep_sims, cid, "losses") if cid in deep_sims else None
        lines.append(
            f"| {cid} | {'—' if inj is None else f'{inj:.3f}'} | "
            f"{'—' if loss is None else f'{loss:.3f}'} | {old_rank[cid]} | "
            f"{rank_of.get(cid, '>40')} |"
        )
    lines.append("")
    lines.append(
        "`[2003] HKCFI 270` and `[1980] HKCFI 66` do **not** enter the top-8: at injury "
        "similarity 0.458 and 0.538 they are now out-ranked by cases with 0.54–0.69. Their "
        "old claim to the top-8 rested on award closeness, which is no longer part of the "
        "clinical score — the original acceptance criterion assumed PSLA-closeness remained "
        "a retrieval term, which would have leaked the query's award in leave-one-out "
        "evaluation. `[2000] HKCFI 1090` (injury sim 0.243, no exact overlap) correctly "
        "drops out. PSLA closeness is tracked separately: on the full benchmark the "
        "PSLA-band NDCG@10 is ~0.21 versus a random baseline of ~0.15, and the award "
        "spread above shows the tension between clinical and quantum closeness remains "
        "visible."
    )
    lines.append("")
    lines.append("## 8. Files")
    lines.append("")
    lines.append("| Path | Contents |")
    lines.append("|---|---|")
    lines.append("| `DEMO.md` | this report |")
    lines.append("| `retrieval.json` | machine-readable query + ranked results |")
    lines.append(f"| `query/{query_doc.name}` | the query document (corpus extraction) |")
    lines.append(f"| `query/{judge_path.name}` | the same case as a judge `FeatureQuery` |")
    for i, r, path in copied:
        lines.append(f"| `{path.relative_to(DEMO_DIR)}` | returned document #{i} ({r.case_id}) |")
    lines.append("")

    (DEMO_DIR / "DEMO.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {DEMO_DIR / 'DEMO.md'}")
    print(f"Wrote {DEMO_DIR / 'retrieval.json'}")
    print(f"Copied 1 query document + {len(copied)} returned documents")
    print(
        f"PSLA: query={money(query.psla_real)}  within ±50%: {within}/{len(errs)}  "
        f"median log-err={median:.3f}  min inj sim={min_inj:.3f}"
    )


if __name__ == "__main__":
    main()