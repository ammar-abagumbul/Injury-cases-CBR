"""Experiment 8 — an LLM-written severity rubric, and how well it classifies new cases.

Phase ``induce``   The LLM reads labelled example cards (built from the extracted JSON, with the
                   human annotation as the tier) and writes a rubric: what each severity tier looks
                   like. Several independent samples are merged into one rubric. Rules that name a
                   numeric feature are then checked against the data.
Phase ``classify`` On held-out cases the LLM assigns a tier using the rubric. The same cases are
                   also classified without the rubric, with a few similar labelled cases, and by
                   two classic ML models, so you can see what the rubric adds.

The train/test split is stored in ``split.json`` and the rubric is induced from training cases
only, so the held-out scores are honest. The LLM steps call Azure/your configured model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Literal, override

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.table import Table
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from hklii_psla.experiments.experiment import BaseExperiment
from hklii_psla.experiments.experiment6 import plots as plots6
from hklii_psla.experiments.experiment6.dataset import LabelledCase, load_labelled_cases
from hklii_psla.experiments.experiment6.features import build_feature_table
from hklii_psla.experiments.experiment6.modeling import (
    MODEL_LABELS,
    build_pipeline,
    score,
)
from hklii_psla.experiments.experiment7.exp_7 import court_of
from hklii_psla.experiments.experiment7.explain import make_explainer
from hklii_psla.experiments.experiment8.cards import build_card
from hklii_psla.experiments.experiment8.rubric import (
    CHECKABLE_FEATURES,
    Rubric,
    check_rules,
    classify_system,
    classify_user,
    induction_system,
    induction_user,
    make_prediction_model,
    merge_system,
    merge_user,
    render_rubric,
)
from hklii_psla.extractor.base import TokenCount
from hklii_psla.extractor.utils import extract_token_counts
from hklii_psla.model_factory import create_model

console = Console()

Mode = Literal["rubric", "no_rubric", "few_shot", "ml_logistic", "ml_forest"]
LLM_MODES = ("rubric", "no_rubric", "few_shot")
MODE_LABELS = {
    "rubric": "LLM + rubric", "no_rubric": "LLM, no rubric", "few_shot": "LLM + similar cases",
    "ml_logistic": MODEL_LABELS["logistic_regression"], "ml_forest": MODEL_LABELS["random_forest"],
}


class Exp8Config(BaseModel):
    id: str
    cases_path: str
    cases_glob: str = "*.json"
    labels_path: str
    label_column: str = "injury_type"
    tiers: dict[str, list[str]] = Field(default_factory=lambda: {
        "non-serious": ["nonserious", "non-serious"],
        "serious": ["serious"],
        "substantial+": ["substantial", "gross", "disaster"],
    })
    output_dir: str
    courts: list[str] | None = None
    model: str = "gpt"
    phase: Literal["all", "induce", "classify"] = "all"
    test_fraction: float = 0.3
    seed: int = 42
    cases_per_tier: int = 15          # example cards per tier in one induction prompt
    n_induction_samples: int = 5      # independent drafts that get merged
    max_concurrency: int = 5
    modes: list[Mode] = Field(default_factory=lambda: ["rubric", "no_rubric", "few_shot", "ml_logistic", "ml_forest"])
    few_shot_k: int = 6
    reuse_rubric: bool = True         # keep an existing rubric.json instead of inducing again
    icd_path: str = "data/ICD-11.json"

    @model_validator(mode="after")
    def _check(self) -> Exp8Config:
        if not 0.05 <= self.test_fraction <= 0.7:
            raise ValueError("test_fraction should be between 0.05 and 0.7")
        if self.n_induction_samples < 1 or self.cases_per_tier < 1:
            raise ValueError("n_induction_samples and cases_per_tier must be at least 1")
        return self


# ------------------------------------------------------------------------- LLM wrapper


class _LLM:
    """Structured-output calls with a concurrency limit, retries and token accounting."""

    def __init__(self, model, max_concurrency: int, attempts: int = 2):
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)
        self._attempts = attempts
        self.tokens = TokenCount()

    async def call(self, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        structured = self._model.with_structured_output(schema, include_raw=True)
        last: Exception | None = None
        for _ in range(self._attempts):
            try:
                async with self._sem:
                    result = await structured.ainvoke([("system", system), ("human", user)])
                try:
                    self.tokens = self.tokens + extract_token_counts(result)
                except Exception:  # usage metadata is best-effort
                    pass
                parsed = result.get("parsed")
                if isinstance(parsed, schema):
                    return parsed
                last = RuntimeError(f"model returned no valid {schema.__name__}")
            except Exception as exc:
                last = exc
        raise RuntimeError(str(last))


# --------------------------------------------------------------------------- experiment


class SeverityRubricExperiment(BaseExperiment[Exp8Config], experiment_id="severity-rubric"):

    _config: Exp8Config
    _config_model: ClassVar[type[BaseModel]] = Exp8Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._out = Path(self._config.output_dir)
        self._result: dict[str, Any] | None = None
        self._state: dict[str, Any] = {}

    # -- data --------------------------------------------------------------
    def _load(self) -> None:
        cfg = self._config
        labelled, report = load_labelled_cases(
            Path(cfg.cases_path), cfg.cases_glob, Path(cfg.labels_path), cfg.label_column, cfg.tiers
        )
        console.print(f"[cyan]{report.n_case_files} case files, {len(labelled)} matched to a human label.[/cyan]")
        if cfg.courts is not None:
            labelled = [lc for lc in labelled if court_of(lc) in cfg.courts]
            console.print(f"[cyan]Kept {len(labelled)} cases from courts {cfg.courts}.[/cyan]")
        present = {lc.tier for lc in labelled}
        tier_names = [t for t in cfg.tiers if t in present]
        if len(tier_names) < 2:
            raise RuntimeError(f"Need cases in at least two tiers, found {tier_names}. "
                               "Check cases_path, labels_path and courts.")
        if len(tier_names) < len(cfg.tiers):
            console.print(f"[yellow]No labelled case in tier(s) {[t for t in cfg.tiers if t not in present]}; "
                          "continuing without them.[/yellow]")

        X, _ = build_feature_table([lc.case for lc in labelled], min_support=1)
        ids = sorted(lc.case_id for lc in labelled)
        order = np.random.default_rng(cfg.seed).permutation(len(ids))
        card_ids = {ids[j]: f"C{n + 1:03d}" for n, j in enumerate(order)}  # neutral, order carries no tier info
        by_id = {lc.case_id: i for i, lc in enumerate(labelled)}
        self._state.update(
            labelled=labelled, tier_names=tier_names, X=X, card_ids=card_ids, by_id=by_id,
            courts=pd.Series([court_of(lc) for lc in labelled]),
            tiers=pd.Series([lc.tier for lc in labelled]),
            cards={lc.case_id: build_card(lc.case, card_ids[lc.case_id]) for lc in labelled},
        )

    def _split(self) -> None:
        """Stratified train/test split, saved so later runs (and new cases) keep the same assignment."""
        cfg, st = self._config, self._state
        path = self._out / "split.json"
        labelled: list[LabelledCase] = st["labelled"]
        known = json.loads(path.read_text())["test"] if path.exists() else None
        known_train = json.loads(path.read_text())["train"] if path.exists() else []

        ids = [lc.case_id for lc in labelled]
        if known is None:
            tiers = [lc.tier for lc in labelled]
            counts = pd.Series(tiers).value_counts()
            stratify = tiers if counts.min() >= 2 else None
            train, test = train_test_split(ids, test_size=cfg.test_fraction, random_state=cfg.seed, stratify=stratify)
        else:
            old = set(known) | set(known_train)
            train = [i for i in ids if i in set(known_train)]
            test = [i for i in ids if i in set(known)]
            for i in ids:  # cases extracted since the split was saved: assign deterministically
                if i not in old:
                    h = int(hashlib.sha256(f"{cfg.seed}:{i}".encode()).hexdigest(), 16) % 1000 / 1000
                    (test if h < cfg.test_fraction else train).append(i)
        st["train"], st["test"] = sorted(train), sorted(test)
        self._out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"train": st["train"], "test": st["test"], "seed": cfg.seed,
                                    "card_ids": st["card_ids"]}, indent=1))
        tc = pd.Series([labelled[st["by_id"][i]].tier for i in st["test"]]).value_counts().to_dict()
        console.print(f"[cyan]Split: {len(st['train'])} train, {len(st['test'])} test (test tiers: {tc}).[/cyan]")

    # -- induction ---------------------------------------------------------
    def _checkable(self) -> dict[str, str]:
        explain = make_explainer(Path(self._config.icd_path))
        cols = set(self._state["X"].columns)
        return {f: explain(f) for f in CHECKABLE_FEATURES if f in cols}

    def _sample_cards(self, k: int) -> dict[str, list[str]]:
        """Up to ``cases_per_tier`` training cards per tier, interleaved across courts."""
        cfg, st = self._config, self._state
        rng = np.random.default_rng(cfg.seed + 1000 + k)
        labelled: list[LabelledCase] = st["labelled"]
        out: dict[str, list[str]] = {}
        for tier in st["tier_names"]:
            pool: dict[str, list[str]] = {}
            for cid in st["train"]:
                lc = labelled[st["by_id"][cid]]
                if lc.tier == tier:
                    pool.setdefault(court_of(lc), []).append(cid)
            for lst in pool.values():
                rng.shuffle(lst)
            picked: list[str] = []
            while len(picked) < cfg.cases_per_tier and any(pool.values()):
                for court in sorted(pool):
                    if pool[court] and len(picked) < cfg.cases_per_tier:
                        picked.append(pool[court].pop())
            rng.shuffle(picked)
            out[tier] = [st["cards"][c] for c in picked]
        return out

    async def _induce(self, llm: _LLM) -> tuple[Rubric, list[Rubric]]:
        cfg, st = self._config, self._state
        feats = self._checkable()
        system = induction_system(st["tier_names"], feats)

        async def draft(k: int) -> Rubric | None:
            try:
                console.print(f"[dim]Induction draft {k + 1}/{cfg.n_induction_samples}...[/dim]")
                return await llm.call(Rubric, system, induction_user(self._sample_cards(k)))  # type: ignore[return-value]
            except Exception as exc:
                console.print(f"[red]Draft {k + 1} failed: {exc}[/red]")
                return None

        drafts = [d for d in await asyncio.gather(*(draft(k) for k in range(cfg.n_induction_samples))) if d]
        if not drafts:
            raise RuntimeError("Every induction draft failed; see the errors above.")
        if len(drafts) == 1:
            return drafts[0], drafts
        console.print(f"[dim]Merging {len(drafts)} drafts...[/dim]")
        merged = await llm.call(Rubric, merge_system(st["tier_names"], feats), merge_user(drafts))
        return merged, drafts  # type: ignore[return-value]

    # -- classification ------------------------------------------------------
    def _neighbours(self) -> dict[str, list[str]]:
        """For each test case, its most similar training cases (standardised extracted features)."""
        st, cfg = self._state, self._config
        X: pd.DataFrame = st["X"]
        idx = st["by_id"]
        train_rows = [idx[c] for c in st["train"]]
        prep = [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
        Z = X.to_numpy()
        for step in prep:
            Z = step.fit(Z[train_rows]).transform(Z)
        k = min(cfg.few_shot_k, len(train_rows))
        nn = NearestNeighbors(n_neighbors=k).fit(Z[train_rows])
        out = {}
        for cid in st["test"]:
            _, nbr = nn.kneighbors(Z[[idx[cid]]])
            out[cid] = [st["train"][j] for j in nbr[0]]
        return out

    async def _classify(self, llm: _LLM, rubric: Rubric | None) -> dict[str, dict[str, dict[str, Any]]]:
        cfg, st = self._config, self._state
        labelled: list[LabelledCase] = st["labelled"]
        Pred = make_prediction_model(st["tier_names"])
        neighbours = self._neighbours() if "few_shot" in cfg.modes else {}
        systems = {
            "rubric": classify_system(st["tier_names"], rubric) if rubric else None,
            "no_rubric": classify_system(st["tier_names"], None),
            "few_shot": classify_system(st["tier_names"], None),
        }
        preds: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in LLM_MODES if m in cfg.modes}
        if "rubric" in preds and rubric is None:
            raise RuntimeError("Mode 'rubric' needs a rubric: run phase 'induce' first.")
        done = 0
        total = len(st["test"]) * len(preds)

        async def one(mode: str, cid: str) -> None:
            nonlocal done
            examples = None
            if mode == "few_shot":
                examples = [(st["cards"][n], labelled[st["by_id"][n]].tier) for n in neighbours[cid]]
            try:
                p = await llm.call(Pred, systems[mode], classify_user(st["cards"][cid], examples))  # type: ignore[arg-type]
                preds[mode][cid] = {"tier": p.tier, "confidence": p.confidence,  # type: ignore[attr-defined]
                                    "criteria": p.criteria_used, "rationale": p.rationale}  # type: ignore[attr-defined]
            except Exception as exc:
                preds[mode][cid] = {"error": str(exc)}
            done += 1
            if done % 10 == 0 or done == total:
                console.print(f"[dim]Classified {done}/{total}[/dim]")

        await asyncio.gather(*(one(m, c) for m in preds for c in st["test"]))
        return preds

    def _ml_predictions(self) -> dict[str, dict[str, dict[str, Any]]]:
        cfg, st = self._config, self._state
        X: pd.DataFrame = st["X"]
        tier_names = st["tier_names"]
        labelled: list[LabelledCase] = st["labelled"]
        tr = [st["by_id"][c] for c in st["train"]]
        te = [st["by_id"][c] for c in st["test"]]
        y = np.array([tier_names.index(lc.tier) for lc in labelled])
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for mode, name in (("ml_logistic", "logistic_regression"), ("ml_forest", "random_forest")):
            if mode not in cfg.modes:
                continue
            pipe = build_pipeline(name, list(X.columns), {}, cfg.seed).fit(X.iloc[tr], y[tr])
            out[mode] = {c: {"tier": tier_names[int(p)]} for c, p in zip(st["test"], pipe.predict(X.iloc[te]))}
        return out

    # -- run ---------------------------------------------------------------
    @override
    def run(self) -> None:
        cfg = self._config
        self._load()
        self._split()
        st = self._state
        rubric_path = self._out / "rubric.json"
        llm: _LLM | None = None

        def get_llm() -> _LLM:
            nonlocal llm
            if llm is None:
                llm = _LLM(create_model(cfg.model), cfg.max_concurrency)
            return llm

        rubric: Rubric | None = None
        drafts: list[Rubric] = []
        if cfg.phase in ("all", "induce"):
            if cfg.reuse_rubric and rubric_path.exists():
                rubric = Rubric.model_validate_json(rubric_path.read_text())
                console.print(f"[cyan]Reusing existing rubric {rubric_path}[/cyan]")
            else:
                rubric, drafts = asyncio.run(self._induce(get_llm()))
        elif rubric_path.exists():
            rubric = Rubric.model_validate_json(rubric_path.read_text())

        preds: dict[str, dict[str, dict[str, Any]]] = {}
        if cfg.phase in ("all", "classify"):
            if any(m in cfg.modes for m in LLM_MODES):
                preds.update(asyncio.run(self._classify(get_llm(), rubric)))
            preds.update(self._ml_predictions())

        st.update(rubric=rubric, drafts=drafts, preds=preds, tokens=llm.tokens if llm else TokenCount())
        self._result = self._evaluate(rubric, preds)
        self._display()

    # -- evaluation ----------------------------------------------------------
    def _evaluate(self, rubric: Rubric | None, preds: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        st = self._state
        tier_names: list[str] = st["tier_names"]
        labelled: list[LabelledCase] = st["labelled"]
        truth = {c: tier_names.index(labelled[st["by_id"][c]].tier) for c in st["test"]}
        n = len(tier_names)

        valid = {m: {c for c, p in d.items() if "tier" in p} for m, d in preds.items()}
        failed = {m: len(truth) - len(v) for m, v in valid.items()}
        common = sorted(set.intersection(*valid.values())) if valid else []
        result: dict[str, Any] = {
            "tier_names": tier_names, "n_train": len(st["train"]), "n_test": len(st["test"]),
            "n_scored": len(common), "failed_predictions": failed, "modes": {}, "mcnemar": {},
        }
        y_true = np.array([truth[c] for c in common])
        confusion = {}
        for mode, d in preds.items():
            if not common:
                break
            y_pred = np.array([tier_names.index(d[c]["tier"]) for c in common])
            s = score(y_true, y_pred, n)
            s["mean_abs_tier_error"] = float(np.abs(y_true - y_pred).mean())
            result["modes"][mode] = s
            cm = np.zeros((n, n))
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            confusion[mode] = cm
        st["confusion"] = confusion

        def mcnemar(a: str, b: str) -> dict[str, Any] | None:
            if a not in preds or b not in preds or not common:
                return None
            ra = np.array([tier_names.index(preds[a][c]["tier"]) == truth[c] for c in common])
            rb = np.array([tier_names.index(preds[b][c]["tier"]) == truth[c] for c in common])
            only_a, only_b = int((ra & ~rb).sum()), int((~ra & rb).sum())
            p = 1.0 if only_a + only_b == 0 else float(stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue)
            return {"only_first_right": only_a, "only_second_right": only_b, "p_value": p}

        for a, b in (("rubric", "no_rubric"), ("rubric", "few_shot"), ("rubric", "ml_logistic")):
            if (m := mcnemar(a, b)) is not None:
                result["mcnemar"][f"{a} vs {b}"] = m

        if "rubric" in preds:  # does the model's stated confidence mean anything?
            by_conf: dict[str, list[bool]] = {}
            for c in common:
                p = preds["rubric"][c]
                by_conf.setdefault(p["confidence"], []).append(tier_names.index(p["tier"]) == truth[c])
            result["rubric_accuracy_by_confidence"] = {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in by_conf.items()}

        checks = pd.DataFrame()
        if rubric is not None:
            frames = []
            for label, ids in (("train", st["train"]), ("test", st["test"])):
                mask = np.zeros(len(labelled), dtype=bool)
                mask[[st["by_id"][c] for c in ids]] = True
                df = check_rules(rubric, st["X"], st["tiers"], mask, tier_names)
                if not df.empty:
                    frames.append(df.assign(cases=label))
            checks = pd.concat(frames) if frames else checks
        st["rule_checks"] = checks
        if len(checks) and "verdict" in checks:
            train = checks[checks["cases"] == "train"]
            result["rule_checks_train"] = train["verdict"].value_counts().to_dict()
        return result

    # -- display / persistence ---------------------------------------------------
    def _display(self) -> None:
        st, res = self._state, self._result
        assert res is not None
        if st["rubric"] is not None:
            console.print("\n[bold]Rubric overview[/bold]\n" + st["rubric"].overview)
            for p in st["rubric"].profiles:
                console.print(f"[bold cyan]{p.tier}[/bold cyan]: {p.summary}")
        if res["modes"]:
            t = Table(title=f"Held-out classification ({res['n_scored']} test cases scored by every mode)")
            for col in ("method", "macro-F1", "QWK", "bal. acc.", "accuracy", "avg tier error", *[f"recall {x}" for x in res["tier_names"]]):
                t.add_column(col, justify="right" if col != "method" else "left")
            for mode, s in res["modes"].items():
                t.add_row(MODE_LABELS[mode], f"{s['macro_f1']:.2f}", f"{s['qwk']:.2f}", f"{s['balanced_accuracy']:.2f}",
                          f"{s['accuracy']:.2f}", f"{s['mean_abs_tier_error']:.2f}", *[f"{r:.2f}" for r in s["recall"]])
            console.print(t)
            for name, m in res["mcnemar"].items():
                console.print(f"{name}: first-only right {m['only_first_right']}, second-only right "
                              f"{m['only_second_right']}, exact p = {m['p_value']:.3f}")
        checks = st["rule_checks"]
        if len(checks) and "p_value" in checks:
            t = Table(title="Rubric rules checked against the data (training cases)")
            for col in ("tier", "condition", "in tier", "in other tiers", "verdict"):
                t.add_column(col)
            for _, r in checks[checks["cases"] == "train"].iterrows():
                t.add_row(r["tier"], r["condition"], str(r.get("in_tier", "")), str(r.get("in_other_tiers", "")), r["verdict"])
            console.print(t)

    @override
    def save_results(self) -> None:
        res = self._result
        if res is None:
            raise RuntimeError("run() must be called before save_results()")
        st, cfg = self._state, self._config
        out = self._out
        (out / "plots").mkdir(parents=True, exist_ok=True)
        rubric: Rubric | None = st["rubric"]
        if rubric is not None:
            (out / "rubric.json").write_text(rubric.model_dump_json(indent=2), encoding="utf-8")
            (out / "rubric.md").write_text(render_rubric(rubric), encoding="utf-8")
        if st["drafts"]:
            (out / "drafts").mkdir(exist_ok=True)
            for i, d in enumerate(st["drafts"]):
                (out / "drafts" / f"draft_{i + 1}.json").write_text(d.model_dump_json(indent=2), encoding="utf-8")

        rows = []
        labelled: list[LabelledCase] = st["labelled"]
        for cid in st["test"]:
            lc = labelled[st["by_id"][cid]]
            row: dict[str, Any] = {"case_id": cid, "card_id": st["card_ids"][cid], "court": court_of(lc),
                                   "human_tier": lc.tier, "raw_label": lc.raw_label}
            for mode, d in st["preds"].items():
                p = d.get(cid, {})
                row[f"{mode}_tier"] = p.get("tier", "")
                if "confidence" in p:
                    row[f"{mode}_confidence"] = p["confidence"]
                if "error" in p:
                    row[f"{mode}_error"] = p["error"]
                if mode == "rubric" and "rationale" in p:
                    row["rubric_rationale"] = p["rationale"]
                    row["rubric_criteria"] = " | ".join(p["criteria"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
        if len(st["rule_checks"]):
            st["rule_checks"].to_csv(out / "rule_checks.csv", index=False)

        if st["confusion"]:
            norm = {m: cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) for m, cm in st["confusion"].items()}
            plots6.confusion_matrices({MODE_LABELS[m]: v for m, v in norm.items()}, res["tier_names"],
                                      out / "plots" / "confusion_matrices.png", f"Held-out cases, n={res['n_scored']}")
            self._plot_metrics(out / "plots" / "metrics.png")

        summary = {**res, "config": cfg.model_dump(), "tokens": {"input": st["tokens"].input_tokens, "output": st["tokens"].output_tokens}}
        (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        (out / "report.md").write_text(self._report(), encoding="utf-8")
        console.print(f"\n[bold green]Report: {out / 'report.md'}[/bold green]")

    def _plot_metrics(self, path: Path) -> None:
        import matplotlib.pyplot as plt

        res = self._result
        assert res is not None
        metrics = ("macro_f1", "qwk", "balanced_accuracy")
        modes = list(res["modes"])
        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.8 / len(modes)
        for j, mode in enumerate(modes):
            vals = [res["modes"][mode][m] for m in metrics]
            ax.bar([i + j * width for i in range(len(metrics))], vals, width, label=MODE_LABELS[mode])
        ax.set_xticks([i + width * (len(modes) - 1) / 2 for i in range(len(metrics))], ["macro-F1", "QWK", "balanced accuracy"])
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Held-out cases (n={res['n_scored']}); small test sets mean wide uncertainty")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _report(self) -> str:
        res, st = self._result, self._state
        assert res is not None
        lines = ["# Severity rubric and held-out classification", "",
                 f"{res['n_train']} training cases (rubric written from these only), {res['n_test']} held-out test cases.", ""]
        if st["rubric"] is not None:
            lines += ["## The rubric", "", render_rubric(st["rubric"]), ""]
        if res["modes"]:
            lines += ["## How well it classifies held-out cases", "",
                      f"{res['n_scored']} test cases scored by every method. Failed LLM calls: {res['failed_predictions']}.", "",
                      "| method | macro-F1 | QWK | balanced acc. | accuracy | avg tier error | "
                      + " | ".join(f"recall {t}" for t in res["tier_names"]) + " |",
                      "|---|" + "---|" * (5 + len(res["tier_names"]))]
            for mode, s in res["modes"].items():
                lines.append(f"| {MODE_LABELS[mode]} | {s['macro_f1']:.2f} | {s['qwk']:.2f} | {s['balanced_accuracy']:.2f} | "
                             f"{s['accuracy']:.2f} | {s['mean_abs_tier_error']:.2f} | " + " | ".join(f"{r:.2f}" for r in s["recall"]) + " |")
            lines.append("")
            for name, m in res["mcnemar"].items():
                lines.append(f"- {name}: {m['only_first_right']} cases only the first method got right, "
                             f"{m['only_second_right']} only the second (exact p = {m['p_value']:.3f}).")
            if "rubric_accuracy_by_confidence" in res:
                lines += ["", "Rubric accuracy by the model's stated confidence: "
                          + ", ".join(f"{k}: {v['accuracy']:.0%} (n={v['n']})" for k, v in res["rubric_accuracy_by_confidence"].items()) + "."]
            lines.append("")
        checks = st["rule_checks"]
        if len(checks) and "verdict" in checks:
            lines += ["## Do the rubric's numeric rules hold in the data?", "",
                      "| tier | condition | cases | in tier | in other tiers | verdict |", "|---|---|---|---|---|---|"]
            for _, r in checks.iterrows():
                lines.append(f"| {r['tier']} | {r['condition']} | {r['cases']} | {r.get('in_tier', '')} | {r.get('in_other_tiers', '')} | {r['verdict']} |")
            lines.append("")
        lines += ["## How to read this", "",
                  "- The rubric is written from training cases only; scores on the held-out cases are the honest ones.",
                  "- With few test cases per tier, differences between methods of a few cases are noise. "
                  "Look at the exact p-values, and treat small tiers as tentative.",
                  "- If tiers and courts overlap in your data, the rubric can encode court style instead of severity. "
                  "Restrict `courts` in the config to avoid this.", ""]
        return "\n".join(lines)
