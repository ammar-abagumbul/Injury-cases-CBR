"""Experiment 7 — what makes a case fall in a severity tier.

Uses the LLM-extracted ``Case`` JSON files as features and the human annotation as the
tier, then looks for *explanations* instead of predictions:

1. per-tier feature statistics (which features separate adjacent tiers, and each tier from the rest)
2. features that rise or fall with severity (Spearman trend)
3. a shallow decision tree turned into if/then rules, with cross-validated reliability
4. checks for court confounding (High Court vs District Court cases mixed up with tier)

Everything is deterministic (no LLM calls). Output: ``report.md`` plus CSVs and figures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, ClassVar, override

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.table import Table

from hklii_psla.experiments.experiment import BaseExperiment
from hklii_psla.experiments.experiment6.dataset import LabelledCase, load_labelled_cases
from hklii_psla.experiments.experiment6.features import (
    FEATURE_GROUPS,
    build_feature_table,
)
from hklii_psla.experiments.experiment7 import plots
from hklii_psla.experiments.experiment7.analysis import (
    Analysis,
    ComparisonResult,
    compare,
    court_proxy_auc,
    feature_kinds,
    fit_tree,
    total_variation,
    trend,
)
from hklii_psla.experiments.experiment7.explain import make_explainer
from hklii_psla.experiments.experiment7.report import build_report

console = Console()

_COURT_RE = re.compile(r"HK[A-Z]+")
_PRESENCE_PREFIXES = ("icd2_", "icd4_", "losscat_", "treatkw_", "injkw_")


class Exp7Config(BaseModel):
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
    courts: list[str] | None = None  # e.g. ["HKCFI"] keeps only High Court cases
    feature_groups: list[str] = Field(default_factory=lambda: list(FEATURE_GROUPS))
    icd_path: str = "data/ICD-11.json"
    use_stated_flags: bool = False  # the "*_stated" columns describe the judgment text, not the injury
    min_feature_support: int = 5   # drop sparse features present in fewer cases than this
    min_group_size: int = 5        # smallest group a comparison may use
    fdr_alpha: float = 0.10        # false-discovery threshold for "significant"
    top_n: int = 12                # rows shown per table
    profile_features: int = 20     # rows in the tier-profile heatmap
    tree_max_depth: int = 3
    tree_min_samples_leaf: int = 5
    n_splits: int = 5
    n_repeats: int = 10
    seed: int = 42

    @model_validator(mode="after")
    def _check(self) -> Exp7Config:
        bad = set(self.feature_groups) - set(FEATURE_GROUPS)
        if bad:
            raise ValueError(f"Unknown feature groups {sorted(bad)}; use {list(FEATURE_GROUPS)}")
        if len(self.tiers) < 2:
            raise ValueError("Need at least two tiers")
        return self


def court_of(lc: LabelledCase) -> str:
    """HKCFI / HKDC / HKCA ... from the neutral citation, falling back to the filename."""
    for text in (lc.case.metadata.neutral_citation, lc.case_id):
        m = _COURT_RE.search(text or "")
        if m:
            return m.group()
    return "other"


class SeverityProfilingExperiment(BaseExperiment[Exp7Config], experiment_id="severity-profiling"):

    _config: Exp7Config
    _config_model: ClassVar[type[BaseModel]] = Exp7Config

    def __init__(self, config_path: Path):
        super().__init__(config_path)
        self._analysis: Analysis | None = None
        self._state: dict[str, Any] = {}
        self._seen_comparisons: set[tuple[bytes, bytes]] = set()

    # -- pipeline ---------------------------------------------------------
    @override
    def run(self) -> None:
        cfg = self._config
        labelled, report = load_labelled_cases(
            Path(cfg.cases_path), cfg.cases_glob, Path(cfg.labels_path), cfg.label_column, cfg.tiers
        )
        console.print(f"[cyan]{report.n_case_files} case files, {len(labelled)} matched to a human label, "
                      f"{len(report.unmatched_cases)} not in the label file.[/cyan]")
        if cfg.courts is not None:
            labelled = [lc for lc in labelled if court_of(lc) in cfg.courts]
            console.print(f"[cyan]Kept {len(labelled)} cases from courts {cfg.courts}.[/cyan]")
        if not labelled:
            raise RuntimeError("No labelled cases to analyse. Check cases_path, labels_path and courts.")

        present = {lc.tier for lc in labelled}
        tier_names = [t for t in cfg.tiers if t in present]
        if len(tier_names) < 2:
            raise RuntimeError(f"Only one tier has labelled cases ({tier_names}); nothing to contrast.")
        dropped = [t for t in cfg.tiers if t not in present]
        if dropped:
            console.print(f"[yellow]No labelled case in tier(s) {dropped}; continuing without them.[/yellow]")

        X_all, groups = build_feature_table([lc.case for lc in labelled], cfg.min_feature_support)
        columns = [c for g in cfg.feature_groups for c in groups[g]]
        if not cfg.use_stated_flags:
            columns = [c for c in columns if not c.endswith("_stated")]
        X = X_all[columns].copy()
        # code-like counts (ICD block, loss category, keyword flags) are compared as "has it / doesn't"
        for col in X.columns:
            if col.startswith(_PRESENCE_PREFIXES):
                X[col] = (X[col] > 0).astype(float)
        kinds = feature_kinds(X)
        y = np.array([tier_names.index(lc.tier) for lc in labelled])
        courts = pd.Series([court_of(lc) for lc in labelled])
        tiers = pd.Series([lc.tier for lc in labelled])
        tier_counts = {t: int((y == i).sum()) for i, t in enumerate(tier_names)}
        court_tier = pd.crosstab(tiers, courts).reindex(tier_names).fillna(0).astype(int)

        strata = self._strata(courts, y, len(tier_names))
        comparisons: list[ComparisonResult] = []
        trends: dict[str, pd.DataFrame] = {}
        trees = {}
        sizes: dict[str, int] = {}
        for name, mask in strata.items():
            console.print(f"[dim]Analysing stratum {name} ({int(mask.sum())} cases)...[/dim]")
            counts = np.bincount(y[mask], minlength=len(tier_names))
            usable = [i for i, c in enumerate(counts) if c >= cfg.min_group_size]
            sizes[name] = int(mask.sum())
            comparisons += self._compare_tiers(name, mask, usable, X, kinds, y, tier_names, courts)

            keep = mask & np.isin(y, usable)
            trends[name] = trend(X[keep].reset_index(drop=True), y[keep])
            remap = {old: new for new, old in enumerate(usable)}
            yk = np.array([remap[v] for v in y[keep]])
            trees[name] = fit_tree(
                X[keep].reset_index(drop=True), yk, [tier_names[i] for i in usable],
                cfg.tree_max_depth, cfg.tree_min_samples_leaf, cfg.n_splits, cfg.n_repeats, cfg.seed,
                label=self._short_label,
            )
            sizes[name] = int(keep.sum())

        court_auc = court_proxy_auc(X, courts, cfg.min_group_size)
        primary = "ALL" if "ALL" in strata else next(iter(strata), "ALL")
        self._analysis = Analysis(
            tier_names=tier_names, tier_counts=tier_counts, court_tier=court_tier,
            comparisons=comparisons, trends=trends, court_auc=court_auc, trees=trees,
            strata_sizes=sizes, warnings=self._warnings(court_tier, tier_counts, strata, courts, y, tier_names),
            alpha=cfg.fdr_alpha, top_n=cfg.top_n, primary=primary, kinds=kinds,
        )
        self._state = {"X": X, "y": y, "tiers": tiers, "courts": courts, "strata": strata, "n_dropped_tiers": dropped}
        self._display()

    # -- helpers ----------------------------------------------------------
    def _strata(self, courts: pd.Series, y: np.ndarray, n_tiers: int) -> dict[str, np.ndarray]:
        """Each court with at least two usable tiers, plus all cases pooled when courts are mixed."""
        out: dict[str, np.ndarray] = {}
        for court in courts.value_counts().index:
            mask = (courts == court).to_numpy()
            counts = np.bincount(y[mask], minlength=n_tiers)
            if (counts >= self._config.min_group_size).sum() >= 2:
                out[court] = mask
        if courts.nunique() > 1 or not out:
            out["ALL"] = np.ones(len(courts), dtype=bool)
        return out

    def _compare_tiers(
        self, stratum: str, mask: np.ndarray, usable: list[int], X: pd.DataFrame,
        kinds: dict[str, str], y: np.ndarray, tier_names: list[str], courts: pd.Series,
    ) -> list[ComparisonResult]:
        min_n = self._config.min_group_size
        specs: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
        for lo, hi in zip(usable, usable[1:]):  # adjacent usable tiers: higher vs lower
            specs.append((
                f"{tier_names[hi]} vs {tier_names[lo]}", tier_names[hi], tier_names[lo],
                mask & (y == hi), mask & (y == lo),
            ))
        for t in usable:  # each tier against all the others
            a, b = mask & (y == t), mask & (y != t)
            if len(usable) > 2 and b.sum() >= min_n:
                specs.append((f"{tier_names[t]} vs all other tiers", tier_names[t], "other tiers", a, b))
        results = []
        for label, name_a, name_b, a, b in specs:
            if a.sum() < min_n or b.sum() < min_n:
                continue
            key = (a.tobytes(), b.tobytes())
            if key in self._seen_comparisons:  # identical cases already compared in a narrower stratum
                continue
            self._seen_comparisons.add(key)
            results.append(ComparisonResult(
                stratum=stratum, label=f"{label} [{stratum}]", name_a=name_a, name_b=name_b,
                n_a=int(a.sum()), n_b=int(b.sum()),
                court_mix_distance=total_variation(courts, a, b),
                table=compare(X, kinds, a, b),
            ))
        return results

    def _warnings(
        self, court_tier: pd.DataFrame, tier_counts: dict[str, int], strata: dict[str, np.ndarray],
        courts: pd.Series, y: np.ndarray, tier_names: list[str],
    ) -> list[str]:
        warns: list[str] = []
        if courts.nunique() > 1:
            for i, ti in enumerate(tier_names):
                for j in range(i + 1, len(tier_names)):
                    tv = total_variation(courts, y == j, y == i)
                    if tv > 0.5:
                        warns.append(f"Tiers '{tier_names[j]}' and '{ti}' come from different courts "
                                     f"(court mix distance {tv:.2f}), so pooled differences between them may be "
                                     "court or era effects, not injury severity.")
        for tier, n in tier_counts.items():
            if n < 10:
                warns.append(f"Tier '{tier}' has only {n} cases: patterns for it are leads to check, not findings.")
        if not [s for s in strata if s != "ALL"]:
            warns.append("No single court has two tiers with enough cases, so there is no within-court comparison. "
                         "Extract more cases of the missing tiers from one court.")
        return warns

    def _takeaways(self) -> list[str]:
        a = self._analysis
        assert a is not None
        explain = self._explain()
        within = [c for c in a.comparisons if c.stratum != "ALL"]
        out: list[str] = [
            f"{len(within)} comparison(s) are free of court confounding "
            + (f"(smallest group {min(min(c.n_a, c.n_b) for c in within)} cases)." if within
               else "— none: no single court has two tiers with enough cases. Everything below is exploratory.")
        ]
        for comp in a.comparisons:
            sig = comp.table[comp.table["q"] < a.alpha] if not comp.table.empty else comp.table
            if len(sig):
                top = sig.reindex(sig["effect"].abs().sort_values(ascending=False).index).iloc[0]
                direction = "more" if top["effect"] > 0 else "less"
                txt = (f"{comp.label}: {len(sig)} significant feature(s); strongest is "
                       f"“{explain(top['feature'])}” ({direction} in {comp.name_a}, effect {top['effect']:+.2f}).")
            else:
                txt = f"{comp.label}: no feature is significant after correcting for multiple testing."
            if comp.court_mix_distance > 0.5:
                txt += f" ⚠ court mix distance {comp.court_mix_distance:.2f}: possibly a court effect."
            out.append(txt)
        for stratum, tr in a.trees.items():
            t, m = tr.cv["tree"]["macro_f1"]["mean"], tr.cv["majority"]["macro_f1"]["mean"]
            verdict = "beats" if t > m + 0.05 else "does not clearly beat"
            out.append(f"Decision tree [{stratum}]: macro-F1 {t:.2f} on unseen cases, which {verdict} "
                       f"always predicting the biggest tier ({m:.2f}).")
        if not out:
            out.append("No comparison had enough cases in both groups.")
        return out

    def _short_label(self, feature: str) -> str:
        """Plain-language name without the bracketed scale hint, for use inside rules."""
        return re.sub(r"\s*\(.*?\)", "", self._explain()(feature))

    def _explain(self):
        if "explain" not in self._state:
            self._state["explain"] = make_explainer(Path(self._config.icd_path))
        return self._state["explain"]

    def _display(self) -> None:
        a = self._analysis
        assert a is not None
        explain = self._explain()

        ct = Table(title="Cases per tier and court")
        ct.add_column("tier", style="cyan")
        for court in a.court_tier.columns:
            ct.add_column(str(court), justify="right")
        for tier, row in zip(a.court_tier.index, a.court_tier.to_numpy()):
            ct.add_row(tier, *map(str, row))
        console.print(ct)
        for w in a.warnings:
            console.print(f"[yellow]⚠ {w}[/yellow]")

        cmp = Table(title="Comparisons (q < %.2f counts as significant)" % a.alpha)
        for col in ("comparison", "cases", "court mix", "significant", "strongest feature (effect)"):
            cmp.add_column(col)
        for comp in a.comparisons:
            df = comp.table
            sig = df[df["q"] < a.alpha] if not df.empty else df
            pool = sig if len(sig) else df
            strongest = ""
            if len(pool):
                top = pool.reindex(pool["effect"].abs().sort_values(ascending=False).index).iloc[0]
                strongest = f"{explain(top['feature'])[:52]} ({top['effect']:+.2f})"
            cmp.add_row(comp.label, f"{comp.n_a} vs {comp.n_b}", f"{comp.court_mix_distance:.2f}",
                        str(len(sig)), strongest)
        console.print(cmp)

        for stratum, tr in a.trees.items():
            t = Table(title=f"Decision-tree rules [{stratum}]  "
                            f"(macro-F1 {tr.cv['tree']['macro_f1']['mean']:.2f} vs "
                            f"{tr.cv['majority']['macro_f1']['mean']:.2f} majority)")
            for col in ("rule", "predicts", "cases", "right"):
                t.add_column(col)
            for _, r in tr.rules.iterrows():
                t.add_row(r["rule"][:110], r["predicts"], str(r["n_cases"]), f"{r['precision']:.0%}")
            console.print(t)

    # -- persistence ------------------------------------------------------
    @override
    def save_results(self) -> None:
        a = self._analysis
        if a is None:
            raise RuntimeError("run() must be called before save_results()")
        cfg = self._config
        out = Path(cfg.output_dir)
        plot_dir = out / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        explain = self._explain()
        X, tiers = self._state["X"], self._state["tiers"]

        # tables
        frames = []
        for comp in a.comparisons:
            if comp.table.empty:
                continue
            t = comp.table.copy()
            t.insert(0, "comparison", comp.label)
            t.insert(1, "stratum", comp.stratum)
            t.insert(2, "court_mix_distance", comp.court_mix_distance)
            t.insert(4, "meaning", t["feature"].map(explain))
            frames.append(t.drop(columns=["ci_a", "ci_b"]))
        if frames:
            pd.concat(frames).to_csv(out / "feature_comparisons.csv", index=False)
        pd.concat([df.assign(stratum=s, meaning=df["feature"].map(explain)) for s, df in a.trends.items() if not df.empty]
                  or [pd.DataFrame()]).to_csv(out / "feature_trends.csv", index=False)
        pd.concat([tr.rules.assign(stratum=s) for s, tr in a.trees.items()]).to_csv(out / "tree_rules.csv", index=False)
        pd.concat([tr.stability.assign(stratum=s) for s, tr in a.trees.items()]).to_csv(out / "tree_stability.csv", index=False)
        a.court_tier.to_csv(out / "court_by_tier.csv")
        if len(a.court_auc):
            a.court_auc.rename("court_auc").to_frame().assign(meaning=lambda d: d.index.map(explain)).to_csv(out / "court_proxy_auc.csv")

        # figures
        plots.court_by_tier(a.court_tier.T.T, plot_dir / "court_by_tier.png")
        primary_mask = self._state["strata"].get(a.primary, np.ones(len(X), dtype=bool))
        trend_df = a.trends.get(a.primary, pd.DataFrame())
        if not trend_df.empty:
            feats = trend_df.reindex(trend_df["rho"].abs().sort_values(ascending=False).index)["feature"].head(cfg.profile_features).tolist()
            plots.tier_profile(X[primary_mask], tiers[primary_mask], feats, a.kinds, a.tier_names, explain,
                               plot_dir / "tier_profile.png")
        for stratum in {c.stratum for c in a.comparisons}:
            plots.top_differences({c.label: c.table for c in a.comparisons if c.stratum == stratum},
                                  explain, a.alpha, cfg.top_n, plot_dir / f"top_differences_{stratum}.png")
        for stratum, tr in a.trees.items():
            plots.decision_tree(tr, explain, plot_dir / f"decision_tree_{stratum}.png")
            plots.tree_stability(tr.stability, explain, plot_dir / f"tree_stability_{stratum}.png")

        takeaways = self._takeaways()
        (out / "report.md").write_text(build_report(a, explain, takeaways), encoding="utf-8")

        summary = {
            "id": cfg.id, "config": cfg.model_dump(),
            "tier_counts": a.tier_counts, "court_by_tier": a.court_tier.to_dict(),
            "warnings": a.warnings, "takeaways": takeaways,
            "comparisons": [{
                "label": c.label, "stratum": c.stratum, "n_a": c.n_a, "n_b": c.n_b,
                "court_mix_distance": c.court_mix_distance,
                "n_significant": int((c.table["q"] < a.alpha).sum()) if not c.table.empty else 0,
            } for c in a.comparisons],
            "trees": {s: {"cv": tr.cv, "rules": tr.rules.to_dict("records")} for s, tr in a.trees.items()},
        }
        (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

        console.print(f"\n[bold green]Report: {out / 'report.md'}[/bold green]")
        console.print(f"[green]Tables and figures: {out}[/green]")
