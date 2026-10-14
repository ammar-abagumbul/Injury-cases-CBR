"""Experiment 7 — turn the analysis into a readable Markdown report."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .analysis import Analysis, ComparisonResult, TreeReport

_PROXY_AUC_GAP = 0.30  # |AUC - 0.5| above this means the feature mostly tells courts apart


def _cell(text: object) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_stat(row: pd.Series, side: str) -> str:
    stat, n, stated = row[f"stat_{side}"], int(row[f"n_{side}"]), row[f"stated_{side}"]
    if row["kind"] == "binary":
        return f"{stat:.0%} ({round(stat * n)}/{n})"
    if stated < 0.995:  # a figure the judgment only sometimes gives
        return f"median {stat:.3g} (stated in {stated:.0%})"
    return f"median {stat:.3g}, mean {row[f'mean_{side}']:.2g}"


def _flags(feature: str, comp: ComparisonResult, court_auc: pd.Series) -> str:
    notes = []
    if feature.endswith("_stated"):
        notes.append("document style: whether the judgment mentions it")
    auc = court_auc.get(feature)
    if auc is not None and abs(auc - 0.5) > _PROXY_AUC_GAP and comp.court_mix_distance > 0.3:
        notes.append(f"court proxy (AUC {auc:.2f})")
    return "; ".join(notes)


def _comparison_section(comp: ComparisonResult, a: Analysis, explain: Callable[[str], str]) -> list[str]:
    lines = [
        f"#### {comp.label}  (n = {comp.n_a} vs {comp.n_b}, "
        f"court mix distance {comp.court_mix_distance:.2f})"
    ]
    df = comp.table
    if df.empty:
        return [*lines, "No feature could be compared (too few cases state any of them).", ""]
    sig = df[df["q"] < a.alpha]
    if len(sig):
        shown = sig.reindex(sig["effect"].abs().sort_values(ascending=False).index).head(a.top_n)
        lines.append(f"{len(sig)} feature(s) pass the false-discovery threshold (q < {a.alpha}); strongest shown.")
    else:
        shown = df.sort_values("p").head(8)
        lines.append(f"**No feature passes the false-discovery threshold (q < {a.alpha}).** "
                     "Showing the 8 smallest p-values as exploratory hints only.")
    rows = [
        [explain(r["feature"]), _fmt_stat(r, "a"), _fmt_stat(r, "b"), f"{r['effect']:+.2f}",
         _fmt_p(r["p"]), _fmt_p(r["q"]), _flags(r["feature"], comp, a.court_auc)]
        for _, r in shown.iterrows()
    ]
    lines += ["", _table(["Feature", f"In {comp.name_a}", f"In {comp.name_b}", "Effect", "p", "q", "Notes"], rows), ""]
    return lines


def _tree_section(stratum: str, tr: TreeReport, n: int) -> list[str]:
    cv = tr.cv
    lines = [f"### Decision rules: {stratum} ({n} cases)", ""]
    lines += [f"_{note}_" for note in tr.notes]
    rows = [
        [r["rule_plain"], r["predicts"], r["n_cases"],
         ", ".join(f"{c} {r[f'n_{c}']}" for c in tr.class_names), f"{r['precision']:.0%}"]
        for _, r in tr.rules.iterrows()
    ]
    lines += [_table(["If ...", "predict", "cases", "actual tiers in this group", "right"], rows), ""]

    def pm(d: dict) -> str:
        return f"{d['mean']:.2f} ± {d['std']:.2f}"

    lines += [
        f"**How far to trust the rules** (same tree refit on {cv['n_fits']} cross-validation splits, "
        "scored on cases it did not see):",
        "",
        _table(["", "macro-F1", "QWK", "balanced accuracy", "accuracy"], [
            ["decision tree"] + [pm(cv["tree"][k]) for k in ("macro_f1", "qwk", "balanced_accuracy", "accuracy")],
            ["always predict the biggest tier"] + [pm(cv["majority"][k]) for k in ("macro_f1", "qwk", "balanced_accuracy", "accuracy")],
        ]),
        "",
        "Recall by tier on unseen cases: "
        + ", ".join(f"{c} {r:.0%}" for c, r in zip(tr.class_names, cv["recall_tree"])) + ".",
        "",
    ]
    return lines


def build_report(a: Analysis, explain: Callable[[str], str], takeaways: list[str]) -> str:
    n = sum(a.tier_counts.values())
    out = [
        "# What makes a case fall in each severity tier",
        "",
        f"{n} cases with a human severity label: "
        + ", ".join(f"{t} {c}" for t, c in a.tier_counts.items()) + ".",
        "Features come from the LLM-extracted case JSON; the tier comes from the human annotation.",
        "",
        "## Key takeaways",
        "",
        *[f"- {t}" for t in takeaways],
        "",
        "## 1. Read this first: data and caveats",
        "",
        "Cases per tier and court:",
        "",
        _table(["tier", *a.court_tier.columns], [[t, *row] for t, row in zip(a.court_tier.index, a.court_tier.to_numpy())]),
        "",
        *[f"- ⚠ {w}" for w in a.warnings],
        "",
    ]
    if len(a.court_auc):
        gap = (a.court_auc - 0.5).abs().sort_values(ascending=False)
        proxies = [(f, a.court_auc[f]) for f in gap.index if gap[f] > _PROXY_AUC_GAP][:8]
        if proxies:
            out += [
                "Features that almost perfectly tell the two biggest courts apart (AUC far from 0.5). "
                "Because tier and court overlap in this data, these cannot yet be credited to severity "
                "rather than to court; only within-court comparisons can separate the two:",
                "",
                *[f"- {explain(f)}: AUC {auc:.2f}" for f, auc in proxies],
                "",
            ]

    within = [c for c in a.comparisons if c.stratum != "ALL"]
    pooled = [c for c in a.comparisons if c.stratum == "ALL"]
    out += ["## 2. Comparisons within a single court (the trustworthy ones)", ""]
    if within:
        out += ["Court cannot explain these differences, because every case in the comparison is from the same court.", ""]
        for c in within:
            out += _comparison_section(c, a, explain)
    else:
        out += ["_No court has two tiers with enough cases to compare. Extract more cases of the "
                "missing tiers from the same court (for example High Court non-serious and serious cases)._", ""]

    out += ["## 3. All cases pooled", ""]
    if pooled:
        out += ["Read the court mix distance of each comparison: close to 1 means the two groups come from "
                "different courts, and differences may reflect the court, not the injury.", ""]
        for c in pooled:
            out += _comparison_section(c, a, explain)
    else:
        out += ["_Not run (a single court, so section 2 already covers it)._", ""]

    out += ["## 4. Features that rise or fall with severity", "",
            "Spearman correlation of each feature with the tier order (+1 = always higher in higher tiers).", ""]
    for stratum, df in a.trends.items():
        if df.empty:
            continue
        sig = df[df["q"] < a.alpha]
        shown = (sig if len(sig) else df).reindex(
            (sig if len(sig) else df)["rho"].abs().sort_values(ascending=False).index).head(a.top_n)
        out += [f"**{stratum}** — " + (f"{len(sig)} feature(s) pass q < {a.alpha}." if len(sig)
                else f"none pass q < {a.alpha}; strongest shown as exploratory."), "",
                _table(["Feature", "correlation", "p", "q", "cases"],
                       [[explain(r["feature"]), f"{r['rho']:+.2f}", _fmt_p(r["p"]), _fmt_p(r["q"]), int(r["n"])]
                        for _, r in shown.iterrows()]), ""]

    out += ["## 5. Decision rules", "",
            "A shallow, class-balanced decision tree gives if/then rules for the tiers. "
            "Rules from the pooled data inherit any court confound.", ""]
    for stratum, tr in a.trees.items():
        out += _tree_section(stratum, tr, a.strata_sizes[stratum])
        top = tr.stability.head(5)
        out += ["Most stable first splits across the cross-validation trees: "
                + ", ".join(f"{explain(r['feature'])} ({r['share_as_first_split']:.0%})" for _, r in top.iterrows()
                            if r["share_as_first_split"] > 0) + ".", ""]

    out += [
        "## 6. How to read the numbers", "",
        "- **Effect** is the difference in share of cases having the feature (flags), or Cliff's delta "
        "(numbers: the chance a random case from the first group is higher than one from the second, minus the reverse). "
        "Both run from -1 to +1; positive means more in the first group.",
        "- **p** is from Fisher's exact test (flags) or the Mann-Whitney test (numbers). **q** adjusts p for testing many "
        "features at once (Benjamini-Hochberg); trust q, not p.",
        "- A feature that is \"stated\" or not is a property of the judgment text as well as the injury.",
        "- Small tiers give wide uncertainty. A pattern in fewer than about 10 cases is a lead to check, not a finding.",
        "",
    ]
    return "\n".join(out)
