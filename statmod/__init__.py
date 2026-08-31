"""
statmod - the statistical module. audit(df, cfg) to (findings, report_text)
profiler column roles, derived features, per-column stats
slice audit the vertical pass: shifts, overwrites, MNAR missingness, and the functional-dependency checks
panel audit the temporal pass, when the dataset declares entity and time columns
"""
import pandas as pd
from .profiler import profile
from . import slice_audit as SA
from . import panel_audit as PA
from .repair import repair_snippet, snippet_with_derived, apply_all

MAX_FINDINGS_IN_REPORT = 25
KIND_ORDER = {"categorical_overwrite": 0, "entity_window_growth": 0, "fd_violation": 1, "mnar_missingness": 1, "additive_shift": 2, "multiplicative_shift": 3,
              "collapse": 4, "year_collapse": 5, "window_growth": 6, "rough_series": 7, "unexplained_shift": 8}


def audit(df, cfg, verbose=True):
    df = df.copy()
    prof = profile(df, cfg)
    slicers, nums, cats = prof["slicers"], prof["numerics"], prof["categoricals"]
    if verbose:
        print(f"[statmod] roles: {sum(1 for r in prof['roles'].values() if r=='numeric')} numeric, "
              f"{len(cats)} categorical, {len(slicers)} slicers, derived={prof['derived']}")
    findings = []
    findings += SA.numeric_audit(df, slicers, nums, cfg.target)
    findings += SA.categorical_audit(df, slicers, cats, cfg.target)
    texts = [c for c, r in prof["roles"].items() if r == "text"]      # high-cardinality discrete (e.g. country, 165 codes)
    findings += SA.missingness_audit(df, slicers, cats + texts, cfg.target)
    panel_text = ""
    if cfg.entity_col and cfg.time_col and cfg.entity_col in df.columns:
        ent, t = cfg.entity_col, cfg.time_col
        pnums = [c for c in nums if c != t]
        yc = PA.year_collapse(df, ent, t, pnums)
        for f in yc:
            f["repair"]["all_years"] = sorted({g["repair"]["year"] for g in yc if g["target"] == f["target"]})
        findings += yc
        findings += PA.demote_mass_windows(PA.merge_entity_windows(PA.window_growth(df, ent, t, pnums)))
        findings += PA.rough_series(df, ent, t, pnums)
        panel_text = PA.entity_rankings(df, ent, pnums)
    findings = _postprocess(findings)
    feats = set(cfg.numeric_features + cfg.categorical_features)
    findings = _rank(findings, feats)
    reviews = [f for f in findings if f["confidence"] == "REVIEW"]
    findings = [f for f in findings if f["confidence"] == "REPAIR"] + reviews[:8]
    cross = SA.cross_summaries(df, slicers, nums, cats, cfg.target)
    report = _render(prof, findings, cross, panel_text)
    if verbose:
        print(f"[statmod] {len(findings)} findings ({sum(f['confidence']=='REPAIR' for f in findings)} REPAIR, "
              f"{sum(f['confidence']=='REVIEW' for f in findings)} REVIEW)")
    return findings, report


def _postprocess(findings):
    """(a) A mechanism found on >=3 levels of the same slicer for the same target is more
    likely natural structure (seasonality, hotel type...) than an injected error on one
    slice -> demote to REVIEW. (b) Additive/multiplicative findings come in mirror pairs
    (A vs B: +d, B vs A: -d) - which slice is the corrupted one is not identifiable from
    the numbers alone; annotate so the agent tests both directions."""
    from collections import Counter
    fam = lambda f: (f["kind"], f["slicer"], f["target"]) if f["kind"] in ("additive_shift", "multiplicative_shift", "collapse", "categorical_overwrite") else None
    counts = Counter(fam(f) for f in findings if fam(f))
    for f in findings:
        if fam(f) and counts[fam(f)] >= 3 and f["confidence"] == "REPAIR":
            f["confidence"] = "REVIEW"
            f["evidence"] += f" | NOTE: {counts[fam(f)]} levels of {f['slicer']} show a mechanism on {f['target']} - looks like natural structure rather than one corrupted slice"
    pairs = {}
    for f in findings:
        if f["kind"] in ("additive_shift", "multiplicative_shift") and f.get("ref") is not None:
            pairs.setdefault((f["kind"], f["slicer"], f["target"], frozenset([str(f["level"]), str(f["ref"])])), []).append(f)
    for grp in pairs.values():
        if len(grp) >= 2:
            for f in grp:
                f["evidence"] += " | DIRECTION AMBIGUOUS: the mirror hypothesis (other slice corrupted) is also listed - test both, keep the one that improves the score"
    return findings


def _rank(findings, feats=frozenset()):
    def on_feature(f):
        return any(t in feats for t in str(f["target"]).split(","))
    def key(f):
        return (0 if f["confidence"] == "REPAIR" else 1,
                0 if on_feature(f) else 1,                     # model features first: they can move the score
                KIND_ORDER.get(f["kind"], 9), -f.get("n", 0))
    return sorted(findings, key=key)


def _fmt(f):
    where = f"{f['slicer']}={f['level']}"
    if f.get("context") is not None:
        where += f" & {f['context']}={f['context_level']}"
    if f.get("ref") is not None:
        where += f" (vs {f['slicer']}={f['ref']})"
    s = f"[{f['confidence']}] {f['kind']} on {f['target']} where {where} (n={f['n']})\n    evidence: {f['evidence']}"
    if f.get("repair"):
        s += "\n    suggested code:\n" + "\n".join("        " + l for l in snippet_with_derived(f["repair"]).splitlines())
    return s


def _cap(text, limit):
    return text if len(text) <= limit else text[:limit] + "\n...[section truncated]"


def _render(prof, findings, cross, panel_text):
    head = ("=== STATISTICAL AUDIT REPORT (computed on the training data only; no ground truth, no hints) ===\n"
            "How to read: each finding compares a SLICE of rows against a reference and reports the simplest "
            "mechanism that explains the difference. REPAIR = a one-parameter mechanism re-aligns the "
            "distributions (strong evidence of an injected systematic error). REVIEW = a real difference "
            "exists but no simple mechanism explains it (may be natural heterogeneity - do not blindly change).")
    profile = "--- COLUMN PROFILE ---\n" + prof["text"]
    if prof["derived"]:
        profile += f"\n(derived helper columns available in the profile only, do NOT add them to the saved csv: {prof['derived']})"
    body = [_fmt(f) for f in findings[:MAX_FINDINGS_IN_REPORT]] or ["(no systematic slice-level anomalies detected)"]
    find_txt = f"--- FINDINGS ({len(findings)} total, top {min(len(findings), MAX_FINDINGS_IN_REPORT)} shown; model-feature columns first) ---\n" + "\n".join(body)
    # reserved budgets: profile 3500 | findings 6800 | rankings 1300 | cross summaries 2000
    parts = [head, "", _cap(profile, 3500), "", _cap(find_txt, 6800)]
    if panel_text:
        parts += ["", _cap("--- ENTITY RANKINGS (apply your world knowledge: does the ranking make sense?) ---\n" + panel_text, 1300)]
    if cross:
        parts += ["", _cap("--- CROSS-COLUMN SUMMARIES (per slice level; look for a level whose numbers differ from its siblings by a CONSTANT) ---\n" + cross, 2000)]
    parts += ["", "=== END OF REPORT ==="]
    return "\n".join(parts)
