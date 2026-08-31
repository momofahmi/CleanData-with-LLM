"""
Reference = the entity's OWN neighbouring years (series are smooth), and the
cross-section of other entities in the same year.
  year_collapse     : in year y, most entities' value drops to ~0 vs their neighbours
  window_growth     : one entity's values in a contiguous window are a constant
                      factor (or compound factor) above the trend fitted outside
  rough_series      : an entity's series is far rougher than peers (values drawn
                      at random into a band leave a signature: no autocorrelation)
Also produces entity rankings per column so the LLM can apply world knowledge
(e.g. landlocked countries at the top of fish consumption).
"""
import numpy as np
import pandas as pd


def _interp_neighbors(g, t, col):
    """value predicted from neighbours (t-1, t+1) if both exist."""
    s = g.set_index(t)[col]
    idx = s.index
    prev = s.shift(1); nxt = s.shift(-1)
    return pd.DataFrame({"y": s, "pred": (prev + nxt) / 2, "prev": prev, "next": nxt}, index=idx)


def year_collapse(df, ent, t, numerics):
    findings = []
    for col in numerics:
        if col == t:
            continue
        rows = []
        for e, g in df.groupby(ent):
            g = g.sort_values(t)
            if len(g) < 5:
                continue
            k = _interp_neighbors(g, t, col)
            k = k[(k["pred"] > 0)]
            k["ratio"] = k["y"] / k["pred"]
            rows.append(k[["ratio"]].assign(entity=e))
        if not rows:
            continue
        R = pd.concat(rows)
        by_year = R.groupby(level=0)["ratio"].agg(lambda r: (r < 0.2).mean())
        n_by_year = R.groupby(level=0)["ratio"].size()
        for y, frac in by_year.items():
            if n_by_year[y] >= 20 and frac >= 0.6:
                findings.append(dict(kind="year_collapse", slicer=t, level=int(y), target=col, n=int(n_by_year[y]),
                                     evidence=f"{frac:.0%} of entities have {col} < 20% of their neighbouring-year average in {y}",
                                     confidence="REPAIR", repair={"type": "interpolate_year", "time": t, "entity": ent, "year": int(y), "target": col}))
    return findings


def window_growth(df, ent, t, numerics, min_len=3, max_len=12, factor_thresh=1.15):
    findings = []
    for e, g in df.groupby(ent):
        g = g.sort_values(t)
        if len(g) < 12:
            continue
        years = g[t].to_numpy()
        for col in numerics:
            if col == t:
                continue
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if len(v) == 0 or np.isnan(v).any() or v.min() <= 0:
                continue
            lv = np.log(v)
            # prefilter by rolling median: if the series is smooth, no window can be a constant factor above the trend
            rm = pd.Series(lv).rolling(21, center=True, min_periods=7).median().to_numpy()
            if np.nanmax(np.abs(lv - rm)) < 0.6 * np.log(factor_thresh):
                continue
            best = None
            for L in range(min_len, min(max_len, len(v) - 6) + 1):
                for i in range(3, len(v) - L - 2):
                    inside = np.arange(i, i + L)
                    outside = np.concatenate([np.arange(0, i), np.arange(i + L, len(v))])
                    a, b = np.polyfit(years[outside], lv[outside], 1)
                    resid = lv[inside] - (a * years[inside] + b)
                    out_resid = lv[outside] - (a * years[outside] + b)
                    noise = np.std(out_resid) + 1e-6
                    if np.median(resid) > np.log(factor_thresh) and np.percentile(resid, 25) > 3 * noise:
                        # constant factor vs compound 
                        slope = np.polyfit(np.arange(L), resid, 1)[0] if L > 2 else 0.0
                        cand = dict(start=int(years[i]), end=int(years[i + L - 1]), factor=float(np.exp(np.median(resid))),
                                    per_year=float(np.exp(slope)), score=float(np.median(resid) / noise))
                        if best is None or cand["score"] > best["score"]:
                            best = cand
            if best:
                mech = "compound" if best["per_year"] > 1.08 else "constant"
                findings.append(dict(kind="window_growth", entity=e, target=col, slicer=ent, level=e, n=best["end"] - best["start"] + 1,
                                     evidence=f"{col} for {e} is x{best['factor']:.2f} above own trend during {best['start']}-{best['end']} ({mech}; per-year x{best['per_year']:.2f})",
                                     confidence="REPAIR" if best["score"] > 4 else "REVIEW",
                                     repair={"type": "divide_window", "entity_col": ent, "entity": e, "time": t, "start": best["start"], "end": best["end"], "target": col,
                                             "factor": best["factor"], "per_year": best["per_year"] if mech == "compound" else None}))
    return findings


def merge_entity_windows(findings, min_cols=3):
    """Several columns of the same entity inflated by a similar factor over overlapping
    windows = one entity-wide mechanism (e.g. 'total consumption +30%'). Merge them."""
    from collections import defaultdict
    by_ent = defaultdict(list)
    out = []
    for f in findings:
        (by_ent[f["entity"]] if f["kind"] == "window_growth" else out).append(f)
    for e, fs in by_ent.items():
        if len(fs) < min_cols:
            out.extend(fs); continue
        starts = [f["repair"]["start"] for f in fs]; ends = [f["repair"]["end"] for f in fs]
        facs = [f["repair"]["factor"] for f in fs]
        if (max(facs) / min(facs)) > 1.15:
            out.extend(fs); continue
        start, end, fac = int(np.median(starts)), int(np.median(ends)), float(np.median(facs))
        cols = [f["target"] for f in fs]
        base = fs[0]
        out.append(dict(kind="entity_window_growth", entity=e, slicer=base["slicer"], level=e, target=",".join(cols),
                        n=end - start + 1, confidence="REPAIR",
                        evidence=f"{len(cols)} columns ({', '.join(cols)}) of {e} are all x{fac:.2f} above their own trend over ~{start}-{end}: an entity-wide multiplicative window (e.g. 'total consumption inflated')",
                        repair={"type": "divide_window_multi", "entity_col": base["repair"]["entity_col"], "entity": e, "time": base["repair"]["time"],
                                "start": start, "end": end, "targets": cols, "factor": fac}))
    return out


def demote_mass_windows(findings, per_col=5, total=12):
    """Real-world panels are full of genuine trend breaks (crises, wars, policy).
    If MANY entities fire window_growth on the same column, that is the texture of
    reality, not one injected error -> demote unmerged single-column windows to REVIEW.
    Entity-merged findings (>=3 columns, same factor & window) keep REPAIR: an
    entity-wide constant multiplication is the signature of an injection, and mass
    natural breaks almost never align across columns."""
    from collections import Counter
    singles = [f for f in findings if f["kind"] == "window_growth"]
    per_column = Counter(f["target"] for f in singles)
    mass = len(singles) >= total or any(v >= per_col for v in per_column.values())
    if mass:
        for f in singles:
            if f["confidence"] == "REPAIR":
                f["confidence"] = "REVIEW"
                f["evidence"] += (f" | NOTE: {len(singles)} single-column windows fired across the panel "
                                  f"({dict(per_column)}) - this density looks like natural trend breaks; "
                                  "verify against world knowledge before repairing")
    return findings


def rough_series(df, ent, t, numerics):
    findings = []
    for col in numerics:
        if col == t:
            continue
        stats = []
        for e, g in df.groupby(ent):
            v = pd.to_numeric(g.sort_values(t)[col], errors="coerce").dropna()
            if len(v) < 8 or (v <= 0).any():
                continue
            d = np.abs(np.diff(np.log(v.to_numpy())))
            stats.append((e, float(np.median(d)), float(v.median())))
        if len(stats) < 20:
            continue
        S = pd.DataFrame(stats, columns=["entity", "rough", "level"])
        med, mad = S["rough"].median(), (S["rough"] - S["rough"].median()).abs().median() + 1e-9
        S["z"] = (S["rough"] - med) / (1.4826 * mad)
        hi_level = S["level"].quantile(0.75)
        for _, r in S[(S["z"] > 4) & (S["level"] >= hi_level)].iterrows():
            findings.append(dict(kind="rough_series", entity=r["entity"], slicer=ent, level=r["entity"], target=col, n=0,
                                 evidence=f"{col} series for {r['entity']} is {r['z']:.1f} robust-SD rougher than peers (median |dlog| {r['rough']:.2f} vs {med:.2f}) while at a high level - looks like values drawn at random rather than a real time series",
                                 confidence="REVIEW", repair={"type": "flag_entity", "entity_col": ent, "entity": r["entity"], "target": col}))
    return findings


def entity_rankings(df, ent, numerics, k=12):
    lines = []
    for col in numerics:
        m = df.groupby(ent)[col].median().sort_values(ascending=False)
        top = ", ".join(f"{e}:{v:.3g}" for e, v in m.head(k).items())
        lines.append(f"Top {k} entities by median {col}: {top}")
    return "\n".join(lines)
