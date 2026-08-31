"""
For every (slicer, level, target) triple we compare the slice against a
reference (pairwise vs other levels when few levels, else vs the complement),
and only report a REPAIR when a ONE-PARAMETER MECHANISM explains the difference:
    additive shift      x -> x + d        (verified: subtracting d aligns the distributions)
    multiplicative      x -> x * r        (verified: dividing by r aligns)
    collapse            x -> ~constant    (variance vanishes inside the slice)
    categorical overwrite  cat -> one value at ~100% inside a (context, slice) cell
    MNAR missingness    NaN rate jumps in some slices; imputation distribution derived
                        so that the slice re-aligns with the clean reference slice.
Natural heterogeneity (different SHAPE, offset that varies across strata) does
not pass the alignment test and is only FLAGGED for review, never repaired.
"""
import numpy as np
import zlib
import pandas as pd
from scipy.stats import ks_2samp

MIN_N = 80


def _ks(a, b):
    return ks_2samp(a, b).statistic if len(a) > 5 and len(b) > 5 else 0.0


def _refine_delta(a, b, d0, scale):
    """delta* = argmin_d KS(a - d, b) around the initial estimate (coarse-to-fine grid)."""
    best_d, best_D = d0, _ks(a - d0, b)
    span = max(0.15 * scale, abs(d0) * 0.5, 1e-9)
    for step in (span / 10, span / 100):
        grid = np.linspace(best_d - 10 * step, best_d + 10 * step, 21)
        for d in grid:
            D = _ks(a - d, b)
            if D < best_D:
                best_d, best_D = float(d), D
    return best_d


def _stratified_offsets(df, mask_a, mask_b, target, strata_col):
    offs = []
    for s in df.loc[mask_a | mask_b, strata_col].dropna().unique():
        ga = df.loc[mask_a & (df[strata_col] == s), target].dropna()
        gb = df.loc[mask_b & (df[strata_col] == s), target].dropna()
        if len(ga) >= 20 and len(gb) >= 20:
            offs.append((ga.median() - gb.median(), ga.median() / gb.median() if gb.median() != 0 else np.nan))
    return offs


def _best_strata(df, slicer, slicers):
    """Choose the stratification column that gives the most WELL-POPULATED cells
    (slicer level x stratum with >= 100 rows): stratified medians are then stable."""
    best, best_key = None, None
    for c in slicers:
        if c == slicer:
            continue
        n = df[c].nunique(dropna=True)
        if not (3 <= n <= 15):
            continue
        cells = pd.crosstab(df[slicer], df[c])
        key = (int((cells >= 100).sum().sum()), -float(df[c].isna().mean()))
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def rank_slicers(df, slicers, cap=12):
    """Prefer plausible acquisition dimensions: time-like / channel-like names,
    then moderate cardinality (3-15 levels), then binaries. Cap the count."""
    KEY = ("year", "month", "date", "time", "channel", "source", "segment", "region", "hotel",
           "type", "site", "device", "batch", "title", "class", "group", "country")
    def score(c):
        n = df[c].nunique(dropna=True)
        s0 = 3 if any(k in c.lower() for k in KEY) else 0
        s1 = 2 if 3 <= n <= 15 else (1 if n == 2 else 0)
        return (s0 + s1, -n)
    return sorted(slicers, key=score, reverse=True)[:cap]


def numeric_audit(df, slicers, numerics, target_col, log=None):
    findings = []
    slicers = rank_slicers(df, slicers)
    numerics = [t for t in numerics if t not in slicers and t != target_col]
    for s in slicers:
        levels = [l for l in df[s].dropna().unique() if (df[s] == l).sum() >= MIN_N]
        if len(levels) < 2:
            continue
        pairwise = len(levels) <= 6
        strata = _best_strata(df, s, slicers)
        for t in numerics:
            x = pd.to_numeric(df[t], errors="coerce")
            if x.notna().sum() < 2 * MIN_N:
                continue
            work = pd.DataFrame({"s": df[s], "x": x})
            if strata is not None:
                work["k"] = df[strata]
                medtab = work.dropna().groupby(["s", "k"])["x"].agg(["median", "size"])
            for lv in levels:
                mask_a = work["s"] == lv
                refs = [l for l in levels if l != lv] if pairwise else [None]
                for ref in refs:
                    mask_b = (work["s"] == ref) if ref is not None else (~mask_a & work["s"].notna())
                    a_ = work.loc[mask_a, "x"].dropna(); b_ = work.loc[mask_b, "x"].dropna()
                    if len(a_) < MIN_N or len(b_) < MIN_N:
                        continue
                    scale = max(float(b_.std()), 1e-9)
                    # ---- collapse ----
                    if a_.std() < 0.05 * scale and abs(a_.median() - b_.median()) > 0.5 * scale:
                        findings.append(dict(kind="collapse", slicer=s, level=lv, ref=ref, target=t, n=int(len(a_)),
                                             evidence=f"std {a_.std():.3g} vs {b_.std():.3g}, med {a_.median():.3g} vs {b_.median():.3g}",
                                             confidence="REPAIR", repair={"type": "impute_from_reference", "slicer": s, "level": lv, "target": t}))
                        continue
                    # ---- stratified consistent offset (cheap, from the median table) ----
                    offs = []
                    if strata is not None and lv in medtab.index.get_level_values(0):
                        ma = medtab.loc[lv]
                        if ref is not None:
                            mb = medtab.loc[ref] if ref in medtab.index.get_level_values(0) else None
                        else:
                            other = work.loc[mask_b].dropna().groupby("k")["x"].agg(["median", "size"])
                            mb = other
                        if mb is not None:
                            common = ma.index.intersection(mb.index)
                            for k in common:
                                if ma.loc[k, "size"] >= 20 and mb.loc[k, "size"] >= 20:
                                    offs.append((ma.loc[k, "median"] - mb.loc[k, "median"],
                                                 ma.loc[k, "median"] / mb.loc[k, "median"] if mb.loc[k, "median"] != 0 else np.nan))
                    if len(offs) >= 4:
                        d_arr = np.array([o[0] for o in offs]); r_arr = np.array([o[1] for o in offs])
                        delta = float(np.median(d_arr)); iqr = float(np.subtract(*np.percentile(d_arr, [75, 25])))
                        sign = max((d_arr > 0).mean(), (d_arr < 0).mean())
                        ratio = float(np.nanmedian(r_arr)) if np.isfinite(r_arr).any() else np.nan
                        detected = abs(delta) > 0.10 * scale and sign >= 0.85
                    else:
                        delta = float(a_.median() - b_.median()); iqr = np.nan; sign = 1.0
                        ratio = float(a_.median() / b_.median()) if b_.median() != 0 else np.nan
                        floor0 = 1.36 * np.sqrt((len(a_) + len(b_)) / (len(a_) * len(b_)))
                        detected = abs(delta) > 0.10 * scale and _ks(a_, b_) >= 2 * floor0
                    if not detected:
                        continue
                    delta = _refine_delta(a_, b_, delta, scale)
                    if (x.dropna() % 1 == 0).all():      # integer-valued column -> integer offset
                        di = float(round(delta))
                        if _ks(a_ - di, b_) <= _ks(a_ - delta, b_) + 0.005:
                            delta = di
                    D0 = _ks(a_, b_)
                    # ---- v1.1 calibrated acceptance  ----------------
                    # (i) CROSS-FIT: the shift is estimated on one half of the slice and
                    #     the residual measured on the other (both directions, worst kept),
                    #     so the fit can no longer flatter its own residual.
                    # (ii) tau_clean: permutation noise floor - the 95th percentile of
                    #     KS distances between random clean-vs-clean splits of the
                    #     reference at matched sizes (200 draws, groups capped at 2000/4000
                    #     for tractability). Empirical, so integer/tied columns are handled.
                    # (iii) CONJUNCTION: accept only if the cross-fitted residual is
                    #     <= 0.25*D0 AND <= tau_clean AND < 0.05 (absolute cap kept).
                    # The applied repair still uses the full-sample estimate; only the
                    # ACCEPTANCE decision is cross-fitted. Detection is unchanged.
                    rng = np.random.default_rng(zlib.crc32(f"{s}|{lv}|{t}|{ref}".encode()) & 0xFFFFFFFF)
                    is_int = bool((x.dropna() % 1 == 0).all())
                    na_c = int(min(len(a_) // 2, 2000))
                    nb_c = int(min(len(b_), 4000))
                    b_eval = b_.sample(nb_c, random_state=int(rng.integers(2**31))) if len(b_) > nb_c else b_

                    def _fold_resid(est, ev, mul=False):
                        if mul:
                            r_ = float(est.median() / b_.median()) if b_.median() != 0 else np.nan
                            return _ks(ev / r_, b_eval) if r_ and np.isfinite(r_) and r_ > 0 else 1.0
                        d_ = _refine_delta(est, b_, float(est.median() - b_.median()), scale)
                        if is_int:
                            dd = float(round(d_))
                            if _ks(est - dd, b_) <= _ks(est - d_, b_) + 0.005:
                                d_ = dd
                        return _ks(ev - d_, b_eval)

                    pidx = rng.permutation(len(a_)); half = len(a_) // 2
                    aA = a_.iloc[pidx[:half]].iloc[:na_c] if half else a_
                    aB = a_.iloc[pidx[half:]].iloc[:na_c] if half else a_
                    D_add = max(_fold_resid(aA, aB), _fold_resid(aB, aA))
                    D_mul = max(_fold_resid(aA, aB, True), _fold_resid(aB, aA, True))

                    bv = b_eval.to_numpy(dtype=float, copy=True)
                    m = int(min(na_c if na_c else len(a_), len(bv) // 2))
                    perm = []
                    if m >= 10:
                        for _ in range(200):
                            rng.shuffle(bv)
                            perm.append(_ks(pd.Series(bv[:m]), pd.Series(bv[m:2 * m])))
                    tau_clean = float(np.percentile(perm, 95)) if perm else 0.05
                    floor = 1.36 * np.sqrt((len(a_) + len(b_)) / (len(a_) * len(b_)))
                    aligned = lambda D: (D <= 0.25 * D0) and (D <= tau_clean) and (D < 0.05)
                    add_ok = aligned(D_add) and D0 > 2 * floor and (np.isnan(iqr) or iqr < 0.5 * abs(delta) + 0.05 * scale)
                    mul_ok = (not add_ok) and aligned(D_mul) and D0 > 2 * floor and abs(ratio - 1) > 0.05
                    if add_ok:
                        findings.append(dict(kind="additive_shift", slicer=s, level=lv, ref=ref, target=t, n=int(len(a_)), delta=round(delta, 4),
                                             evidence=f"KS {D0:.3f}->{D_add:.3f} (cross-fitted) after subtracting {delta:.4g}; tau_clean={tau_clean:.3f}; consistent across {len(offs)} strata (sign {sign:.0%})",
                                             confidence="REPAIR", repair={"type": "subtract", "slicer": s, "level": lv, "target": t, "value": delta}))
                    elif mul_ok:
                        findings.append(dict(kind="multiplicative_shift", slicer=s, level=lv, ref=ref, target=t, n=int(len(a_)), ratio=round(ratio, 4),
                                             evidence=f"KS {D0:.3f}->{D_mul:.3f} (cross-fitted) after dividing by {ratio:.4g}; tau_clean={tau_clean:.3f}",
                                             confidence="REPAIR", repair={"type": "divide", "slicer": s, "level": lv, "target": t, "value": ratio}))
                    elif D0 >= 0.20 and sign >= 0.90 and abs(delta) > 0.25 * scale:
                        findings.append(dict(kind="unexplained_shift", slicer=s, level=lv, ref=ref, target=t, n=int(len(a_)), delta=round(delta, 4),
                                             evidence=f"KS {D0:.3f}; median offset {delta:.4g} but no single mechanism aligns (KS after add {D_add:.3f}, mult {D_mul:.3f})",
                                             confidence="REVIEW", repair=None))
    return _dedupe(findings)


def categorical_audit(df, slicers, categoricals, target_col):
    """Two opposite signatures on a (context, slicer) x target table:
      categorical_overwrite : exactly ONE slicer level is ~100% a single value while
                              other levels are mixed  -> injected constant overwrite
      fd_violation          : several levels are ~pure (near-functional dependency
                              slicer -> target) but some levels are mixed -> broken there
    """
    findings = []
    slicers = rank_slicers(df, slicers)
    cats = [c for c in categoricals if c != target_col and 2 <= df[c].nunique() <= 12]
    def table(sub, s, t):
        ct = pd.crosstab(sub[s], sub[t])
        ct = ct[ct.sum(axis=1) >= 20]
        return ct, ct.div(ct.sum(axis=1), axis=0)
    for t in cats:
        for s in slicers:
            if s == t:
                continue
            ct, pr = table(df, s, t)
            if len(pr) < 2:
                continue
            gpure = int((pr.max(axis=1) >= 0.97).sum())
            fd_like = gpure >= 2
            if fd_like:
                pure_lv = pr.index[pr.max(axis=1) >= 0.97]
                for lv in pr.index[pr.max(axis=1) <= 0.90]:
                    findings.append(dict(kind="fd_violation", slicer=s, level=lv, context=None, context_level=None, target=t, n=int(ct.loc[lv].sum()),
                                         evidence=f"{s}->{t} is near-functional ({gpure} levels >=97% pure, e.g. {', '.join(f'{k}->{pr.loc[k].idxmax()}' for k in list(pure_lv)[:3])}) but {s}={lv} is mixed: {dict((k, round(v,2)) for k,v in pr.loc[lv].sort_values(ascending=False).head(3).items())}. Which value is right needs domain judgement.",
                                         confidence="REVIEW", repair={"type": "flag_fd", "slicer": s, "level": lv, "target": t}))
                continue
            for ctx in [None] + [c for c in slicers[:6] if c not in (s, t)]:
                ctx_levels = [None] if ctx is None else [l for l in df[ctx].dropna().unique() if (df[ctx] == l).sum() >= 2 * MIN_N]
                for cl in ctx_levels:
                    sub = df if cl is None else df[df[ctx] == cl]
                    ct, pr = table(sub, s, t)
                    if len(pr) < 2:
                        continue
                    pure = pr.index[(pr.max(axis=1) >= 0.97) & (ct.sum(axis=1) >= MIN_N)]
                    mixed = pr.index[pr.max(axis=1) <= 0.90]
                    if len(pure) == 1 and len(mixed) >= 1:
                        lv = pure[0]; pa = pr.loc[lv]
                        rest = sub.loc[sub[s] != lv, t].dropna(); pb = rest.value_counts(normalize=True)
                        # cross-context check: the collapsed value must be a MINORITY for the same
                        # slice in the other context levels (else it is stable structure seen sideways)
                        other_share = 0.0
                        if ctx is not None:
                            shares = []
                            for cl2 in df[ctx].dropna().unique():
                                if cl2 == cl: continue
                                q = df.loc[(df[ctx] == cl2) & (df[s] == lv), t].dropna()
                                if len(q) >= 30: shares.append((q == pa.idxmax()).mean())
                            other_share = float(np.median(shares)) if shares else 0.0
                        if pb.max() <= 0.90 and pb.get(pa.idxmax(), 0) <= 0.60 and other_share <= 0.60:
                            tv = 0.5 * sum(abs(pa.get(c, 0) - pb.get(c, 0)) for c in pa.index.union(pb.index))
                            findings.append(dict(kind="categorical_overwrite", slicer=s, level=lv, context=ctx, context_level=cl, target=t,
                                                 n=int(ct.loc[lv].sum()), collapsed_to=str(pa.idxmax()),
                                                 evidence=f"{pa.max():.0%} '{pa.idxmax()}' inside vs {pb.get(pa.idxmax(),0):.0%} outside (TV {tv:.2f}); other {s} levels are mixed; reference mix {dict((k, round(v,3)) for k,v in pb.head(4).items())}",
                                                 confidence="REPAIR", repair={"type": "reimpute_categorical", "slicer": s, "level": lv, "context": ctx, "context_level": cl, "target": t}))
    return _dedupe(findings)


def missingness_audit(df, slicers, cols, target_col):
    """Categorical targets only (numeric NaN handled by the evaluator's median fill)."""
    findings = []
    slicers = rank_slicers(df, slicers)
    for t in cols:
        if t == target_col or df[t].isna().mean() < 0.02:
            continue
        if pd.api.types.is_numeric_dtype(df[t]):
            continue
        for s in slicers:
            if s == t:
                continue
            rates = df.groupby(s)[t].apply(lambda x: x.isna().mean())
            counts = df.groupby(s)[t].size()
            rates = rates[counts >= MIN_N]
            if len(rates) < 2:
                continue
            ref = rates.idxmin()
            if rates[ref] > 0.03:
                continue
            for lv, r in rates.items():
                if lv == ref or r < rates[ref] + 0.15:
                    continue
                p_ref = df.loc[df[s] == ref, t].value_counts(normalize=True)
                p_obs = df.loc[df[s] == lv, t].dropna().value_counts(normalize=True)
                cats = p_ref.index.union(p_obs.index)
                imp = {c: max(0.0, (p_ref.get(c, 0) - (1 - r) * p_obs.get(c, 0)) / r) for c in cats}
                z = sum(imp.values()) or 1.0
                imp = {c: v / z for c, v in sorted(imp.items(), key=lambda kv: -kv[1]) if v / z > 0.02}
                findings.append(dict(kind="mnar_missingness", slicer=s, level=lv, ref=ref, target=t, n=int(counts[lv]),
                                     missing_rate=round(float(r), 3),
                                     evidence=f"{r:.0%} missing vs {rates[ref]:.1%} in reference {s}={ref}; MNAR imputation dist {dict((k, round(v,3)) for k,v in list(imp.items())[:4])}",
                                     confidence="REPAIR", repair={"type": "impute_missing", "slicer": s, "level": lv, "target": t, "dist": imp}))
    return _dedupe(findings)


def cross_summaries(df, slicers, numerics, categoricals, target_col, max_tables=6):
    """Cross-column views the LLM never computes by itself: for the most
    informative slicers, medians of numerics and mixes of categoricals per level.
    Acquisition-like slicers (year, month, channel...) come first - corruptions
    are typically conditioned on how/when the data was collected."""
    out = []
    ranked = [s for s in rank_slicers(df, [s for s in slicers if s != target_col], cap=max_tables)
              if 2 <= df[s].nunique(dropna=True) <= 25]
    for s in ranked:
        g = df.groupby(s)
        rows = []
        for lv, sub in g:
            if len(sub) < 10:
                continue
            parts = [f"n={len(sub)}"]
            for t in numerics[:6]:
                if t == s: continue
                v = pd.to_numeric(sub[t], errors="coerce")
                parts.append(f"{t}:med={v.median():.3g},min={v.min():.3g},max={v.max():.3g}")
            for t in categoricals[:3]:
                if t == s: continue
                vc = sub[t].value_counts(normalize=True).head(3)
                parts.append(f"{t}=" + "/".join(f"{k}:{p:.0%}" for k, p in vc.items()))
            rows.append(f"  {s}={lv}: " + " | ".join(parts))
        if rows:
            out.append(f"By {s}:\n" + "\n".join(rows[:30]))
    return "\n".join(out)


def _dedupe(findings):
    seen, out = set(), []
    for f in findings:
        key = (f["kind"], f["slicer"], str(f["level"]), f["target"], str(f.get("context")), str(f.get("context_level")))
        # (ref intentionally excluded: one finding per slice/target)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out
