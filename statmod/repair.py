"""
The deterministic repair executor. Applies the fitted mechanisms, 
and renders each repair as a pandas snippet the agent can review, 
adapt or run in hybrid mode.
"""
import numpy as np
import pandas as pd


def _mask(df, r):
    m = df[r["slicer"]] == r["level"]
    if r.get("context") is not None:
        m &= df[r["context"]] == r["context_level"]
    return m


def apply_repair(df, r, seed=0):
    rng = np.random.default_rng(seed)
    t = r.get("target")
    typ = r["type"]
    if typ in ("subtract", "divide", "impute_from_reference", "interpolate_year", "divide_window") and t in df.columns:
        df[t] = pd.to_numeric(df[t], errors="coerce").astype(float)     # avoid int-dtype assignment errors
    if typ == "subtract":
        m = _mask(df, r); df.loc[m, t] = df.loc[m, t] - r["value"]
    elif typ == "divide":
        m = _mask(df, r); df.loc[m, t] = df.loc[m, t] / r["value"]
    elif typ == "impute_from_reference":
        m = _mask(df, r)
        ref = pd.to_numeric(df.loc[~m, t], errors="coerce").dropna()
        df.loc[m, t] = rng.choice(ref.to_numpy(), size=int(m.sum()), replace=True) if len(ref) else np.nan
    elif typ == "reimpute_categorical":
        m = _mask(df, r)
        ctx = df[df[r["context"]] == r["context_level"]] if r.get("context") is not None else df
        ref = ctx.loc[ctx[r["slicer"]] != r["level"], t].dropna()
        if len(ref):
            p = ref.value_counts(normalize=True)
            df.loc[m, t] = rng.choice(p.index.to_numpy(), size=int(m.sum()), p=p.to_numpy())
    elif typ == "impute_missing":
        m = _mask(df, r) & df[t].isna()
        dist = r["dist"]
        if dist and m.any():
            cats, p = list(dist.keys()), np.array(list(dist.values()), dtype=float); p /= p.sum()
            if p.max() >= 0.75:      # one category dominates the missing mass -> deterministic fill
                df.loc[m, t] = cats[int(p.argmax())]
            else:
                df.loc[m, t] = rng.choice(cats, size=int(m.sum()), p=p)
    elif typ == "interpolate_year":
        # replace the collapsed year by linear interpolation between the nearest intact years on each side; the split may have removed the adjacent year
        e, tc, y = r["entity"], r["time"], r["year"]
        bad_years = set(r.get("all_years", [y]))
        for ent, g in df.groupby(e):
            g = g.sort_values(tc)
            s = pd.to_numeric(g.set_index(tc)[t], errors="coerce")
            if y not in s.index:
                continue
            good = s[[yy for yy in s.index if yy not in bad_years]].dropna()
            prev = good[good.index < y]; nxt = good[good.index > y]
            if len(prev) and len(nxt):
                y0, y1 = prev.index.max(), nxt.index.min()
                if y1 - y0 <= 8:
                    val = s[y0] + (s[y1] - s[y0]) * (y - y0) / (y1 - y0)
                    df.loc[g.index[g[tc] == y], t] = val
            elif len(prev) and (y - prev.index.max()) <= 3:
                df.loc[g.index[g[tc] == y], t] = prev.iloc[-1]
            elif len(nxt) and (nxt.index.min() - y) <= 3:
                df.loc[g.index[g[tc] == y], t] = nxt.iloc[0]
    elif typ == "divide_window":
        m = (df[r["entity_col"]] == r["entity"]) & df[r["time"]].between(r["start"], r["end"])
        if r.get("per_year"):
            k = df.loc[m, r["time"]] - r["start"] + 1
            df.loc[m, t] = pd.to_numeric(df.loc[m, t], errors="coerce") / (r["per_year"] ** k)
        else:
            df.loc[m, t] = pd.to_numeric(df.loc[m, t], errors="coerce") / r["factor"]
    elif typ == "divide_window_multi":
        m = (df[r["entity_col"]] == r["entity"]) & df[r["time"]].between(r["start"], r["end"])
        for c in r["targets"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
            df.loc[m, c] = df.loc[m, c] / r["factor"]
    elif typ == "flag_entity":
        pass   # review-only
    return df


def repair_snippet(r):
    """pandas code the LLM can run as-is (on a fresh df)."""
    t = r.get("target"); typ = r["type"]
    if typ in ("subtract", "divide"):
        op = "-" if typ == "subtract" else "/"
        m = f"(df[{r['slicer']!r}] == {r['level']!r})" + (f" & (df[{r['context']!r}] == {r['context_level']!r})" if r.get("context") is not None else "")
        return f"m = {m}\ndf.loc[m, {t!r}] = df.loc[m, {t!r}] {op} {r['value']}"
    if typ == "reimpute_categorical":
        ctx = f"df[df[{r['context']!r}] == {r['context_level']!r}]" if r.get("context") is not None else "df"
        return (f"ctx = {ctx}\nref = ctx.loc[ctx[{r['slicer']!r}] != {r['level']!r}, {t!r}].dropna()\n"
                f"p = ref.value_counts(normalize=True)\nm = (df[{r['slicer']!r}] == {r['level']!r})"
                + (f" & (df[{r['context']!r}] == {r['context_level']!r})" if r.get("context") is not None else "")
                + f"\ndf.loc[m, {t!r}] = np.random.default_rng(0).choice(p.index, size=m.sum(), p=p.values)")
    if typ == "impute_missing":
        top = max(r["dist"], key=r["dist"].get) if r["dist"] else None
        return (f"m = (df[{r['slicer']!r}] == {r['level']!r}) & df[{t!r}].isna()\n"
                f"df.loc[m, {t!r}] = {top!r}   # dominant MNAR category; or sample from {dict(list(r['dist'].items())[:3])}")
    if typ == "impute_from_reference":
        return (f"m = (df[{r['slicer']!r}] == {r['level']!r})\nref = df.loc[~m, {t!r}].dropna()\n"
                f"df.loc[m, {t!r}] = np.random.default_rng(0).choice(ref.values, size=m.sum())")
    if typ == "interpolate_year":
        return (f"for e, g in df.groupby({r['entity']!r}):\n    s = g.sort_values({r['time']!r}).set_index({r['time']!r})[{t!r}]\n"
                f"    if {r['year']} in s.index and {r['year']-1} in s.index and {r['year']+1} in s.index:\n"
                f"        df.loc[g.index[g[{r['time']!r}] == {r['year']}], {t!r}] = (s[{r['year']-1}] + s[{r['year']+1}]) / 2")
    if typ == "divide_window":
        if r.get("per_year"):
            return (f"m = (df[{r['entity_col']!r}] == {r['entity']!r}) & df[{r['time']!r}].between({r['start']}, {r['end']})\n"
                    f"k = df.loc[m, {r['time']!r}] - {r['start']} + 1\ndf.loc[m, {t!r}] = df.loc[m, {t!r}] / ({r['per_year']:.4f} ** k)")
        return (f"m = (df[{r['entity_col']!r}] == {r['entity']!r}) & df[{r['time']!r}].between({r['start']}, {r['end']})\n"
                f"df.loc[m, {t!r}] = df.loc[m, {t!r}] / {r['factor']:.4f}")
    if typ == "divide_window_multi":
        return (f"m = (df[{r['entity_col']!r}] == {r['entity']!r}) & df[{r['time']!r}].between({r['start']}, {r['end']})\n"
                f"for c in {r['targets']}:\n    df.loc[m, c] = df.loc[m, c] / {r['factor']:.4f}")
    if typ == "flag_entity":
        return f"# review: inspect df[df[{r['entity_col']!r}] == {r['entity']!r}][{t!r}] against world knowledge"
    if typ == "flag_fd":
        return f"# review: print(pd.crosstab(df[{r['slicer']!r}], df[{t!r}])) and decide the correct {t} for {r['slicer']}={r['level']!r}"
    return "# no snippet"


def snippet_with_derived(r):
    """Prefix the derivation of helper columns (e.g. Name__title) so the snippet runs on a fresh df."""
    code = repair_snippet(r)
    for key in ("slicer", "context"):
        col = r.get(key)
        if isinstance(col, str) and col.endswith("__title"):
            src = col[:-7]
            code = (f"df[{col!r}] = df[{src!r}].astype(str).str.extract(r{DERIVED_RE!r})[0].str.strip()\n" + code
                    + f"\ndf = df.drop(columns=[{col!r}])   # helper column must not be saved")
    return code


DERIVED_RE = r",\s*([A-Za-z ]+?)\."


def add_derived(df):
    """Recreate helper columns used by findings (e.g. Name__title) - dropped again before saving."""
    added = []
    for c in list(df.columns):
        if (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object) and f"{c}__title" not in df.columns:
            tok = df[c].astype(str).str.extract(DERIVED_RE)[0].str.strip()
            if tok.notna().mean() > 0.5 and 2 <= tok.nunique() <= 60:
                df[f"{c}__title"] = tok; added.append(f"{c}__title")
    return added


def apply_all(df, findings, only_confidence=("REPAIR",), seed=0):
    df = df.copy()
    added = add_derived(df)
    applied = []
    for f in findings:
        if f.get("repair") and f["confidence"] in only_confidence:
            df = apply_repair(df, f["repair"], seed=seed)
            applied.append(f)
    df = df.drop(columns=[c for c in added if c in df.columns])
    return df, applied
