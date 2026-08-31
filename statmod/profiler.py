"""
Types every column, extracts simple text features (like honorific tokens), 
and decides which columns can act as slicers: the years, channels or categories
a corruption could be conditioned on.
"""
import re
import numpy as np
import pandas as pd

MAX_SLICER_LEVELS = 40


def infer_role(s: pd.Series, name, cfg):
    if name == cfg.target:
        return "target"
    if name in cfg.id_cols:
        return "id"
    n, nun = len(s), s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        vals = s.dropna()
        if nun <= 2:
            return "binary"
        # year-like or small-int categorical -> slicer-able numeric
        if nun <= MAX_SLICER_LEVELS and (vals % 1 == 0).all():
            return "numeric_categorical"
        return "numeric"
    if nun == n or nun > 0.8 * n:
        return "id"
    if nun <= MAX_SLICER_LEVELS:
        return "categorical"
    return "text"


def derive_features(df, roles):
    """Cheap derivations the LLM would otherwise have to think of.
    Currently: honorific/title token from text columns (e.g. 'Braund, Mr. Owen' -> 'Mr')."""
    derived = {}
    for c, r in roles.items():
        if r in ("text", "id") and (pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object):
            tok = df[c].astype(str).str.extract(r",\s*([A-Za-z ]+?)\.")[0].str.strip()
            if tok.notna().mean() > 0.5 and 2 <= tok.nunique() <= 60:
                derived[f"{c}__title"] = tok
    return derived


def profile(df, cfg):
    roles = {c: infer_role(df[c], c, cfg) for c in df.columns}
    derived = derive_features(df, roles)
    for k, v in derived.items():
        df[k] = v
        roles[k] = "categorical"
    lines = [f"Rows: {len(df)}  Columns: {len(df.columns)}"]
    for c in df.columns:
        s, r = df[c], roles[c]
        miss = s.isna().mean()
        if r in ("numeric", "numeric_categorical", "binary"):
            v = pd.to_numeric(s, errors="coerce")
            q = v.quantile([0, .25, .5, .75, 1]).values
            lines.append(f"- {c} [{r}] missing={miss:.1%} min={q[0]:.4g} q25={q[1]:.4g} med={q[2]:.4g} q75={q[3]:.4g} max={q[4]:.4g} nunique={s.nunique()}")
        elif r == "categorical":
            vc = s.value_counts(normalize=True).head(6)
            top = ", ".join(f"{k}:{p:.0%}" for k, p in vc.items())
            lines.append(f"- {c} [{r}] missing={miss:.1%} nunique={s.nunique()} top={{ {top} }}")
        else:
            lines.append(f"- {c} [{r}] missing={miss:.1%} nunique={s.nunique()}")
    slicers = [c for c, r in roles.items() if r in ("categorical", "numeric_categorical", "binary")]
    numerics = [c for c, r in roles.items() if r in ("numeric", "numeric_categorical")]
    categoricals = [c for c, r in roles.items() if r in ("categorical", "binary")]
    return {"roles": roles, "derived": list(derived), "slicers": slicers,
            "numerics": numerics, "categoricals": categoricals, "text": "\n".join(lines)}
