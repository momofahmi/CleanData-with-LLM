"""
The fixed evaluation from the paper. Train on the (possibly cleaned) train split,
report F1 on the clean held-out test split. Preprocessing only uses the declared
feature columns, so added or dropped columns are ignored: agents can only change values.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def get_split(df, cfg, seed=42, test_size=0.3):
    idx = np.arange(len(df))
    strat = df[cfg.target] if df[cfg.target].nunique() > 1 else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    return tr, te


def _preprocess(df, cfg, maps=None):
    df = df.copy()
    parts = []
    num = pd.DataFrame(index=df.index)
    for c in cfg.numeric_features:
        num[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    num = num.fillna(num.median(numeric_only=True)).fillna(0.0)
    parts.append(num)
    learn = maps is None
    if learn:
        maps = {}
    for c in cfg.categorical_features:
        col = df[c].astype(str) if c in df.columns else pd.Series("OTHER", index=df.index)
        col = col.replace({"nan": "OTHER", "None": "OTHER"})
        if learn:
            k = cfg.top_k.get(c, 10)
            maps[c] = [v for v in col.value_counts().head(k).index if v != "OTHER"]
        levels = maps[c] + ["OTHER"]
        col = col.where(col.isin(maps[c]), "OTHER")
        oh = pd.DataFrame({f"{c}={lv}": (col == lv).astype(int) for lv in levels}, index=df.index)
        parts.append(oh)
    X = pd.concat(parts, axis=1)
    return X.reindex(sorted(X.columns), axis=1), maps


def evaluate(train_df, test_df, cfg, seed=42):
    Xtr, maps = _preprocess(train_df, cfg)
    Xte, _ = _preprocess(test_df, cfg, maps)
    ytr = pd.to_numeric(train_df[cfg.target], errors="coerce").fillna(0).astype(int).values
    yte = pd.to_numeric(test_df[cfg.target], errors="coerce").fillna(0).astype(int).values
    sc = StandardScaler()
    m = LogisticRegression(max_iter=2000, random_state=seed)
    m.fit(sc.fit_transform(Xtr), ytr)
    return float(f1_score(yte, m.predict(sc.transform(Xte))))
