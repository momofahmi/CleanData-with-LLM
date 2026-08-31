#!/usr/bin/env python3
"""
Statistics-only runs under the 60/20/20 protocol.
60% train (corrupted) - what the cleaner sees and repairs
20% validation (clean) - the only split the gate may query
20% test (clean) - read once, after all repairs are frozen
Splits are stratified and derived from the seed. The audit, 
the repairs and the gate come from statmod, unmodified.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_split3(df, cfg, seed):
    idx = np.arange(len(df))
    strat = df[cfg.target] if df[cfg.target].nunique() > 1 else None
    rest, te = train_test_split(idx, test_size=0.20, random_state=seed, stratify=strat)
    strat2 = df[cfg.target].iloc[rest] if strat is not None else None
    tr, va = train_test_split(rest, test_size=0.25, random_state=seed, stratify=strat2)
    return tr, va, te   # 60 / 20 / 20 (protocol)


def neq(a, b):
    an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    both = an.notna() & bn.notna()
    return (both & ((an - bn).abs() > 1e-9 * (1 + an.abs()))) \
        | (~both & ~(a.isna() & b.isna()) & (a.astype(str) != b.astype(str))) \
        | (a.isna() ^ b.isna())


def cell_counts(clean, dirty, cleaned):
    tot = dict(injected=0, repaired=0, changed=0, missed=0, collateral=0)
    for c in dirty.columns:
        if c not in clean.columns or c not in cleaned.columns:
            continue
        inj = neq(clean[c], dirty[c])
        rep = inj & ~neq(cleaned[c], clean[c])
        mis = inj & ~neq(cleaned[c], dirty[c])
        tot["injected"] += int(inj.sum()); tot["repaired"] += int(rep.sum())
        tot["missed"] += int(mis.sum()); tot["changed"] += int((inj & ~rep & ~mis).sum())
        tot["collateral"] += int((~inj & neq(cleaned[c], clean[c])).sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["titanic", "hotel", "meat"])
    ap.add_argument("--data", required=True, help="clean dataset CSV")
    ap.add_argument("--root", default=".", help="folder with the hybrid_agent package")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    sys.path.insert(0, args.root)
    import bench_datasets
    import statmod
    from evaluation import evaluate
    from statmod.repair import apply_repair, add_derived

    cfg = bench_datasets.get(args.dataset)
    clean = pd.read_csv(args.data)

    print(f"{'seed':>4s} {'Pdirty(test)':>12s} {'Pclean(test)':>12s} {'kept':>4s} "
          f"{'F1_val_final':>12s} {'F1_TEST_once':>12s} {'exact':>6s} {'chg':>4s} {'miss':>5s} {'collat':>6s}")
    for seed in args.seeds:
        tr, va, te = get_split3(clean, cfg, seed)
        clean_tr = clean.loc[tr].reset_index(drop=True)
        val_df = clean.loc[va].reset_index(drop=True)
        test_df = clean.loc[te].reset_index(drop=True)

        out = cfg.inject(clean.copy(), seed=seed) if hasattr(cfg, "inject") else None
        if out is None:
            sys.exit("this dataset config has no inject(); adapt the runner to your injector name")
        dirty_full = out[0] if isinstance(out, tuple) else out   # inject() returns (df, ground_truth)
        dirty_tr = dirty_full.loc[tr].reset_index(drop=True)

        p_dirty_test = evaluate(dirty_tr, test_df, cfg)
        p_clean_test = evaluate(clean_tr, test_df, cfg)

        # audit, then apply repairs one by one; the gate only queries the validation split
        findings, _ = statmod.audit(dirty_tr.copy(), cfg, verbose=False)
        current = dirty_tr.copy()
        add_derived(current)
        base_val = evaluate(current, val_df, cfg)
        kept, f1_val = 0, base_val
        for f in findings:
            if f.get("confidence") != "REPAIR" or not f.get("repair"):
                continue
            cand = current.copy()
            try:
                apply_repair(cand, f["repair"], seed=seed)
            except Exception:
                continue
            v = evaluate(cand, val_df, cfg)
            if v >= f1_val:          # strict gate, validation only
                current, f1_val, kept = cand, v, kept + 1

        # ---- repairs frozen; the TEST split is consulted exactly once ----
        drop = [c for c in current.columns if c not in dirty_tr.columns]
        final = current.drop(columns=drop)
        f1_test_once = evaluate(final, test_df, cfg)
        cc = cell_counts(clean_tr, dirty_tr, final)
        print(f"{seed:>4d} {p_dirty_test:>12.4f} {p_clean_test:>12.4f} {kept:>4d} "
              f"{f1_val:>12.4f} {f1_test_once:>12.4f} {cc['repaired']:>6d} {cc['changed']:>4d} "
              f"{cc['missed']:>5d} {cc['collateral']:>6d}")


if __name__ == "__main__":
    main()

