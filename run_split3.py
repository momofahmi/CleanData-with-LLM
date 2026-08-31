#!/usr/bin/env python3
"""1. Monkeypatches evaluation.get_split so the whole pipeline - agent prompt,
     submission tool, do-no-harm gate, escalation ladder - sees the VALIDATION
     split (20%) wherever it used to see the test split. The pipeline itself is
     not edited: agent behaviour, budgets and prompts are byte-identical.
  2. Runs run_pipeline.main() with whatever arguments you pass.
  3. AFTER the run, from outside the agentic loop, computes the final held-out
     TEST score (20%) exactly once, on the best submitted file, and appends a
     "split3" block to the run's summary.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import evaluation
from evaluation import evaluate


def split3_parts(df, cfg, seed):
    idx = np.arange(len(df))
    strat = df[cfg.target] if df[cfg.target].nunique() > 1 else None
    rest, te = train_test_split(idx, test_size=0.20, random_state=seed, stratify=strat)
    s2 = df[cfg.target].iloc[rest] if strat is not None else None
    tr, va = train_test_split(rest, test_size=0.25, random_state=seed, stratify=s2)
    return tr, va, te                      # 60 / 20 / 20


def patched_get_split(df, cfg, seed=42, test_size=0.3):
    tr, va, _te = split3_parts(df, cfg, seed)
    return tr, va                          # the pipeline's 'test' is now the validation


def main():
    evaluation.get_split = patched_get_split
    import run_pipeline                    # imports AFTER the patch on the module attr
    run_pipeline.get_split = patched_get_split   # and override the imported symbol too

    before = set(glob.glob("runs/*"))
    run_pipeline.main()
    new = sorted(set(glob.glob("runs/*")) - before)
    if not new:
        print("[split3] no new run directory found - nothing to score"); return
    run_dir = new[-1]

    # -------- final test, computed ONCE, outside the agentic loop --------
    args = sys.argv[1:]
    ds = args[args.index("--dataset") + 1]
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 42
    import bench_datasets
    cfg = bench_datasets.get(ds)
    clean = pd.read_csv(f"data/{ds}_clean.csv")
    dirty = pd.read_csv(f"data/{ds}_dirty.csv")
    tr, va, te = split3_parts(clean, cfg, seed)
    clean_test = clean.loc[te].reset_index(drop=True)
    clean_train = clean.loc[tr].reset_index(drop=True)
    dirty_train = dirty.loc[tr].reset_index(drop=True)

    summ = json.load(open(f"{run_dir}/summary.json"))
    hist = summ.get("history") or []
    best = os.path.basename(max(hist, key=lambda h: h[1])[0]) if hist else None
    cleaned = pd.read_csv(f"{run_dir}/{best}") if best and os.path.exists(f"{run_dir}/{best}") \
        else dirty_train

    block = {
        "protocol": "60/20/20; gate and submissions saw VALIDATION only; this test was computed once",
        "p_dirty_test": evaluate(dirty_train, clean_test, cfg, seed=seed),
        "p_clean_test": evaluate(clean_train, clean_test, cfg, seed=seed),
        "f1_test_once": evaluate(cleaned, clean_test, cfg, seed=seed),
        "best_file": best,
    }
    summ["split3"] = block
    json.dump(summ, open(f"{run_dir}/summary.json", "w"), indent=1)
    print(f"[split3] P_dirty(test)={block['p_dirty_test']:.4f}  "
          f"P_clean(test)={block['p_clean_test']:.4f}  F1_TEST_once={block['f1_test_once']:.4f}  "
          f"({best})")


if __name__ == "__main__":
    main()
