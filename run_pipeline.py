"""
Runs one experiment: load the raw csv, inject the benchmark errors, split, 
compute the two anchor scores, run the audit and/or the agent depending on 
the mode, score the submission.
"""
import os
import sys
import json
import shutil
import argparse
from datetime import datetime
import pandas as pd

import bench_datasets as datasets
from evaluation import get_split, evaluate
from tools import PersistentIPython, PerformanceEvaluator
import statmod
from statmod.repair import apply_all


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
    def __call__(self, *a):
        s = " ".join(str(x) for x in a); print(s); self.f.write(s + "\n"); self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(datasets.REGISTRY))
    ap.add_argument("--mode", default="hybrid", choices=["llm", "stats", "hybrid"])
    ap.add_argument("--hint", default="none", choices=["none", "weak", "strong"])
    ap.add_argument("--model", default="deepseek-reasoner")
    ap.add_argument("--api-base", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--goal-margin", type=float, default=0.03)
    ap.add_argument("--token-budget", type=int, default=200_000)
    ap.add_argument("--reinject", action="store_true", help="regenerate the dirty dataset even if it exists")
    ap.add_argument("--no-anti-memorization", action="store_true")
    ap.add_argument("--stats-review-too", action="store_true", help="in stats mode also apply REVIEW findings")
    ap.add_argument("--audit-rounds", type=int, default=2, help="stats mode: re-audit the repaired data and repeat (catches errors masked by others)")
    ap.add_argument("--gate-tol", type=float, default=0.0, help="stats mode: keep a repair unless the score drops by more than this (do-no-harm band); 0 = strict improvement only")
    a = ap.parse_args()

    cfg = datasets.get(a.dataset)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = f"runs/{stamp}_{a.dataset}_{a.mode}_{a.hint}_{a.model.replace('/', '-')}"
    os.makedirs(run_dir, exist_ok=True); os.makedirs("sandbox", exist_ok=True); os.makedirs("data", exist_ok=True)
    log = Tee(f"{run_dir}/run.log")
    log(f"===== HYBRID CLEANING PIPELINE | dataset={a.dataset} mode={a.mode} hint={a.hint} model={a.model} =====")

    # ---- 1. load + inject ----
    if not os.path.exists(cfg.raw_path):
        log(f"[ERROR] raw csv not found at {cfg.raw_path}. Download it into data/ first."); sys.exit(1)
    clean_path, dirty_path, gt_path = f"data/{a.dataset}_clean.csv", f"data/{a.dataset}_dirty.csv", f"data/{a.dataset}_ground_truth.json"
    if a.reinject or not os.path.exists(dirty_path):
        clean = cfg.load(cfg.raw_path)
        dirty, gt = cfg.inject(clean, seed=a.seed)
        clean.to_csv(clean_path, index=False); dirty.to_csv(dirty_path, index=False)
        json.dump(gt, open(gt_path, "w"))
        log(f"[inject] {a.dataset}: " + ", ".join(f"{k}={len(v)} rows" for k, v in gt.items()))
    clean = pd.read_csv(clean_path); dirty = pd.read_csv(dirty_path); gt = json.load(open(gt_path))

    # ---- 2. split + baselines ----
    tr, te = get_split(clean, cfg, seed=a.seed)
    clean_test = clean.loc[te].reset_index(drop=True)
    clean_train = clean.loc[tr].reset_index(drop=True)
    dirty_train = dirty.loc[tr].reset_index(drop=True)
    for f in os.listdir("sandbox"):
        os.remove(os.path.join("sandbox", f))
    dirty_train.to_csv("sandbox/train.csv", index=False)
    p_dirty = evaluate(dirty_train, clean_test, cfg, seed=a.seed)
    p_clean = evaluate(clean_train, clean_test, cfg, seed=a.seed)
    goal = p_dirty + a.goal_margin
    log(f"P_Clean = {p_clean:.4f} | P_Dirty = {p_dirty:.4f} | Goal = {goal:.4f}")

    evaluator = PerformanceEvaluator(clean_test, cfg, evaluate, sandbox="sandbox")

    # ---- 3. statistical audit ----
    report, findings = None, []
    if a.mode in ("stats", "hybrid"):
        findings, report = statmod.audit(dirty_train, cfg)
        open(f"{run_dir}/stat_report.txt", "w").write(report)
        json.dump(findings, open(f"{run_dir}/findings.json", "w"), indent=1, default=str)
        log(f"[statmod] report saved ({len(report)} chars). Top findings:")
        for f in findings[:12]:
            where = f"{f['slicer']}={f['level']}" + (f" & {f['context']}={f['context_level']}" if f.get('context') is not None else "")
            log(f"   [{f['confidence']}] {f['kind']:22s} {f['target']:18s} where {where}")

    # ---- 4. repair ----
    tokens = 0
    if a.mode == "stats":
        conf = ("REPAIR", "REVIEW") if a.stats_review_too else ("REPAIR",)
        feats = set(cfg.numeric_features + cfg.categorical_features)
        def mirror_key(f):
            if f["kind"] in ("additive_shift", "multiplicative_shift") and f.get("ref") is not None:
                return (f["kind"], f["slicer"], f["target"], frozenset([str(f["level"]), str(f["ref"])]))
            return None
        current = dirty_train.copy(); best = p_dirty; kept_all = []; v = 0
        current.to_csv("sandbox/train_cleaned_v0.csv", index=False); evaluator.submit("sandbox/train_cleaned_v0.csv")
        round_findings = findings
        for rnd in range(1, a.audit_rounds + 1):
            if rnd > 1:
                round_findings, _ = statmod.audit(current, cfg, verbose=False)
                log(f"[stats-only] audit round {rnd}: {len(round_findings)} findings")
            cands = [f for f in round_findings if f.get("repair") and f["confidence"] in conf and f["repair"]["type"] not in ("flag_entity", "flag_fd")]
            cands.sort(key=lambda f: (0 if f["target"].split(",")[0] in feats else 1))
            done_mirrors, kept = set(), []
            for f in cands:
                mk = mirror_key(f)
                if mk in done_mirrors:
                    log(f"[stats-only] skip mirror of an accepted repair: {f['kind']} on {f['target']} where {f['slicer']}={f['level']}"); continue
                trial, _ = apply_all(current, [f], only_confidence=conf, seed=a.seed)
                v += 1; path = f"sandbox/train_cleaned_v{v}.csv"; trial.to_csv(path, index=False)
                evaluator.submit(path); score = evaluator.history[-1][1] if evaluator.history else None
                ok = score is not None and (score > best + 1e-6 if a.gate_tol <= 0 else score >= best - a.gate_tol)
                log(f"[stats-only] {f['kind']} on {f['target']} where {f['slicer']}={f['level']} -> {score:.4f} {'KEPT' if ok else 'REJECTED'}")
                if ok:
                    current, best = trial, max(best, score); kept.append(f)
                    if mk: done_mirrors.add(mk)
            kept_all += kept
            log(f"[stats-only] round {rnd}: kept {len(kept)}/{len(cands)} repairs")
            if not kept:
                break
        v += 1; current.to_csv(f"sandbox/train_cleaned_v{v}.csv", index=False); evaluator.submit(f"sandbox/train_cleaned_v{v}.csv")
        log(f"[stats-only] total kept {len(kept_all)} repairs over {rnd} round(s)")
    else:
        from agent import LLMClient, build_P0, run_agent
        llm = LLMClient(a.api_base, a.model, a.api_key_env)
        P0 = build_P0(p_dirty, goal, cfg.target, cfg.description, "sandbox",
                      stat_report=report if a.mode == "hybrid" else None,
                      anti_memorization=not a.no_anti_memorization) + cfg.hints[a.hint]
        open(f"{run_dir}/P0.txt", "w").write(P0)
        shell = PersistentIPython()
        evaluator, tokens = run_agent(llm, shell, evaluator, P0, "sandbox", log=log, token_budget=a.token_budget)

    # ---- 5. results + analysis ----
    log("=" * 55)
    if evaluator.best_score is not None:
        rec, avail = evaluator.best_score - p_dirty, p_clean - p_dirty
        pct = rec / avail * 100 if avail > 0 else float("nan")
        log(f"P_Dirty: {p_dirty:.4f} | Best: {evaluator.best_score:.4f} | P_Clean: {p_clean:.4f}")
        log(f"Recovery: {rec*100:+.2f}%  ({pct:.0f}% of the {avail*100:.2f}% available) | tokens used: {tokens}")
        best = pd.read_csv(evaluator.best_path)
        o2p = {int(o): p for p, o in enumerate(tr)}
        log("--- WHICH INJECTED ERRORS DID THE AGENT FIX? ---")
        for line in cfg.analyze(best, clean_train, dirty_train, gt, o2p):
            log("  " + line)
        summary = dict(dataset=a.dataset, mode=a.mode, hint=a.hint, model=a.model, p_dirty=p_dirty, p_clean=p_clean,
                       best=evaluator.best_score, recovery_pct=pct, tokens=tokens, history=evaluator.history)
    else:
        log("The agent never submitted a valid dataset.")
        summary = dict(dataset=a.dataset, mode=a.mode, hint=a.hint, model=a.model, p_dirty=p_dirty, p_clean=p_clean, best=None)
    json.dump(summary, open(f"{run_dir}/summary.json", "w"), indent=1, default=str)
    for f in os.listdir("sandbox"):
        shutil.copy2(os.path.join("sandbox", f), run_dir)
    log(f"[archived] everything saved under {run_dir}/")


if __name__ == "__main__":
    main()
