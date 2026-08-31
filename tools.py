"""The two agent tools from Bendinelli et al.: a persistent Python shell and a
performance-evaluation tool (submit a cleaned train csv -> F1 on hidden clean test)."""
import io
import os
import contextlib
import traceback
import pandas as pd
import numpy as np


class PersistentIPython:
    """State (variables, dataframes) persists across calls, like a notebook."""
    def __init__(self):
        import re, json, math
        self.ns = {"pd": pd, "np": np, "re": re, "json": json, "math": math, "os": os,
                   "display": print}
    def run(self, code, max_chars=3000):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(code, self.ns)
            out = buf.getvalue()
        except Exception:
            out = buf.getvalue() + "\n[ERROR] " + traceback.format_exc(limit=3)
        if not out.strip():
            out = "[Code executed successfully with no output]"
        return out[:max_chars] + ("\n...[truncated]" if len(out) > max_chars else "")


class PerformanceEvaluator:
    def __init__(self, clean_test_df, cfg, evaluate_fn, sandbox="sandbox"):
        self.clean_test = clean_test_df
        self.cfg = cfg
        self.evaluate = evaluate_fn
        self.sandbox = sandbox
        self.best_score = None
        self.best_path = None
        self.history = []
    def submit(self, path):
        if not path or not os.path.exists(path):
            files = sorted(os.listdir(self.sandbox)) if os.path.isdir(self.sandbox) else []
            return (f"[ERROR] File not found: {path!r}. Files in {self.sandbox}/: {files}. "
                    "Re-run your cleaning code, confirm with print() that to_csv succeeded, then submit that exact path.")
        try:
            df = pd.read_csv(path)
            if self.cfg.target not in df.columns:
                return f"[ERROR] Submitted file lacks the target column {self.cfg.target!r}."
            score = self.evaluate(df, self.clean_test, self.cfg)
        except Exception as e:
            return f"[ERROR] Evaluation failed: {e}"
        self.history.append((path, score))
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_path = score, path
        return f"Model F1 score on the held-out test set: {score:.4f}. (Best so far: {self.best_score:.4f})"
