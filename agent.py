"""
The cleaning agent. Talks to an OpenAI-compatible endpoint and reads the key from an environment variable. 
In hybrid mode the prompt also carries the statistical report.
"""
import os
import glob
import json
import time
import pandas as pd

TOKEN_BUDGET = 200_000
CONTEXT_CHAR_LIMIT = 60_000
REMIND_EVERY = 6
FINALIZE_AT = 10
FORCE_BUDGET_FRACTION = 0.80

TOOLS = [
    {"type": "function", "function": {
        "name": "execute_code_ipython_shell",
        "description": "Execute Python code in a persistent IPython-like shell (state is preserved between calls). pandas as pd and numpy as np are preloaded.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "submit_clean_data",
        "description": "Submit a cleaned training csv; trains the fixed model and returns the F1 score on the hidden clean test set.",
        "parameters": {"type": "object", "properties": {"dataset_path": {"type": "string", "description": "path to the cleaned csv, e.g. sandbox/train_cleaned_v1.csv"}}, "required": ["dataset_path"]}}},
]


def build_P0(p_dirty, goal, target, description, sandbox, stat_report=None, anti_memorization=True):
    p = f"""You have been provided with a dataset containing structural errors that degrade your model's performance. Your goal is to identify and correct these errors so that the model's F1 score increases from {p_dirty:.4f} to at least {goal:.4f}. The target column is {target}.

1. Locate and fix any inconsistent or illogical values in the data. Examples of inconsistencies might include negative values where they cannot logically occur, incorrect category assignments, or strange drift in the data.
2. Your training dataset is located in {sandbox}/train.csv.
3. Each time you attempt a fix, save your cleaned dataset as {sandbox}/train_cleaned_v*.csv with an updated version number (v1, v2, ...).
4. After saving, call submit_clean_data() to evaluate the model on the test set.
5. You may run code with execute_code_ipython_shell(), but you cannot plot, add new columns, or add new rows to the saved dataset.
6. Coding rules: start each code block by reloading a fresh copy df = pd.read_csv("{sandbox}/train.csv") and apply ALL your fixes in that same block, then save; use df.loc[mask, col] = value (no chained inplace); print("saved", path) after to_csv and submit that exact path. Do NOT do ML preprocessing (no encoding, scaling, feature engineering) - the evaluator handles that; only correct wrong VALUES in existing cells.
7. Prefer statistically grounded corrections (medians, group distributions, inverse of an identified systematic operation) over arbitrary factors.
"""
    if anti_memorization:
        p += ("8. Do NOT rely on any prior knowledge of the true values of this dataset (it may be public). "
              "Only correct what you can infer from the data present here; for values that cannot be inferred, "
              "use statistical imputation (group medians / distributions), never recalled originals.\n")
    p += f"\nTip: understand the dataset before submitting. The dataset description is as follows:\n{description}\n"
    if stat_report:
        p += ("\nA statistical audit of the training data has been run for you (below). It is EVIDENCE, not orders: "
              "verify each finding with code, decide which are genuine systematic errors, apply those repairs "
              "(you may run the suggested code as-is), ignore findings you judge to be natural variation, and "
              "still look for anything the audit could not see (row-level inconsistencies, semantic implausibilities).\n\n"
              + stat_report + "\n")
    return p


def build_context(messages, char_limit=CONTEXT_CHAR_LIMIT):
    """Trim history in whole assistant(tool_calls)+tool blocks (API-valid pairing)."""
    if not messages:
        return messages
    p0, rest = messages[0], messages[1:]
    blocks, i = [], 0
    while i < len(rest):
        m = rest[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            block = [m]; j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool":
                block.append(rest[j]); j += 1
            blocks.append(block); i = j
        else:
            blocks.append([m]); i += 1
    total, kept = len(str(p0.get("content") or "")), []
    for b in reversed(blocks):
        size = sum(len(str(m.get("content") or "")) for m in b)
        if total + size > char_limit and kept:
            break
        kept.insert(0, b); total += size
    out = [p0]
    for b in kept:
        out.extend(b)
    return out


class LLMClient:
    def __init__(self, api_base, model, api_key_env):
        from openai import OpenAI
        key = os.environ.get(api_key_env, "") if api_key_env and api_key_env.lower() != "none" else "none"
        if not key:
            raise RuntimeError(f"Set {api_key_env} in your environment first.")
        self.client = OpenAI(api_key=key or "none", base_url=api_base)
        self.model = model
    def chat(self, messages, tool_choice=None, model=None, retries=5):
        kw = {}
        if tool_choice:
            kw["tool_choice"] = tool_choice
        for k in range(retries):
            try:
                return self.client.chat.completions.create(model=model or self.model, messages=build_context(messages),
                                                           tools=TOOLS, temperature=0.2, **kw)
            except Exception as e:
                s = str(e)
                if "429" in s or "rate" in s.lower() or "limit" in s.lower():
                    w = 15 * (k + 1); print(f"  [rate limit, waiting {w}s]"); time.sleep(w); continue
                raise
        raise RuntimeError("rate-limited after retries")


def force_submit(llm, messages, evaluator, sandbox, fallback_model="deepseek-chat"):
    """Level-3: force the submission. Thinking models reject tool_choice, so use a
    plain model for the trivial forced call; else the harness submits directly."""
    try:
        r = llm.chat(messages, tool_choice={"type": "function", "function": {"name": "submit_clean_data"}}, model=fallback_model)
        tcs = r.choices[0].message.tool_calls or []
        if tcs:
            tc = tcs[0]; args = json.loads(tc.function.arguments or "{}")
            res = evaluator.submit(args.get("dataset_path", ""))
            return tc, res, r.usage.prompt_tokens + r.usage.completion_tokens
    except Exception as e:
        print(f"  [forced call failed ({str(e)[:80]}); harness submitting directly]")
    files = sorted(glob.glob(f"{sandbox}/train_cleaned_*.csv"), key=os.path.getmtime)
    if files:
        return None, f"(harness submitted {files[-1]}) {evaluator.submit(files[-1])}", 0
    return None, None, 0


def run_agent(llm, shell, evaluator, P0, sandbox, log=print, token_budget=TOKEN_BUDGET):
    messages = [{"role": "user", "content": P0}]
    tokens, it, since_submit, n_sub = 0, 0, 0, 0
    last_submit_time = 0.0
    REM = (f"Reminder: you have explored for a while without submitting. Apply the fixes you identified: reload {sandbox}/train.csv, "
           f"apply all corrections in one block, save as {sandbox}/train_cleaned_vN.csv, print('saved'), then call submit_clean_data.")
    FIN = (f"STOP EXPLORING NOW. In your NEXT code call: reload {sandbox}/train.csv, apply ALL fixes identified, save as "
           f"{sandbox}/train_cleaned_v1.csv, print('saved'); then immediately call submit_clean_data with that path.")
    def newest_cleaned():
        files = sorted(glob.glob(f"{sandbox}/train_cleaned_*.csv"), key=os.path.getmtime)
        return files[-1] if files else None
    def cleaned_exists():
        return newest_cleaned() is not None
    while tokens < token_budget:
        it += 1
        log(f"===== Iteration {it} (tokens {tokens}/{token_budget}) =====")
        # end of budget: if a cleaned file is newer than the last submission, submit it
        # submission (the agent kept working without submitting), the harness submits it.
        nc = newest_cleaned()
        if tokens > 0.92 * token_budget and nc and os.path.getmtime(nc) > last_submit_time + 1:
            res = evaluator.submit(nc)
            log(f"[last-chance] harness submitted {nc}: {res}")
            if "[ERROR]" not in str(res): n_sub += 1
            last_submit_time = time.time(); since_submit = 0
            messages.append({"role": "user", "content": f"System note: your latest saved file was submitted for you: {res}"})
        if n_sub == 0 and cleaned_exists() and (since_submit >= 2 or tokens > FORCE_BUDGET_FRACTION * token_budget):
            log("[FORCING submission - level 3]")
            tc, res, extra = force_submit(llm, messages, evaluator, sandbox)
            tokens += extra
            if res is not None:
                log(f"[eval] {res}")
                if "[ERROR]" not in str(res): n_sub += 1
                since_submit = 0
                if tc is not None:
                    messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(res)})
                else:
                    messages.append({"role": "user", "content": f"System note: {res}. Continue - apply remaining fixes and submit again."})
                continue
        r = llm.chat(messages)
        tokens += r.usage.prompt_tokens + r.usage.completion_tokens
        msg = r.choices[0].message
        content, tcs = msg.content or "", msg.tool_calls or []
        if content:
            log(f"[assistant] {content[:300]}")
        am = {"role": "assistant", "content": content}
        if tcs:
            am["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tcs]
        messages.append(am)
        if not tcs:
            log("No tool call -> agent is done.")
            break
        for tc in tcs:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            if tc.function.name == "execute_code_ipython_shell":
                res = shell.run(args.get("code", "")) if args.get("code", "").strip() else "[ERROR] Empty code."
                log(f"[ipython] {res[:300]}")
            elif tc.function.name == "submit_clean_data":
                res = evaluator.submit(args.get("dataset_path", ""))
                log(f"[eval] {res}")
                if "[ERROR]" not in str(res): n_sub += 1; last_submit_time = time.time()
                since_submit = -1
            else:
                res = f"[ERROR] unknown tool {tc.function.name}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": res})
        since_submit += 1
        if since_submit >= FINALIZE_AT and (n_sub == 0 or since_submit % FINALIZE_AT == 0):
            log("[escalation L2] FINALIZE order"); messages.append({"role": "user", "content": FIN if n_sub == 0 else
                f"You have analysed for a while since your last submission. APPLY the fixes you have identified NOW: reload {sandbox}/train.csv, apply ALL fixes (old and new) in one block, save as a NEW train_cleaned_vN.csv, print('saved'), and submit it. Unsubmitted analysis is lost work."})
        elif since_submit >= REMIND_EVERY and since_submit % REMIND_EVERY == 0:
            log("[escalation L1] submit reminder"); messages.append({"role": "user", "content": REM})
    return evaluator, tokens
