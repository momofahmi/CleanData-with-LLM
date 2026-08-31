# Hybrid Statistical-LLM Agent for Tabular Data Cleaning

MSc dissertation project (University of Surrey, 2026). An LLM cleaning agent is
paired with a statistical audit module so it can repair distribution-level
errors: systematic shifts, conditional overwrites, structured deletions.

(NO API key needed for mode stats only)

## How it works

1. The audit profiles the columns, compares slices of the data, and fits each
   anomaly to a simple mechanism (shift, factor, overwrite, conditional missing).
2. A finding becomes a `REPAIR` only if applying the inverse of the mechanism
   brings the slice back in line with clean reference data. Otherwise it stays
   a `REVIEW`.
3. The agent gets the findings as an evidence report, checks them with its own
   code in a sandbox, and keeps the final say.
4. A gate drops any submission that hurts the validation score. The test split
   is scored once, at the end, outside the agent loop.

Three modes: `stats` (audit only), `llm` (agent only, the published baseline),
`hybrid` (both).

## Layout

| path | role |
|---|---|
| `agent.py`, `tools.py` | the agent and its two tools (shell, submit) |
| `statmod/` | profiler, slice audit, panel audit, repair executor |
| `bench_datasets/` | dataset configs and error injection, one file per dataset |
| `run_split3.py` | 60/20/20 protocol used for all reported runs |
| `run_pipeline.py` | single-run driver called by the protocol |
| `E1_split3_stats.py` | statistics-only runs |
| `verify_against_clean.py` | cell-by-cell scoring against ground truth |
| `data/` | clean datasets (see below) |
| `runs/` | a few sample runs |

## Setup

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY="sk-..."
```

## Data

Titanic ,Hotel Bookings, Adult and the meat-production data are included. 

## Running

```bash
python run_split3.py --dataset hotel --mode hybrid --seed 42
python run_split3.py --dataset hotel --mode llm    --seed 42
python E1_split3_stats.py hotel --data data/hotel_clean.csv --seeds 42
```

Datasets: `titanic`, `hotel`, `meat`, `adult`. Each run writes a timestamped
folder with the transcript, the report, the summary and the submitted files.

## Scale

The dissertation archives 72 runs. One agentic run uses about 200k tokens. The whole study took
around 1,800 API requests and 19 million tokens over two months.

## Not in this repository

Only a few representative runs and main implementations are kept here. 
