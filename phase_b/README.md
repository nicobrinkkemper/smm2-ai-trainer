# Phase B — Autonomous Local-Model Decompilation Loop

This package is the bridge between **smm2-decomp**'s Phase A tooling
(FKB, structured diffs, regression hook) and a **local LLM** (Gemma 2,
Qwen 2.5 Coder, …) running under Ollama. Together they form an
end-to-end autonomous decompilation loop: the model proposes candidate
C++, the loop compiles it and structurally diffs against the original,
and either accepts the result, asks the model to iterate, or escalates
to a human.

## The shape of the strategy

```
                ┌────────────────────────────────────┐
                │  smm2-decomp Phase A (already done)│
                │  ─────────────────────────────────  │
                │  fkb context <addr> --json   ◄──────┼──┐
                │  diff_classify (build_diff)         │  │
                │  regression_check pre-commit hook   │  │
                └────────────────────────────────────┘  │
                                                        │
   ┌─────────┐     ┌─────────┐     ┌──────────┐         │
   │  ready  │ ──▶ │ context │ ──▶ │  prompt  │ ──▶ Ollama (Gemma/Qwen/…)
   │ (fkb)   │     │  (json) │     │ (chatml) │         │
   └─────────┘     └─────────┘     └──────────┘         │
                                          │             │
                                          ▼             │
                                    candidate.cpp       │
                                          │             │
                                          ▼             │
   ┌──────────┐   ┌──────────────┐  ┌─────────┐         │
   │ evaluate │ ◀ │ ninja build  │ ◀│ place   │         │
   │ (verdict)│   │  Slope.elf   │  │ src tree│         │
   └──────────┘   └──────────────┘  └─────────┘         │
        │                                               │
        ▼                                               │
   ┌─────────────────────┐                              │
   │ results.sqlite      │                              │
   │  attempt(...)       │                              │
   └─────────────────────┘                              │
        │                                               │
        ├── if FIX → iterate.py feeds diff back ─ ──── ─┘
        ├── if MATCH/ACCEPT_W → dataset_grow.py        
        │         (add to training corpus)
        └── dashboard.py (per-model scoreboard,
                          escalation queue, recent attempts)
```

Every Phase B subcommand reads from or writes to `results.sqlite`. The
ledger is the source of truth; everything else (training data, scoreboards,
escalation queues) is derived from it on demand.

## Why a loop and not just inference

A weak local model's first guess often fails. But the same model, given
the **structured diff** from its prior attempt, can frequently produce
targeted fixes — because diff_classify decomposes the failure into
`scheduling`, `register_rename`, `csel_canonical`, `constant_different`,
etc., and each class has a known fix recipe in `COMPILER_GOTCHAS.md`. The
iteration controller surfaces those recipes back to the model.

Concrete claim: a 10% one-shot rate at five iterations should approach
40% (compounded), not 50% — because oscillation guards bail when the
model can't escape stuck states. The benchmark harness measures the real
shape of this curve.

## Prerequisites

The benchmark and prompt subcommands shell to smm2-decomp's FKB CLI, which
needs a populated FKB sqlite. **Run this once in your smm2-decomp checkout
before the first benchmark / iterate run:**

```bash
cd ~/code/smm2-decomp
python3 -m tools.fkb.cli sync     # 104k rows from functions.csv
python3 -m tools.fkb.cli xref     # ~370k call edges from main.elf
```

If the FKB is empty (`ready` returns "(no candidates)"), the benchmark
will exit with that message. Pass `--fkb-db /custom/path/fkb.sqlite` to
phase_b benchmark if you keep the FKB elsewhere; otherwise the FKB CLI's
default (`smm2-decomp/data/v3.0.3/fkb.sqlite`) is used.

## CLI quickstart

```bash
# 0) Sanity: show the prompt that would be sent (no model needed).
python3 -m phase_b.cli prompt 0x71008B92C0

# 1) One-shot full loop: context → infer → evaluate → log.
python3 -m phase_b.cli attempt 0x71008B92C0 --model gemma2:27b

# 2) Permuter-style budgeted iteration on one function.
python3 -m phase_b.cli iterate 0x71008B92C0 \
    --model gemma2:27b --max-attempts 5

# 3) Cross-product benchmark over many candidates × models.
python3 -m phase_b.benchmark \
    --limit 20 --max-size 200 \
    --model gemma2:27b --model qwen2.5-coder:7b \
    --out-csv bench-2026-04-28.csv

# 4) Grow the training dataset from successful matches (no leakage).
python3 -m phase_b.dataset_grow \
    --out dataset_v3_chatml_phase_b.jsonl \
    --holdout-file my_holdout.txt

# 5) Read-only dashboard.
python3 -m phase_b.dashboard status
python3 -m phase_b.dashboard recent --limit 25
python3 -m phase_b.dashboard escalations
```

## The verdict ladder

| Verdict | Source | Iteration controller behaviour |
|---|---|---|
| `MATCH` | diff_classify | terminal — success |
| `ACCEPT_W` | diff_classify | terminal — success (cosmetic-only diff) |
| `FIX` | diff_classify | feed diff back, retry |
| `ESCALATE` | diff_classify | feed diff back, retry; oscillation guard takes over if stuck |
| `BUILD_FAIL` | evaluator | retry with full source if model produced fragment; bail otherwise |
| `RESOLVE_FAIL` | evaluator | symbol/section missing — usually a configuration bug, escalate to human |

## Module reference

| Module | Job |
|---|---|
| `context.py` | `fetch_context(target)` → ContextBundle (subprocess to fkb cli) |
| `prompt.py` | `build_prompt(ctx)` and `build_iteration_prompt(ctx, prior, diff)` |
| `ollama.py` | Tiny urllib-based Ollama HTTP client |
| `runner.py` | `decompile_one(target, model)` — one-shot context+infer+write artefact |
| `evaluate.py` | `evaluate(...)` — place candidate, build, structurally diff, restore tree |
| `iterate.py` | `iterate_one(...)` — budgeted multi-attempt loop with diff feedback |
| `benchmark.py` | `run_benchmark(cfg)` — N candidates × M models cross product |
| `dataset_grow.py` | `grow_dataset(cfg)` — emit ChatML rows from MATCH/ACCEPT_W attempts |
| `dashboard.py` | Read-only views: status, recent, escalations |
| `results.py` | SQLite ledger schema + `log_attempt`, `get_attempts`, `model_scoreboard` |

## Training feedback loop

`dataset_grow.py` reads MATCH/ACCEPT_W attempts and emits ChatML lines
matching `dataset_v3_chatml.jsonl`'s shape. With the existing Unsloth
training scripts, the workflow is:

1. Run `phase_b benchmark` against the current best model.
2. For functions stuck in FIX, run `phase_b iterate` to drive matches.
3. `phase_b dataset_grow` collects successful candidates → new JSONL.
4. Append to the training corpus (with holdout-set respected).
5. Re-run Unsloth fine-tune, push GGUF.
6. Repeat from (1) with the new model.

The holdout sidecar (`<jsonl>.meta.json`) tracks emitted `function_id`s
to prevent the same function from appearing in both training and
benchmark splits across iterations.

## Tests

```bash
python3 -m pytest tests/
```

68 tests, all hermetic (Ollama, ninja, llvm-nm, readelf are subprocess-
mocked; main.elf and build/Slope are synthesised on the fly). Runs in
under 3 seconds without the toolchain.

## What this isn't

- **Not a training pipeline.** Existing scripts (`train_lora_*.py`,
  `export_qwen.py`, …) handle that; Phase B feeds them.
- **Not a scheduler.** Use cron / `loop` skill for periodic runs.
- **Not multi-machine.** Single-host; Ollama URL is configurable but
  there's no fleet management.
- **Not a database.** `results.sqlite` is a flat ledger; analyses are
  ad-hoc SQL.
