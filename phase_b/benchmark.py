"""Benchmark harness: run N candidates × M models through the full Phase B
loop and emit a scoreboard.

Picks candidates via smm2-decomp's ``tools.fkb.cli ready`` (so the same
score function used in human workflows drives the benchmark). Drives one
attempt per (function, model) combination through the runner+evaluator,
logs each into ``results.sqlite``, and prints / writes a CSV summary at
the end.

Build is invoked once per candidate, not per model — the candidate file
is overwritten in place between models. This is correct because the
evaluator restores the source tree after each attempt.

Designed to be re-runnable: existing rows in ``attempt`` are appended to,
not replaced. ``--skip-already`` filters out (function, model) pairs that
already have any prior attempt — useful for incremental top-ups.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .context import DEFAULT_DECOMP_REPO, fetch_context
from .evaluate import evaluate
from .results import open_db, log_attempt, model_scoreboard
from .runner import decompile_one


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #

def pick_ready(
    *,
    decomp_repo: Optional[Path] = None,
    db_path: Optional[Path] = None,
    qualities: Sequence[str] = ("U", "W"),
    max_size: int = 400,
    min_size: int = 4,
    limit: int = 50,
    unblocked_only: bool = False,
    timeout: float = 30.0,
) -> list[dict]:
    """Shell to ``tools.fkb.cli ready`` (no JSON output yet, so we use the
    CLI's text output). For the benchmark we need addresses; fall through
    to a structured query if the CLI is unavailable."""
    repo = (decomp_repo or DEFAULT_DECOMP_REPO).resolve()
    cmd = ["python3", "-m", "tools.fkb.cli"]
    if db_path is not None:
        cmd += ["--db", str(db_path)]
    cmd += ["ready"]
    for q in qualities:
        cmd += ["--quality", q]
    cmd += [
        "--min-size", str(min_size),
        "--max-size", str(max_size),
        "--limit", str(limit),
    ]
    if unblocked_only:
        cmd.append("--unblocked-only")

    proc = subprocess.run(
        cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fkb ready failed: {proc.stderr.strip()}")
    return _parse_ready_table(proc.stdout)


def _parse_ready_table(text: str) -> list[dict]:
    """Parse the text scoreboard from ``fkb ready``. Stable enough for
    benchmark use; if the CLI ever grows --json output we'll prefer that."""
    out: list[dict] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip() or line.startswith(("score", "-", "(", "Status:")):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            score = float(parts[0])
            addr = parts[1]
            quality = parts[2]
            size = int(parts[3])
        except ValueError:
            continue
        # callees column looks like "0/2"; attempts is an int; the rest is
        # the function name (which can contain spaces only inside angle
        # brackets — names are mangled, so no spaces in practice).
        try:
            callees_str = parts[4]
            attempts = int(parts[5])
            name = " ".join(parts[6:])
        except (ValueError, IndexError):
            continue
        out.append({
            "addr": addr,
            "score": score,
            "quality": quality,
            "size": size,
            "callees": callees_str,
            "attempts": attempts,
            "name": name,
        })
    return out


# --------------------------------------------------------------------------- #
# Run loop
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkConfig:
    models: Sequence[str]
    candidates: Sequence[dict]
    decomp_repo: Path
    db_path: Optional[Path] = None
    out_csv: Optional[Path] = None
    skip_already: bool = False
    skip_build: bool = False
    temperature: float = 0.1
    num_predict: int = 4096
    progress: bool = True


def _seen_pairs(conn, models: Sequence[str]) -> set[tuple[str, str]]:
    rows = conn.execute(
        "SELECT DISTINCT function_id, model FROM attempt"
    ).fetchall()
    return {(r["function_id"], r["model"]) for r in rows}


def run_benchmark(cfg: BenchmarkConfig) -> list[dict]:
    """Execute the (candidate × model) cross-product, log each attempt,
    return per-attempt rows for CSV emission."""
    rows: list[dict] = []
    conn = open_db(cfg.db_path)
    seen = _seen_pairs(conn, cfg.models) if cfg.skip_already else set()

    total = len(cfg.candidates) * len(cfg.models)
    n = 0
    for cand in cfg.candidates:
        addr = cand["addr"]
        ctx = fetch_context(addr, decomp_repo=cfg.decomp_repo, db_path=cfg.db_path)
        for model in cfg.models:
            n += 1
            pair = (ctx.function_id, model)
            if pair in seen:
                if cfg.progress:
                    print(f"[{n}/{total}] skip {model} {ctx.function_name}",
                          file=sys.stderr)
                continue
            t0 = time.time()
            decomp = decompile_one(
                addr,
                model=model,
                decomp_repo=cfg.decomp_repo,
                temperature=cfg.temperature,
                num_predict=cfg.num_predict,
            )
            eval_res = None
            if decomp.error is None and decomp.candidate_cpp:
                eval_res = evaluate(
                    function_id=ctx.function_id,
                    function_name=ctx.function_name,
                    expected_addr=ctx.function_addr,
                    expected_size=ctx.function_size,
                    candidate_cpp=decomp.candidate_cpp,
                    decomp_repo=cfg.decomp_repo,
                    skip_build=cfg.skip_build,
                )
            log_attempt(conn, decomp_result=decomp, eval_result=eval_res)

            row = {
                "function_id": ctx.function_id,
                "function_name": ctx.function_name,
                "size": ctx.function_size,
                "model": model,
                "verdict": (eval_res.verdict if eval_res else None),
                "severity": (eval_res.severity if eval_res else None),
                "tokens": (decomp.tokens_prompt or 0) + (decomp.tokens_eval or 0),
                "wall_seconds": time.time() - t0,
                "error": decomp.error,
            }
            rows.append(row)
            if cfg.progress:
                v = row["verdict"] or ("ERR" if decomp.error else "?")
                print(
                    f"[{n}/{total}] {v:<12} {model:<22} "
                    f"{ctx.function_name[:60]}  ({row['wall_seconds']:.1f}s)",
                    file=sys.stderr,
                )

    if cfg.out_csv is not None:
        _write_csv(cfg.out_csv, rows)
    conn.close()
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("(no rows)\n")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #

def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="phase_b benchmark", description=__doc__)
    p.add_argument(
        "--decomp-repo", default=None,
        help="path to smm2-decomp checkout",
    )
    p.add_argument(
        "--model", action="append", default=None,
        help="model to benchmark (repeatable). Default: gemma2:27b qwen2.5-coder:7b",
    )
    p.add_argument("--db", default=None, help="results.sqlite path")
    p.add_argument("--out-csv", default=None, help="write per-attempt rows to CSV")
    p.add_argument("--limit", type=int, default=20, help="number of candidate functions")
    p.add_argument("--max-size", type=int, default=400)
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument(
        "--quality", action="append", default=None,
        help="quality letters to consider (repeatable, default U,W)",
    )
    p.add_argument("--unblocked-only", action="store_true")
    p.add_argument(
        "--skip-already", action="store_true",
        help="skip (function, model) pairs already in the ledger",
    )
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--num-predict", type=int, default=4096)
    args = p.parse_args(argv)

    repo = Path(args.decomp_repo) if args.decomp_repo else DEFAULT_DECOMP_REPO
    qualities = tuple(args.quality) if args.quality else ("U", "W")
    models = tuple(args.model) if args.model else ("gemma2:27b", "qwen2.5-coder:7b")

    candidates = pick_ready(
        decomp_repo=repo,
        qualities=qualities,
        min_size=args.min_size,
        max_size=args.max_size,
        limit=args.limit,
        unblocked_only=args.unblocked_only,
    )
    if not candidates:
        print("(no candidates)", file=sys.stderr)
        return 1

    cfg = BenchmarkConfig(
        models=models,
        candidates=candidates,
        decomp_repo=repo,
        db_path=Path(args.db) if args.db else None,
        out_csv=Path(args.out_csv) if args.out_csv else None,
        skip_already=args.skip_already,
        skip_build=args.skip_build,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    rows = run_benchmark(cfg)

    # Print scoreboard summary at the end.
    conn = open_db(cfg.db_path)
    sb = model_scoreboard(conn)
    print()
    print(f"{'model':<24} {'#':>4} {'M':>4} {'W':>4} {'F':>4} {'E':>4} "
          f"{'BF':>4} {'tok':>8} {'sec':>6}")
    print("-" * 80)
    for r in sb:
        print(
            f"{r['model'][:24]:<24} "
            f"{r['attempts']:>4} {r['matches']:>4} {r['accept_w']:>4} "
            f"{r['fix']:>4} {r['escalate']:>4} {r['build_fail']:>4} "
            f"{(r['mean_tokens'] or 0):>8.0f} "
            f"{(r['mean_seconds'] or 0):>6.1f}"
        )
    conn.close()

    if args.out_csv:
        print(f"\n{len(rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
