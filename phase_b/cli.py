"""Command-line entry point.

Examples:
    # default model: gemma2:27b on local Ollama
    python3 -m phase_b.cli decompile 0x71008B92C0

    # different model + custom output dir
    python3 -m phase_b.cli decompile StateMachine::reset \\
        --model qwen2.5-coder:7b \\
        --out /tmp/phase_b

    # dry run: print the prompt that WOULD be sent
    python3 -m phase_b.cli prompt 0x71008B92C0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context import fetch_context
from .prompt import build_prompt
from .runner import decompile_one
from .evaluate import evaluate
from .iterate import iterate_one
from .results import open_db, log_attempt, model_scoreboard


def _cmd_decompile(args: argparse.Namespace) -> int:
    res = decompile_one(
        args.target,
        model=args.model,
        decomp_repo=Path(args.decomp_repo) if args.decomp_repo else None,
        out_dir=Path(args.out) if args.out else None,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    print(f"function: {res.function_name}  ({res.function_id})")
    print(f"  quality:  {res.quality}")
    print(f"  model:    {res.model}")
    print(f"  artifact: {res.artifact_path}")
    if res.tokens_prompt is not None:
        print(
            f"  tokens:   {res.tokens_prompt}+{res.tokens_eval} = "
            f"{(res.tokens_prompt or 0) + (res.tokens_eval or 0)}"
        )
    if res.duration_ns:
        print(f"  duration: {res.duration_ns / 1e9:.1f}s")
    if res.error:
        print(f"  ERROR:    {res.error}", file=sys.stderr)
        return 2
    return 0 if res.candidate_cpp else 1


def _cmd_attempt(args: argparse.Namespace) -> int:
    """One-shot full pipeline: context → infer → evaluate → log."""
    decomp_repo = Path(args.decomp_repo) if args.decomp_repo else None
    res = decompile_one(
        args.target,
        model=args.model,
        decomp_repo=decomp_repo,
        out_dir=Path(args.out) if args.out else None,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    print(f"function: {res.function_name}  ({res.function_id})")
    print(f"  model:    {res.model}")
    if res.error:
        print(f"  ERROR:    {res.error}", file=sys.stderr)
        if not args.no_log:
            with open_db(args.db) as conn:
                log_attempt(conn, decomp_result=res)
        return 2
    if not res.candidate_cpp:
        print("  WARN:     model returned no usable candidate", file=sys.stderr)
        if not args.no_log:
            with open_db(args.db) as conn:
                log_attempt(conn, decomp_result=res)
        return 1

    # Evaluate (compile + diff).
    ctx = fetch_context(args.target, decomp_repo=decomp_repo)
    eval_res = evaluate(
        function_id=ctx.function_id,
        function_name=ctx.function_name,
        expected_addr=ctx.function_addr,
        expected_size=ctx.function_size,
        candidate_cpp=res.candidate_cpp,
        decomp_repo=decomp_repo,
        skip_build=args.skip_build,
    )
    print(f"  verdict:  {eval_res.verdict}    severity: {eval_res.severity}")
    if eval_res.summary:
        cls = ", ".join(f"{k}={v}" for k, v in eval_res.summary.items())
        print(f"  edits:    {cls}")
    if eval_res.build_error:
        print(f"  build:    {eval_res.build_error[:200]}")

    if not args.no_log:
        with open_db(args.db) as conn:
            log_attempt(conn, decomp_result=res, eval_result=eval_res)

    return {
        "MATCH": 0, "ACCEPT_W": 0, "FIX": 1, "ESCALATE": 2,
        "BUILD_FAIL": 3, "RESOLVE_FAIL": 3,
    }.get(eval_res.verdict, 1)


def _cmd_scoreboard(args: argparse.Namespace) -> int:
    with open_db(args.db) as conn:
        rows = model_scoreboard(conn)
    if not rows:
        print("(no attempts logged yet)")
        return 0
    print(f"{'model':<24} {'#':>4} {'M':>4} {'W':>4} {'F':>4} {'E':>4} "
          f"{'BF':>4} {'tok':>8} {'sec':>6}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['model'][:24]:<24} "
            f"{r['attempts']:>4} {r['matches']:>4} {r['accept_w']:>4} "
            f"{r['fix']:>4} {r['escalate']:>4} {r['build_fail']:>4} "
            f"{(r['mean_tokens'] or 0):>8.0f} "
            f"{(r['mean_seconds'] or 0):>6.1f}"
        )
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    ctx = fetch_context(args.target, decomp_repo=Path(args.decomp_repo) if args.decomp_repo else None)
    p = build_prompt(ctx)
    if args.json:
        print(json.dumps(p, indent=2))
    else:
        print("=== SYSTEM ===")
        print(p["system"])
        print()
        print("=== PROMPT ===")
        print(p["prompt"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase_b", description=__doc__)
    p.add_argument(
        "--decomp-repo",
        default=None,
        help="path to smm2-decomp checkout (default $SMM2_DECOMP_REPO or ~/code/smm2-decomp)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_d = sub.add_parser("decompile", help="run one shot through a model")
    sp_d.add_argument("target")
    sp_d.add_argument("--model", default="gemma2:27b")
    sp_d.add_argument("--out", default=None)
    sp_d.add_argument("--temperature", type=float, default=0.1)
    sp_d.add_argument("--num-predict", type=int, default=4096)
    sp_d.set_defaults(func=_cmd_decompile)

    sp_p = sub.add_parser("prompt", help="show the prompt without sending it")
    sp_p.add_argument("target")
    sp_p.add_argument("--json", action="store_true")
    sp_p.set_defaults(func=_cmd_prompt)

    sp_a = sub.add_parser(
        "attempt",
        help="full one-shot loop: context → infer → evaluate → log",
    )
    sp_a.add_argument("target")
    sp_a.add_argument("--model", default="gemma2:27b")
    sp_a.add_argument("--out", default=None)
    sp_a.add_argument("--temperature", type=float, default=0.1)
    sp_a.add_argument("--num-predict", type=int, default=4096)
    sp_a.add_argument("--db", default=None, help="results.sqlite path")
    sp_a.add_argument(
        "--skip-build",
        action="store_true",
        help="don't run ninja; assume build/Slope is already current",
    )
    sp_a.add_argument(
        "--no-log",
        action="store_true",
        help="don't write to results.sqlite",
    )
    sp_a.set_defaults(func=_cmd_attempt)

    sp_s = sub.add_parser("scoreboard", help="per-model verdict distribution")
    sp_s.add_argument("--db", default=None)
    sp_s.set_defaults(func=_cmd_scoreboard)

    sp_i = sub.add_parser(
        "iterate",
        help="iteration controller: budgeted permuter-style loop on one function",
    )
    sp_i.add_argument("target")
    sp_i.add_argument("--model", default="gemma2:27b")
    sp_i.add_argument("--db", default=None, help="results.sqlite path")
    sp_i.add_argument("--max-attempts", type=int, default=5)
    sp_i.add_argument("--max-seconds", type=float, default=600.0)
    sp_i.add_argument("--temperature", type=float, default=0.3)
    sp_i.add_argument("--num-predict", type=int, default=4096)
    sp_i.add_argument("--skip-build", action="store_true")
    sp_i.add_argument("--no-log", action="store_true")
    sp_i.set_defaults(func=_cmd_iterate)

    return p


def _cmd_iterate(args: argparse.Namespace) -> int:
    outcome = iterate_one(
        args.target,
        model=args.model,
        max_attempts=args.max_attempts,
        max_seconds=args.max_seconds,
        decomp_repo=Path(args.decomp_repo) if args.decomp_repo else None,
        db_path=Path(args.db) if args.db else None,
        skip_build=args.skip_build,
        temperature=args.temperature,
        num_predict=args.num_predict,
        log=not args.no_log,
    )
    print(f"function: {outcome.function_name}")
    print(f"  attempts:      {outcome.step_count()}")
    print(f"  final verdict: {outcome.final_verdict or '-'}")
    print(f"  success:       {outcome.success}")
    print(f"  stop reason:   {outcome.stop_reason}")
    for st in outcome.steps:
        v = st.eval.verdict if st.eval else "ERR"
        print(
            f"  [{st.attempt_num}] {v:<12} "
            f"prompt={st.decomp.prompt_chars:>5}c "
            f"resp={st.decomp.response_chars:>5}c {st.duration_s:5.1f}s"
        )
    return 0 if outcome.success else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
