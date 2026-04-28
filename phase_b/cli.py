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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
