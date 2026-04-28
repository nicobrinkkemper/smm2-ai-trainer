"""Auto-grow the SFT training dataset from successful Phase B attempts.

After the iteration loop produces a MATCH (or ACCEPT_W) candidate for a
function, that (assembly → C++) pair is high-quality, in-distribution
training data — usually higher signal than the equivalent pair extracted
from an existing source file, because the model itself produced the
solution after seeing the structured-diff feedback.

This module reads the Phase B ledger, picks rows whose verdict is in a
configured success set, fetches the function's disassembly via the FKB
context CLI, and emits ChatML lines (matching ``dataset_v3_chatml.jsonl``
format) that can be appended to the training dataset.

To prevent train/eval leakage we record the ``function_id`` of every
emitted row in a sidecar ``<jsonl>.meta.json`` and skip duplicates on
subsequent runs.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .context import DEFAULT_DECOMP_REPO, fetch_context
from .results import open_db


SUCCESS_VERDICTS_DEFAULT = ("MATCH", "ACCEPT_W")


@dataclass
class GrowConfig:
    """Configuration for dataset growth."""
    db_path: Optional[Path] = None
    decomp_repo: Optional[Path] = None
    out_path: Path = Path("dataset_v3_chatml_phase_b.jsonl")
    success_verdicts: tuple[str, ...] = SUCCESS_VERDICTS_DEFAULT
    holdout_set: Optional[set[str]] = None     # function_ids never to emit
    limit: Optional[int] = None
    dedupe: bool = True


def successful_attempts(
    conn: sqlite3.Connection,
    *,
    success_verdicts: Iterable[str] = SUCCESS_VERDICTS_DEFAULT,
    limit: Optional[int] = None,
) -> list[sqlite3.Row]:
    """Return one row per (function_id) that ever succeeded — the most
    recent successful attempt wins (so retraining sees the latest fix)."""
    placeholders = ",".join("?" * len(tuple(success_verdicts)))
    sql = f"""
        SELECT function_id, function_name, model, candidate_path, started_at,
               verdict, summary_json
          FROM attempt
         WHERE verdict IN ({placeholders})
           AND candidate_path IS NOT NULL
         GROUP BY function_id
         HAVING MAX(started_at)
         ORDER BY started_at DESC
    """
    if limit:
        sql += " LIMIT ?"
        rows = conn.execute(sql, (*success_verdicts, limit)).fetchall()
    else:
        rows = conn.execute(sql, tuple(success_verdicts)).fetchall()
    return rows


def _format_disasm(disasm: list[dict]) -> str:
    lines = []
    for ins in disasm:
        offset = int(ins["offset"])
        lines.append(f"{offset:>4x}: {ins['mnemonic']}\t{ins['op_str']}".rstrip())
    return "\n".join(lines)


def build_chatml_row(*, function_name: str, asm_text: str, candidate_cpp: str) -> dict:
    """Match the existing ``dataset_v3_chatml.jsonl`` shape: a ChatML
    conversation with a system prompt, a user prompt holding the asm, and
    the assistant's reply being the matching C++."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert C++ AArch64 decompilation assistant. "
                    "Output the C++ function body only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Decompile {function_name} (AArch64, Clang 8 -O2 -enable-post-misched=false):\n\n"
                    f"```asm\n{asm_text}\n```"
                ),
            },
            {"role": "assistant", "content": candidate_cpp.strip()},
        ],
        "function_id": "",  # caller fills in (we want it on the metadata sidecar)
    }


@dataclass
class GrowSummary:
    rows_written: int
    rows_skipped_holdout: int
    rows_skipped_duplicate: int
    rows_skipped_missing_candidate: int


def grow_dataset(cfg: GrowConfig) -> GrowSummary:
    """Read the ledger, write new training rows. Returns counts.

    We append to ``cfg.out_path`` rather than rewrite, so multiple runs
    accumulate without losing prior data. Sidecar
    ``<out>.meta.json`` tracks emitted ``function_id``s for dedupe.
    """
    conn = open_db(cfg.db_path)
    rows = successful_attempts(
        conn, success_verdicts=cfg.success_verdicts, limit=cfg.limit
    )

    sidecar = cfg.out_path.with_suffix(cfg.out_path.suffix + ".meta.json")
    emitted_ids: set[str]
    if sidecar.exists():
        emitted_ids = set(json.loads(sidecar.read_text()).get("function_ids", []))
    else:
        emitted_ids = set()

    summary = GrowSummary(0, 0, 0, 0)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cfg.out_path, "a") as f:
        for r in rows:
            fid = r["function_id"]
            if cfg.holdout_set and fid in cfg.holdout_set:
                summary.rows_skipped_holdout += 1
                continue
            if cfg.dedupe and fid in emitted_ids:
                summary.rows_skipped_duplicate += 1
                continue
            cand_path_str = r["candidate_path"]
            if not cand_path_str:
                summary.rows_skipped_missing_candidate += 1
                continue
            candidate_path = _resolve_candidate_path(cand_path_str, cfg.decomp_repo)
            if not candidate_path or not candidate_path.exists():
                summary.rows_skipped_missing_candidate += 1
                continue
            candidate_cpp = candidate_path.read_text()

            try:
                ctx = fetch_context(fid, decomp_repo=cfg.decomp_repo)
            except Exception:
                summary.rows_skipped_missing_candidate += 1
                continue

            row = build_chatml_row(
                function_name=ctx.function_name,
                asm_text=_format_disasm(ctx.disasm),
                candidate_cpp=candidate_cpp,
            )
            row["function_id"] = fid
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            emitted_ids.add(fid)
            summary.rows_written += 1

    sidecar.write_text(json.dumps({
        "function_ids": sorted(emitted_ids),
        "out_path": str(cfg.out_path),
    }, indent=2))
    conn.close()
    return summary


def _resolve_candidate_path(p: str, decomp_repo: Optional[Path]) -> Optional[Path]:
    """Candidate paths in the ledger may be either absolute (when written
    by the runner) or repo-relative (when written by the evaluator)."""
    path = Path(p)
    if path.is_absolute():
        return path
    repo = (decomp_repo or DEFAULT_DECOMP_REPO).resolve()
    return repo / path


def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="phase_b grow", description=__doc__)
    p.add_argument("--db", default=None, help="results.sqlite path")
    p.add_argument("--decomp-repo", default=None)
    p.add_argument(
        "--out", default="dataset_v3_chatml_phase_b.jsonl",
        help="output JSONL (appended; sidecar holds emitted IDs for dedupe)",
    )
    p.add_argument(
        "--verdict", action="append", default=None,
        help="success verdicts to emit (default MATCH ACCEPT_W; repeatable)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--holdout-file", default=None,
        help="file with one function_id per line — never emit these",
    )
    p.add_argument(
        "--no-dedupe", action="store_true",
        help="emit even if function_id is already in the sidecar",
    )
    args = p.parse_args(argv)

    holdout = None
    if args.holdout_file:
        holdout = {
            line.strip() for line in Path(args.holdout_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
    cfg = GrowConfig(
        db_path=Path(args.db) if args.db else None,
        decomp_repo=Path(args.decomp_repo) if args.decomp_repo else None,
        out_path=Path(args.out),
        success_verdicts=tuple(args.verdict) if args.verdict else SUCCESS_VERDICTS_DEFAULT,
        holdout_set=holdout,
        limit=args.limit,
        dedupe=not args.no_dedupe,
    )
    summary = grow_dataset(cfg)
    print(f"wrote {summary.rows_written} new rows -> {cfg.out_path}")
    print(f"  skipped holdout:           {summary.rows_skipped_holdout}")
    print(f"  skipped duplicate:         {summary.rows_skipped_duplicate}")
    print(f"  skipped missing candidate: {summary.rows_skipped_missing_candidate}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
