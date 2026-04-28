"""Read-only views over the Phase B ledger.

Three CLI subcommands surface the most useful slices:

- ``status``     — overall numbers + per-model scoreboard
- ``recent``     — last N attempts as a table
- ``escalations``— functions still in FIX/ESCALATE state, ranked by attempt count
                   (the human-review queue)

Implemented as plain SQL over ``results.sqlite``; never mutates state.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Optional

from .results import open_db, model_scoreboard


def overall_summary(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """
        SELECT COUNT(*)                                        AS attempts,
               COUNT(DISTINCT function_id)                     AS unique_functions,
               COUNT(DISTINCT model)                           AS unique_models,
               SUM(CASE WHEN verdict = 'MATCH'    THEN 1 ELSE 0 END) AS matches,
               SUM(CASE WHEN verdict = 'ACCEPT_W' THEN 1 ELSE 0 END) AS accept_w,
               SUM(CASE WHEN verdict = 'FIX'      THEN 1 ELSE 0 END) AS fix,
               SUM(CASE WHEN verdict = 'ESCALATE' THEN 1 ELSE 0 END) AS escalate,
               SUM(CASE WHEN verdict IS NULL      THEN 1 ELSE 0 END) AS errors,
               SUM(tokens_prompt + tokens_eval)                AS total_tokens
          FROM attempt
        """
    ).fetchone()
    return dict(row) if row else {}


def recent_attempts(
    conn: sqlite3.Connection, *, limit: int = 25
) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, started_at, function_name, model, verdict, severity,
               tokens_prompt, tokens_eval, duration_ns, candidate_path
          FROM attempt
         ORDER BY started_at DESC
         LIMIT ?
        """,
        (limit,),
    ).fetchall()


def escalation_queue(
    conn: sqlite3.Connection, *, limit: int = 25
) -> list[sqlite3.Row]:
    """Functions whose most-recent attempt is FIX or ESCALATE, with their
    total attempt count. Hot list for human review."""
    return conn.execute(
        """
        WITH last AS (
            SELECT function_id,
                   MAX(started_at) AS started_at
              FROM attempt
             GROUP BY function_id
        )
        SELECT a.function_id, a.function_name, a.verdict, a.severity, a.model,
               (SELECT COUNT(*) FROM attempt b
                 WHERE b.function_id = a.function_id) AS total_attempts
          FROM attempt a
          JOIN last ON last.function_id = a.function_id
                  AND last.started_at = a.started_at
         WHERE a.verdict IN ('FIX', 'ESCALATE')
         ORDER BY total_attempts DESC, a.started_at DESC
         LIMIT ?
        """,
        (limit,),
    ).fetchall()


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #

def _h_seconds(ns: int | None) -> str:
    if ns is None:
        return "-"
    return f"{ns / 1e9:>5.1f}s"


def render_status(conn: sqlite3.Connection) -> str:
    s = overall_summary(conn)
    if not s or not s.get("attempts"):
        return "(no attempts logged yet)"
    sb = model_scoreboard(conn)

    lines = [
        f"Attempts:        {s['attempts']}",
        f"Unique funcs:    {s['unique_functions']}",
        f"Unique models:   {s['unique_models']}",
        "",
        f"Matches:         {s['matches']}",
        f"Accept-W:        {s['accept_w']}",
        f"Fix:             {s['fix']}",
        f"Escalate:        {s['escalate']}",
        f"Errors (no v):   {s['errors']}",
        f"Total tokens:    {s['total_tokens'] or 0}",
        "",
        f"{'model':<24} {'#':>4} {'M':>4} {'W':>4} {'F':>4} {'E':>4} "
        f"{'BF':>4} {'tok':>8} {'sec':>6}",
        "-" * 80,
    ]
    for r in sb:
        lines.append(
            f"{r['model'][:24]:<24} "
            f"{r['attempts']:>4} {r['matches']:>4} {r['accept_w']:>4} "
            f"{r['fix']:>4} {r['escalate']:>4} {r['build_fail']:>4} "
            f"{(r['mean_tokens'] or 0):>8.0f} "
            f"{(r['mean_seconds'] or 0):>6.1f}"
        )
    return "\n".join(lines)


def render_recent(conn: sqlite3.Connection, limit: int = 25) -> str:
    rows = recent_attempts(conn, limit=limit)
    if not rows:
        return "(no attempts)"
    lines = [
        f"{'when':<19} {'verdict':<10} {'model':<22} {'tok':>6} "
        f"{'sec':>6}  function",
        "-" * 100,
    ]
    for r in rows:
        v = r["verdict"] or "-"
        toks = (r["tokens_prompt"] or 0) + (r["tokens_eval"] or 0)
        lines.append(
            f"{r['started_at'][:19]:<19} {v:<10} "
            f"{(r['model'] or '')[:22]:<22} "
            f"{toks:>6} {_h_seconds(r['duration_ns']):>6}  "
            f"{r['function_name'][:50]}"
        )
    return "\n".join(lines)


def render_escalations(conn: sqlite3.Connection, limit: int = 25) -> str:
    rows = escalation_queue(conn, limit=limit)
    if not rows:
        return "(no functions in FIX/ESCALATE state)"
    lines = [
        f"{'function_id':<22} {'attempts':>8} {'verdict':<10} "
        f"{'model':<22}  function",
        "-" * 100,
    ]
    for r in rows:
        lines.append(
            f"{r['function_id']:<22} {r['total_attempts']:>8}  "
            f"{r['verdict']:<10} {(r['model'] or '')[:22]:<22}  "
            f"{r['function_name'][:50]}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="phase_b dashboard", description=__doc__)
    p.add_argument("--db", default=None, help="results.sqlite path")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("status",       help="overall + per-model scoreboard")
    sr = sub.add_parser("recent",       help="last N attempts")
    sr.add_argument("--limit", type=int, default=25)
    se = sub.add_parser("escalations",  help="functions still in FIX/ESCALATE")
    se.add_argument("--limit", type=int, default=25)
    args = p.parse_args(argv)

    db_path = Path(args.db) if args.db else None
    conn = open_db(db_path)
    if args.cmd == "status":
        print(render_status(conn))
    elif args.cmd == "recent":
        print(render_recent(conn, limit=args.limit))
    elif args.cmd == "escalations":
        print(render_escalations(conn, limit=args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
