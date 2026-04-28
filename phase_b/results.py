"""SQLite ledger for Phase B attempts.

Schema is intentionally flat — every row is one (function, model, ts) attempt.
Re-runs simply append. Aggregations (mean tokens, verdict distribution per
model, etc.) are queries, not state.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path(
    os.environ.get(
        "PHASE_B_DB",
        str(Path(__file__).resolve().parent / "results.sqlite"),
    )
)

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS attempt (
    id              INTEGER PRIMARY KEY,
    function_id     TEXT NOT NULL,
    function_name   TEXT NOT NULL,
    model           TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    verdict         TEXT,                  -- MATCH/ACCEPT_W/FIX/ESCALATE/BUILD_FAIL/RESOLVE_FAIL
    severity        TEXT,                  -- NONE/COSMETIC/PATTERN/LOGICAL
    summary_json    TEXT,                  -- {"scheduling": 1, "constant_different": 0, ...}
    edits_json      TEXT,                  -- per-edit details (truncated when very long)
    candidate_path  TEXT,
    prompt_chars    INTEGER,
    response_chars  INTEGER,
    tokens_prompt   INTEGER,
    tokens_eval     INTEGER,
    duration_ns     INTEGER,
    build_ok        INTEGER,               -- 0/1
    build_error     TEXT,
    inference_error TEXT
);

CREATE INDEX IF NOT EXISTS attempt_function_idx ON attempt(function_id, started_at DESC);
CREATE INDEX IF NOT EXISTS attempt_model_idx    ON attempt(model, started_at DESC);
CREATE INDEX IF NOT EXISTS attempt_verdict_idx  ON attempt(verdict);
"""


def open_db(path: Optional[os.PathLike] = None) -> sqlite3.Connection:
    db_path = Path(path) if path is not None else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def log_attempt(
    conn: sqlite3.Connection,
    *,
    decomp_result,                      # phase_b.runner.DecompResult
    eval_result=None,                   # phase_b.evaluate.EvaluationResult or None
) -> int:
    """Persist a single attempt. Returns the new row id."""
    summary = {}
    edits = "[]"
    verdict = None
    severity = None
    candidate_path = None
    build_ok = None
    build_error = None
    finished_at = decomp_result.finished_at

    if eval_result is not None:
        verdict = eval_result.verdict
        severity = eval_result.severity
        summary = eval_result.summary
        edits = eval_result.edits_json
        candidate_path = eval_result.candidate_path
        build_ok = 1 if eval_result.build_ok else 0
        build_error = eval_result.build_error
        finished_at = eval_result.finished_at
    elif decomp_result.error is None:
        # We have a candidate but no eval; the candidate file is on disk.
        candidate_path = (
            str(decomp_result.artifact_path)
            if decomp_result.artifact_path else None
        )

    cur = conn.execute(
        """
        INSERT INTO attempt (
            function_id, function_name, model, started_at, finished_at,
            verdict, severity, summary_json, edits_json,
            candidate_path,
            prompt_chars, response_chars,
            tokens_prompt, tokens_eval, duration_ns,
            build_ok, build_error, inference_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decomp_result.function_id,
            decomp_result.function_name,
            decomp_result.model,
            decomp_result.started_at,
            finished_at,
            verdict,
            severity,
            json.dumps(summary),
            edits,
            candidate_path,
            decomp_result.prompt_chars,
            decomp_result.response_chars,
            decomp_result.tokens_prompt,
            decomp_result.tokens_eval,
            decomp_result.duration_ns,
            build_ok,
            build_error,
            decomp_result.error,
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_attempts(
    conn: sqlite3.Connection,
    *,
    function_id: Optional[str] = None,
    model: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    where = []
    params: list = []
    if function_id is not None:
        where.append("function_id = ?")
        params.append(function_id)
    if model is not None:
        where.append("model = ?")
        params.append(model)
    if verdict is not None:
        where.append("verdict = ?")
        params.append(verdict)
    sql = "SELECT * FROM attempt"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_at DESC LIMIT ?"
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def model_scoreboard(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Per-model verdict distribution."""
    return conn.execute(
        """
        SELECT model,
               COUNT(*)                                            AS attempts,
               SUM(CASE WHEN verdict = 'MATCH'        THEN 1 ELSE 0 END) AS matches,
               SUM(CASE WHEN verdict = 'ACCEPT_W'     THEN 1 ELSE 0 END) AS accept_w,
               SUM(CASE WHEN verdict = 'FIX'          THEN 1 ELSE 0 END) AS fix,
               SUM(CASE WHEN verdict = 'ESCALATE'     THEN 1 ELSE 0 END) AS escalate,
               SUM(CASE WHEN verdict = 'BUILD_FAIL'   THEN 1 ELSE 0 END) AS build_fail,
               AVG(tokens_prompt + tokens_eval)                    AS mean_tokens,
               AVG(duration_ns / 1.0e9)                            AS mean_seconds
          FROM attempt
         WHERE verdict IS NOT NULL
         GROUP BY model
         ORDER BY matches DESC, accept_w DESC
        """
    ).fetchall()
