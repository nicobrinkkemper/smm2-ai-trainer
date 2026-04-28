"""Tests for the dataset_grow module and the dashboard views."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phase_b.dataset_grow as grow_mod
import phase_b.dashboard as dash_mod
from phase_b.context import ContextBundle
from phase_b.dataset_grow import (
    GrowConfig,
    SUCCESS_VERDICTS_DEFAULT,
    build_chatml_row,
    grow_dataset,
    successful_attempts,
)
from phase_b.dashboard import (
    overall_summary,
    recent_attempts,
    escalation_queue,
    render_status,
    render_recent,
    render_escalations,
)
from phase_b.evaluate import EvaluationResult
from phase_b.results import open_db, log_attempt
from phase_b.runner import DecompResult


# --------------------------------------------------------------------------- #
# Helpers shared with the evaluator suite
# --------------------------------------------------------------------------- #

def _make_decomp(model="gemma2:27b", function_id="0x71", function_name="foo",
                 candidate="void foo() {}", error=None, started_at="2026-04-28T10:00:00+00:00"):
    return DecompResult(
        function_id=function_id, function_name=function_name, quality="W",
        model=model, prompt_chars=1000, response_chars=200,
        candidate_cpp=candidate, artifact_path=None,
        started_at=started_at, finished_at=started_at,
        tokens_prompt=80, tokens_eval=40, duration_ns=int(1.5e9),
        error=error, raw_response="",
    )


def _make_eval(verdict="MATCH", function_id="0x71", function_name="foo",
               candidate_path="src/auto/foo.cpp", build_ok=True):
    return EvaluationResult(
        function_id=function_id, function_name=function_name,
        verdict=verdict,
        severity={"MATCH":"NONE","ACCEPT_W":"COSMETIC","FIX":"LOGICAL",
                  "ESCALATE":"LOGICAL"}.get(verdict,"LOGICAL"),
        summary={}, build_ok=build_ok, expected_size=8, actual_size=8,
        edits_json="[]", candidate_path=candidate_path,
        started_at="t0", finished_at="t1",
    )


# --------------------------------------------------------------------------- #
# successful_attempts
# --------------------------------------------------------------------------- #

def test_successful_attempts_returns_match_only(tmp_path):
    conn = open_db(tmp_path / "r.sqlite")
    log_attempt(conn, decomp_result=_make_decomp(function_id="0xA"),
                eval_result=_make_eval(verdict="MATCH", function_id="0xA"))
    log_attempt(conn, decomp_result=_make_decomp(function_id="0xB"),
                eval_result=_make_eval(verdict="FIX", function_id="0xB"))
    log_attempt(conn, decomp_result=_make_decomp(function_id="0xC"),
                eval_result=_make_eval(verdict="ACCEPT_W", function_id="0xC"))
    rows = successful_attempts(conn)
    ids = {r["function_id"] for r in rows}
    assert ids == {"0xA", "0xC"}


def test_successful_attempts_picks_most_recent_per_function(tmp_path):
    conn = open_db(tmp_path / "r.sqlite")
    log_attempt(conn, decomp_result=_make_decomp(
                    function_id="0xA", started_at="2026-04-28T08:00:00+00:00"),
                eval_result=_make_eval(verdict="MATCH", function_id="0xA",
                                       candidate_path="old.cpp"))
    log_attempt(conn, decomp_result=_make_decomp(
                    function_id="0xA", started_at="2026-04-28T10:00:00+00:00"),
                eval_result=_make_eval(verdict="MATCH", function_id="0xA",
                                       candidate_path="new.cpp"))
    rows = successful_attempts(conn)
    assert len(rows) == 1
    assert rows[0]["candidate_path"] == "new.cpp"


def test_build_chatml_row_shape():
    row = build_chatml_row(
        function_name="reset",
        asm_text="ldr w0, [x1]\nret",
        candidate_cpp="void reset() {}\n",
    )
    assert "messages" in row
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "ldr w0" in row["messages"][1]["content"]
    assert "void reset() {}" in row["messages"][2]["content"]


# --------------------------------------------------------------------------- #
# grow_dataset (with stubbed fetch_context + on-disk candidate file)
# --------------------------------------------------------------------------- #

def _stub_bundle(fid, name="foo", disasm=None):
    return ContextBundle(raw={
        "function": {
            "id": fid, "addr": 0x1000, "size": 8,
            "name": name, "quality": "W", "attempts": 0,
        },
        "callers": [], "callees": [], "neighbours": [],
        "disasm": disasm or [
            {"offset": 0, "bytes_hex": "00000091",
             "mnemonic": "mov", "op_str": "x0, x0"},
        ],
        "features": {}, "gotchas": [], "struct_hints": [],
    })


def test_grow_dataset_writes_one_jsonl_per_match(monkeypatch, tmp_path):
    db = tmp_path / "r.sqlite"
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("void foo() {}\n")
    conn = open_db(db)
    log_attempt(conn,
                decomp_result=_make_decomp(function_id="0xA"),
                eval_result=_make_eval(verdict="MATCH",
                                       function_id="0xA",
                                       candidate_path=str(candidate)))
    conn.close()

    monkeypatch.setattr(grow_mod, "fetch_context",
                        lambda target, **kw: _stub_bundle(target, name="reset"))

    out = tmp_path / "out.jsonl"
    summary = grow_dataset(GrowConfig(db_path=db, out_path=out))
    assert summary.rows_written == 1
    line = out.read_text().strip()
    obj = json.loads(line)
    assert obj["function_id"] == "0xA"
    assert obj["messages"][2]["content"].strip() == "void foo() {}"
    sidecar = out.with_suffix(".jsonl.meta.json")
    meta = json.loads(sidecar.read_text())
    assert "0xA" in meta["function_ids"]


def test_grow_dataset_dedupes(monkeypatch, tmp_path):
    db = tmp_path / "r.sqlite"
    candidate = tmp_path / "candidate.cpp"
    candidate.write_text("void foo() {}\n")
    conn = open_db(db)
    log_attempt(conn,
                decomp_result=_make_decomp(function_id="0xA"),
                eval_result=_make_eval(verdict="MATCH",
                                       function_id="0xA",
                                       candidate_path=str(candidate)))
    conn.close()
    monkeypatch.setattr(grow_mod, "fetch_context",
                        lambda target, **kw: _stub_bundle(target))
    out = tmp_path / "out.jsonl"
    grow_dataset(GrowConfig(db_path=db, out_path=out))
    summary = grow_dataset(GrowConfig(db_path=db, out_path=out))
    assert summary.rows_written == 0
    assert summary.rows_skipped_duplicate == 1
    # The output JSONL still has only one line.
    assert out.read_text().strip().count("\n") == 0


def test_grow_dataset_holdout(monkeypatch, tmp_path):
    db = tmp_path / "r.sqlite"
    cand = tmp_path / "candidate.cpp"
    cand.write_text("void foo() {}\n")
    conn = open_db(db)
    for fid in ("0xA", "0xB"):
        log_attempt(
            conn,
            decomp_result=_make_decomp(function_id=fid),
            eval_result=_make_eval(
                verdict="MATCH", function_id=fid, candidate_path=str(cand)
            ),
        )
    conn.close()
    monkeypatch.setattr(grow_mod, "fetch_context",
                        lambda t, **kw: _stub_bundle(t))
    out = tmp_path / "out.jsonl"
    summary = grow_dataset(
        GrowConfig(db_path=db, out_path=out, holdout_set={"0xB"})
    )
    assert summary.rows_written == 1
    assert summary.rows_skipped_holdout == 1


def test_grow_dataset_skips_missing_candidate_file(monkeypatch, tmp_path):
    db = tmp_path / "r.sqlite"
    conn = open_db(db)
    log_attempt(conn,
                decomp_result=_make_decomp(function_id="0xA"),
                eval_result=_make_eval(verdict="MATCH",
                                       function_id="0xA",
                                       candidate_path=str(tmp_path / "nope.cpp")))
    conn.close()
    monkeypatch.setattr(grow_mod, "fetch_context",
                        lambda t, **kw: _stub_bundle(t))
    out = tmp_path / "out.jsonl"
    summary = grow_dataset(GrowConfig(db_path=db, out_path=out))
    assert summary.rows_written == 0
    assert summary.rows_skipped_missing_candidate == 1


# --------------------------------------------------------------------------- #
# Dashboard views
# --------------------------------------------------------------------------- #

@pytest.fixture
def populated_db(tmp_path):
    db = tmp_path / "r.sqlite"
    conn = open_db(db)
    log_attempt(
        conn,
        decomp_result=_make_decomp("gemma2:27b", function_id="0xA",
                                   function_name="reset",
                                   started_at="2026-04-28T08:00:00+00:00"),
        eval_result=_make_eval("MATCH", function_id="0xA",
                               function_name="reset"),
    )
    log_attempt(
        conn,
        decomp_result=_make_decomp("qwen2.5-coder:7b", function_id="0xB",
                                   function_name="changeState",
                                   started_at="2026-04-28T09:00:00+00:00"),
        eval_result=_make_eval("FIX", function_id="0xB",
                               function_name="changeState"),
    )
    log_attempt(
        conn,
        decomp_result=_make_decomp("qwen2.5-coder:7b", function_id="0xB",
                                   function_name="changeState",
                                   started_at="2026-04-28T10:00:00+00:00"),
        eval_result=_make_eval("FIX", function_id="0xB",
                               function_name="changeState"),
    )
    log_attempt(
        conn,
        decomp_result=_make_decomp("gemma2:27b", function_id="0xC",
                                   function_name="finalize",
                                   started_at="2026-04-28T11:00:00+00:00",
                                   error="OllamaError"),
        eval_result=None,
    )
    conn.close()
    return db


def test_overall_summary_counts(populated_db):
    s = overall_summary(open_db(populated_db))
    assert s["attempts"] == 4
    assert s["unique_functions"] == 3
    assert s["matches"] == 1
    assert s["fix"] == 2
    assert s["errors"] == 1


def test_recent_attempts_ordered_desc(populated_db):
    rows = recent_attempts(open_db(populated_db), limit=10)
    starts = [r["started_at"] for r in rows]
    assert starts == sorted(starts, reverse=True)


def test_escalation_queue_includes_only_fix_or_escalate(populated_db):
    rows = escalation_queue(open_db(populated_db), limit=10)
    assert len(rows) == 1
    assert rows[0]["function_id"] == "0xB"
    assert rows[0]["total_attempts"] == 2
    assert rows[0]["verdict"] == "FIX"


def test_render_status_smoke(populated_db):
    out = render_status(open_db(populated_db))
    assert "Attempts:" in out
    assert "gemma2:27b" in out
    assert "qwen2.5-coder:7b" in out


def test_render_recent_smoke(populated_db):
    out = render_recent(open_db(populated_db), limit=10)
    assert "reset" in out or "changeState" in out
    assert "MATCH" in out


def test_render_escalations_smoke(populated_db):
    out = render_escalations(open_db(populated_db), limit=10)
    assert "0xB" in out
    assert "changeState" in out


# --------------------------------------------------------------------------- #
# CLI mains
# --------------------------------------------------------------------------- #

def test_dashboard_cli_status(populated_db, capsys):
    rc = dash_mod.cli_main(["--db", str(populated_db), "status"])
    assert rc == 0
    assert "Matches" in capsys.readouterr().out


def test_dashboard_cli_recent(populated_db, capsys):
    rc = dash_mod.cli_main(["--db", str(populated_db), "recent", "--limit", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict" in out


def test_dashboard_cli_escalations(populated_db, capsys):
    rc = dash_mod.cli_main(["--db", str(populated_db), "escalations"])
    assert rc == 0
    assert "0xB" in capsys.readouterr().out


def test_grow_cli_smoke(monkeypatch, populated_db, tmp_path, capsys):
    cand = tmp_path / "candidate.cpp"
    cand.write_text("void reset() {}\n")
    # Re-open and patch the candidate path on the MATCH row so the CLI
    # finds something to emit.
    conn = open_db(populated_db)
    conn.execute(
        "UPDATE attempt SET candidate_path = ? WHERE function_id = '0xA'",
        (str(cand),),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        grow_mod, "fetch_context",
        lambda t, **kw: _stub_bundle(t, name="reset"),
    )
    out_path = tmp_path / "out.jsonl"
    rc = grow_mod.cli_main([
        "--db", str(populated_db),
        "--out", str(out_path),
    ])
    assert rc == 0
    assert "wrote 1 new rows" in capsys.readouterr().out
