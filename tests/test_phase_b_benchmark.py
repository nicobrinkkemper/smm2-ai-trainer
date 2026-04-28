"""Tests for phase_b.benchmark — the cross-product harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phase_b.benchmark as bench
from phase_b import context as phase_b_context
from phase_b import runner as phase_b_runner
from phase_b.benchmark import (
    BenchmarkConfig,
    _parse_ready_table,
    pick_ready,
    run_benchmark,
)
from phase_b.context import ContextBundle
from phase_b.evaluate import EvaluationResult
from phase_b.runner import DecompResult


SAMPLE_TABLE = """\
 score  addr                q    size   callees  attempts  name
--------------------------------------------------------------------------------
  1.00  0x00000071008b8dd0  W     364       0/2    0  _ZN2Lp3Utl12StateMachine10initialize
  1.00  0x00000071008b8f40  W      84       0/1    0  _ZN2Lp3Utl12StateMachine8finalizeEv
  1.00  0x00000071008b9240  W     128       0/0    0  _ZN2Lp3Utl12StateMachine10clearStateEv
  0.95  0x000000710159d510  W     110       0/0    0  sub_710159D510

(4 candidates; quality in ('W',))
"""


# --------------------------------------------------------------------------- #
# Table parser
# --------------------------------------------------------------------------- #

def test_parse_ready_table_basic():
    rows = _parse_ready_table(SAMPLE_TABLE)
    assert len(rows) == 4
    assert rows[0]["addr"] == "0x00000071008b8dd0"
    assert rows[0]["score"] == 1.0
    assert rows[0]["quality"] == "W"
    assert rows[0]["size"] == 364
    assert rows[0]["callees"] == "0/2"
    assert rows[0]["attempts"] == 0
    assert rows[0]["name"].endswith("initialize")


def test_parse_ready_table_skips_header_and_summary():
    rows = _parse_ready_table(SAMPLE_TABLE)
    names = [r["name"] for r in rows]
    assert "score" not in names
    assert all("(" not in n for n in names)


def test_parse_ready_table_handles_empty_input():
    assert _parse_ready_table("") == []
    assert _parse_ready_table("(no candidates)\n\n") == []


def test_parse_ready_table_robust_to_extra_whitespace():
    text = "  1.00   0x00000071008b9240  W   128   0/0   0  foo_bar\n"
    rows = _parse_ready_table(text)
    assert len(rows) == 1
    assert rows[0]["name"] == "foo_bar"


# --------------------------------------------------------------------------- #
# pick_ready (CLI subprocess mocked)
# --------------------------------------------------------------------------- #

def test_pick_ready_invokes_cli(monkeypatch, tmp_path):
    captured = {}
    def stub_run(cmd, *a, **kw):
        captured["cmd"] = cmd
        captured["cwd"] = kw.get("cwd")
        return subprocess.CompletedProcess(
            cmd, 0, SAMPLE_TABLE, ""
        )
    monkeypatch.setattr(subprocess, "run", stub_run)
    rows = pick_ready(
        decomp_repo=tmp_path, qualities=("W",), max_size=400, limit=10,
    )
    assert len(rows) == 4
    assert "ready" in captured["cmd"]
    assert "--quality" in captured["cmd"]
    assert "W" in captured["cmd"]
    assert "--max-size" in captured["cmd"]


def test_pick_ready_threads_db_path_to_fkb_cli(monkeypatch, tmp_path):
    """Found in real-world: pick_ready was discarding db_path, so users
    who pointed at a populated FKB elsewhere got '(no candidates)'."""
    captured = {}
    def stub_run(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(cmd, 0, SAMPLE_TABLE, "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    fkb = tmp_path / "elsewhere.sqlite"
    fkb.touch()
    pick_ready(decomp_repo=tmp_path, db_path=fkb, qualities=("W",))
    # The --db arg must be in the CLI invocation, BEFORE the `ready` subcmd.
    assert "--db" in captured["cmd"]
    db_idx = captured["cmd"].index("--db")
    assert captured["cmd"][db_idx + 1] == str(fkb)
    assert captured["cmd"].index("--db") < captured["cmd"].index("ready")


def test_pick_ready_raises_on_cli_failure(monkeypatch, tmp_path):
    def stub_run(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")
    monkeypatch.setattr(subprocess, "run", stub_run)
    with pytest.raises(RuntimeError, match="boom"):
        pick_ready(decomp_repo=tmp_path)


# --------------------------------------------------------------------------- #
# run_benchmark (full loop with stubs)
# --------------------------------------------------------------------------- #

class _StubBundle:
    def __init__(self, addr_hex, name, size=128):
        self.raw = {
            "function": {
                "id": addr_hex, "addr": int(addr_hex, 16),
                "size": size, "name": name, "quality": "W",
            },
        }
        self.function_id = addr_hex
        self.function_addr = int(addr_hex, 16)
        self.function_size = size
        self.function_name = name
        self.quality = "W"


@pytest.fixture
def benchmark_stubs(monkeypatch, tmp_path):
    """Stub fetch_context, decompile_one, and evaluate so run_benchmark
    is hermetic."""
    bundles = {
        "0x71008b8f40": _StubBundle("0x71008b8f40", "func_a", size=84),
        "0x71008b9240": _StubBundle("0x71008b9240", "func_b", size=128),
    }
    fetch_calls = []

    def stub_fetch(target, **kwargs):
        fetch_calls.append(target)
        return bundles[target]

    decompile_calls = []

    def stub_decompile(target, *, model, **kwargs):
        decompile_calls.append((target, model))
        return DecompResult(
            function_id=bundles[target].function_id,
            function_name=bundles[target].function_name,
            quality="W",
            model=model,
            prompt_chars=1000,
            response_chars=200,
            candidate_cpp=f"// {model}: {target}\n",
            artifact_path=None,
            started_at="2026-04-28T16:00:00+00:00",
            finished_at="2026-04-28T16:00:01+00:00",
            tokens_prompt=100, tokens_eval=50,
            duration_ns=int(1.5e9),
            error=None,
            raw_response="",
        )

    eval_calls = []

    def stub_evaluate(*, function_id, function_name, expected_addr,
                      expected_size, candidate_cpp, **kwargs):
        eval_calls.append((function_id, function_name, kwargs.get("skip_build")))
        # Verdict cycles to test aggregation: gemma2 → MATCH, qwen → FIX.
        verdict = "MATCH" if "gemma2" in candidate_cpp else "FIX"
        sev = "NONE" if verdict == "MATCH" else "LOGICAL"
        return EvaluationResult(
            function_id=function_id, function_name=function_name,
            verdict=verdict, severity=sev,
            summary={}, build_ok=True, expected_size=expected_size,
            actual_size=expected_size, edits_json="[]",
            candidate_path="src/auto/x.cpp",
            started_at="t0", finished_at="t1",
        )

    monkeypatch.setattr(phase_b_context, "fetch_context", stub_fetch)
    monkeypatch.setattr(bench, "fetch_context", stub_fetch)
    monkeypatch.setattr(phase_b_runner, "decompile_one", stub_decompile)
    monkeypatch.setattr(bench, "decompile_one", stub_decompile)
    monkeypatch.setattr(bench, "evaluate", stub_evaluate)

    return bundles, fetch_calls, decompile_calls, eval_calls


def test_run_benchmark_cross_product(benchmark_stubs, tmp_path, capsys):
    bundles, _, decompile_calls, _ = benchmark_stubs
    cfg = BenchmarkConfig(
        models=["gemma2:27b", "qwen2.5-coder:7b"],
        candidates=[
            {"addr": "0x71008b8f40", "score": 1.0, "quality": "W",
             "size": 84, "callees": "0/0", "attempts": 0, "name": "func_a"},
            {"addr": "0x71008b9240", "score": 1.0, "quality": "W",
             "size": 128, "callees": "0/0", "attempts": 0, "name": "func_b"},
        ],
        decomp_repo=tmp_path,
        db_path=tmp_path / "r.sqlite",
        out_csv=tmp_path / "out.csv",
        progress=False,
        skip_build=True,
    )
    rows = run_benchmark(cfg)
    assert len(rows) == 4  # 2 candidates × 2 models
    assert len(decompile_calls) == 4

    # CSV emitted with the right columns and counts.
    out = (tmp_path / "out.csv").read_text()
    assert "verdict" in out.splitlines()[0]
    assert out.count("\n") == 5   # header + 4 rows


def test_run_benchmark_skip_already(benchmark_stubs, tmp_path):
    """With --skip-already, a (function, model) pair already in the ledger
    is not re-run."""
    bundles, _, decompile_calls, _ = benchmark_stubs
    db = tmp_path / "r.sqlite"
    # Pre-populate one row.
    from phase_b.results import open_db, log_attempt
    conn = open_db(db)
    log_attempt(
        conn,
        decomp_result=DecompResult(
            function_id="0x71008b8f40", function_name="func_a", quality="W",
            model="gemma2:27b",
            prompt_chars=0, response_chars=0,
            candidate_cpp="", artifact_path=None,
            started_at="t", finished_at="t", error=None,
            raw_response="",
        ),
        eval_result=EvaluationResult(
            function_id="0x71008b8f40", function_name="func_a",
            verdict="MATCH", severity="NONE", summary={},
            build_ok=True, expected_size=84, actual_size=84,
            edits_json="[]", candidate_path=None,
            started_at="t", finished_at="t",
        ),
    )
    conn.close()
    cfg = BenchmarkConfig(
        models=["gemma2:27b", "qwen2.5-coder:7b"],
        candidates=[
            {"addr": "0x71008b8f40", "score": 1.0, "quality": "W",
             "size": 84, "callees": "0/0", "attempts": 0, "name": "func_a"},
        ],
        decomp_repo=tmp_path,
        db_path=db,
        progress=False,
        skip_already=True,
        skip_build=True,
    )
    rows = run_benchmark(cfg)
    # Skipped (gemma2 already done) + ran (qwen new).
    assert len(rows) == 1
    assert rows[0]["model"] == "qwen2.5-coder:7b"
    # decompile_one only called for the unseen model.
    assert len(decompile_calls) == 1
    assert decompile_calls[0][1] == "qwen2.5-coder:7b"


def test_run_benchmark_handles_inference_error(monkeypatch, tmp_path):
    """If decompile_one returns an error, evaluate is NOT called and the
    row is still logged."""
    def stub_fetch(target, **kwargs):
        return _StubBundle(target, "func_x", size=8)
    monkeypatch.setattr(bench, "fetch_context", stub_fetch)
    monkeypatch.setattr(phase_b_context, "fetch_context", stub_fetch)

    eval_called = {"count": 0}

    def stub_decompile(target, *, model, **kwargs):
        return DecompResult(
            function_id="0x71",
            function_name="func_x", quality="W", model=model,
            prompt_chars=10, response_chars=0, candidate_cpp="",
            artifact_path=None, started_at="t", finished_at="t",
            error="OllamaError: refused",
            raw_response="",
        )
    monkeypatch.setattr(bench, "decompile_one", stub_decompile)
    def stub_evaluate(**kwargs):  # noqa: ARG001
        eval_called["count"] += 1
        raise AssertionError("evaluate should not run on inference failure")
    monkeypatch.setattr(bench, "evaluate", stub_evaluate)

    cfg = BenchmarkConfig(
        models=["m"],
        candidates=[{"addr": "0x71", "score": 0, "quality": "W",
                     "size": 8, "callees": "0/0", "attempts": 0, "name": "x"}],
        decomp_repo=tmp_path,
        db_path=tmp_path / "r.sqlite",
        progress=False,
    )
    rows = run_benchmark(cfg)
    assert len(rows) == 1
    assert rows[0]["verdict"] is None
    assert rows[0]["error"] == "OllamaError: refused"
    assert eval_called["count"] == 0
