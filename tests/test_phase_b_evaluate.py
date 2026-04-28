"""Tests for the Phase B evaluator (Task 2)."""

from __future__ import annotations

import json
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phase_b.evaluate as phase_b_evaluate_mod
import phase_b.results as phase_b_results
from phase_b.evaluate import (
    EvaluationResult,
    place_candidate,
    restore_placement,
    evaluate,
)
from phase_b.results import open_db, log_attempt, get_attempts, model_scoreboard
from phase_b.runner import DecompResult


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #

def _make_repo(tmp_path):
    repo = tmp_path / "smm2-decomp"
    (repo / "src" / "Lp" / "Utl").mkdir(parents=True)
    (repo / "src" / "auto").mkdir(parents=True)
    return repo


def test_place_candidate_overwrites_existing(tmp_path):
    repo = _make_repo(tmp_path)
    target = repo / "src" / "Lp" / "Utl" / "StateMachine.cpp"
    original = "void _ZN2Lp3Utl12StateMachine5resetEv() { /* old */ }\n"
    target.write_text(original)

    placement = place_candidate(
        repo, "_ZN2Lp3Utl12StateMachine5resetEv",
        "void _ZN2Lp3Utl12StateMachine5resetEv() { /* new */ }\n",
    )
    assert placement.pre_existing
    assert placement.path == target
    assert "/* new */" in target.read_text()

    restore_placement(placement)
    assert target.read_text() == original


def test_place_candidate_creates_auto_file_for_unknown(tmp_path):
    repo = _make_repo(tmp_path)
    placement = place_candidate(
        repo, "sub_71XXXXXXXX",
        "void sub_71XXXXXXXX() {}\n",
    )
    assert not placement.pre_existing
    assert placement.path.parent.name == "auto"
    assert placement.path.exists()
    restore_placement(placement)
    assert not placement.path.exists()


def test_safe_filename_strips_dangerous_chars():
    from phase_b.evaluate import _safe_filename
    assert _safe_filename("foo/bar:baz") == "foo_bar_baz"
    assert _safe_filename("") == "anon"
    assert "/" not in _safe_filename("../etc/passwd")


# --------------------------------------------------------------------------- #
# evaluate() with build mocked out
# --------------------------------------------------------------------------- #

def _stub_subprocess_run_factory(behaviour):
    """Build a subprocess.run replacement that dispatches by argv[0]."""
    def _run(cmd, *args, **kwargs):
        if not cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        head = Path(cmd[0]).name if isinstance(cmd[0], str) else cmd[0]
        handler = behaviour.get(head)
        if handler is None:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return handler(cmd, *args, **kwargs)
    return _run


def _build_main_elf_payload(addr, code_bytes):
    """Synthesise a minimal main.elf-like file: TEXT_OFFSET pad + code_bytes
    placed at ``addr - BASE_ADDR``."""
    base = phase_b_evaluate_mod.BASE_ADDR
    text_off = phase_b_evaluate_mod.TEXT_OFFSET_MAIN_ELF
    rel = addr - base
    return b"\x00" * (text_off + rel) + code_bytes


def _build_slope_elf_with(text_va, text_off, sym_addr, code_bytes):
    """Synthesise a Slope-like file with one .text section. We don't need
    a real ELF — readelf and llvm-nm are stubbed; we only need the file
    contents at ``sym_addr - text_va + text_off``."""
    file_off = sym_addr - text_va + text_off
    out = bytearray(b"\x00" * (file_off + len(code_bytes)))
    out[file_off : file_off + len(code_bytes)] = code_bytes
    return bytes(out)


@pytest.fixture
def fake_repo(tmp_path):
    """A repo skeleton with stubbed main.elf, build/Slope, and a place to
    drop the candidate."""
    repo = _make_repo(tmp_path)

    addr = phase_b_evaluate_mod.BASE_ADDR + 0x123450
    code = struct.pack("<I", 0xD2800028) + struct.pack("<I", 0xD65F03C0)  # mov; ret
    (repo / "data" / "v3.0.3").mkdir(parents=True)
    (repo / "data" / "v3.0.3" / "main.elf").write_bytes(
        _build_main_elf_payload(addr, code)
    )
    (repo / "build").mkdir()
    text_va, text_off = 0xb0000, 0xb0000
    sym_addr = 0xc0000
    (repo / "build" / "Slope").write_bytes(
        _build_slope_elf_with(text_va, text_off, sym_addr, code)
    )
    return repo, addr, sym_addr, text_va, text_off, code


def test_evaluate_match_path(monkeypatch, fake_repo, tmp_path):
    repo, addr, sym_addr, text_va, text_off, code = fake_repo
    fname = "_ZN2test4funcEv"

    def nm(cmd, *a, **kw):
        out = f"{sym_addr:016x} T {fname}\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    def readelf(cmd, *a, **kw):
        out = f"  [12] .text   PROGBITS  {text_va:016x}  {text_off:016x}  0001000\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")
    def grep(cmd, *a, **kw):
        # No existing source for the candidate.
        return subprocess.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run_factory({
            "llvm-nm": nm,
            "readelf": readelf,
            "grep": grep,
        }),
    )

    res = evaluate(
        function_id="0x71",
        function_name=fname,
        expected_addr=addr,
        expected_size=len(code),
        candidate_cpp="// stub\n",
        decomp_repo=repo,
        skip_build=True,    # build is the only thing not mockable here
    )
    assert res.verdict == "MATCH"
    assert res.severity == "NONE"
    assert res.build_ok is True
    assert res.actual_size == len(code)
    # Auto-placement file should be cleaned up.
    auto_files = list((repo / "src" / "auto").glob("_phase_b_*.cpp"))
    assert auto_files == []


def test_evaluate_build_failure_short_circuits(monkeypatch, fake_repo):
    repo, addr, *_ = fake_repo

    def ninja(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "fatal: foo")
    def grep(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run_factory({"ninja": ninja, "grep": grep}),
    )

    res = evaluate(
        function_id="0x71", function_name="_ZN2test4funcEv",
        expected_addr=addr, expected_size=8,
        candidate_cpp="// will not compile\n",
        decomp_repo=repo,
    )
    assert res.verdict == "BUILD_FAIL"
    assert res.build_ok is False
    assert "fatal" in (res.build_error or "")


def test_evaluate_resolve_failure_when_symbol_missing(
    monkeypatch, fake_repo
):
    repo, addr, *_ = fake_repo

    def nm(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")  # no symbols
    def readelf(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")
    def grep(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "")
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run_factory({"llvm-nm": nm, "readelf": readelf, "grep": grep}),
    )

    res = evaluate(
        function_id="0x71", function_name="_ZN2test4funcEv",
        expected_addr=addr, expected_size=8,
        candidate_cpp="// stub\n",
        decomp_repo=repo,
        skip_build=True,
    )
    assert res.verdict == "RESOLVE_FAIL"
    assert "could not locate" in (res.build_error or "")


def test_evaluate_pre_existing_file_is_restored(
    monkeypatch, fake_repo
):
    repo, addr, sym_addr, text_va, text_off, code = fake_repo
    fname = "_ZN2test4funcEv"
    # Pre-existing source file mentioning the symbol.
    src = repo / "src" / "Lp" / "Utl" / "Test.cpp"
    original = f"// extern void {fname}();\nvoid {fname}() {{ /* original */ }}\n"
    src.write_text(original)

    def nm(cmd, *a, **kw):
        return subprocess.CompletedProcess(
            cmd, 0, f"{sym_addr:016x} T {fname}\n", ""
        )
    def readelf(cmd, *a, **kw):
        return subprocess.CompletedProcess(
            cmd, 0,
            f"  [12] .text PROGBITS {text_va:016x} {text_off:016x} 0001000\n",
            "",
        )
    def grep_for_existing(cmd, *a, **kw):
        return subprocess.CompletedProcess(cmd, 0, str(src) + "\n", "")
    monkeypatch.setattr(
        subprocess, "run",
        _stub_subprocess_run_factory({
            "llvm-nm": nm, "readelf": readelf, "grep": grep_for_existing,
        }),
    )

    evaluate(
        function_id="0x71", function_name=fname,
        expected_addr=addr, expected_size=len(code),
        candidate_cpp="// candidate\n",
        decomp_repo=repo, skip_build=True,
    )
    # Source file must be restored to its original contents.
    assert src.read_text() == original


# --------------------------------------------------------------------------- #
# results.py — SQLite ledger
# --------------------------------------------------------------------------- #

def _make_decomp(model="gemma2:27b", error=None):
    return DecompResult(
        function_id="0x00000071008b92c0",
        function_name="_ZN2Lp3Utl12StateMachine5resetEv",
        quality="W",
        model=model,
        prompt_chars=2000,
        response_chars=400,
        candidate_cpp="void reset() {}",
        artifact_path=None,
        started_at="2026-04-28T16:00:00+00:00",
        finished_at="2026-04-28T16:00:01+00:00",
        tokens_prompt=120,
        tokens_eval=80,
        duration_ns=int(2e9),
        error=error,
        raw_response="```cpp\nvoid reset() {}\n```",
    )


def _make_eval(verdict="MATCH", severity="NONE", summary=None, build_ok=True,
               build_error=None):
    return EvaluationResult(
        function_id="0x00000071008b92c0",
        function_name="_ZN2Lp3Utl12StateMachine5resetEv",
        verdict=verdict,
        severity=severity,
        summary=summary or {},
        build_ok=build_ok,
        build_error=build_error,
        expected_size=84,
        actual_size=84,
        edits_json="[]",
        candidate_path="src/Lp/Utl/StateMachine.cpp",
        started_at="2026-04-28T16:00:01+00:00",
        finished_at="2026-04-28T16:00:02+00:00",
    )


def test_open_db_creates_schema(tmp_path):
    db = tmp_path / "results.sqlite"
    conn = open_db(db)
    tables = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "attempt" in tables


def test_log_attempt_inserts_row_with_eval(tmp_path):
    conn = open_db(tmp_path / "r.sqlite")
    rid = log_attempt(
        conn,
        decomp_result=_make_decomp(),
        eval_result=_make_eval(verdict="MATCH"),
    )
    assert rid > 0
    rows = get_attempts(conn)
    assert len(rows) == 1
    assert rows[0]["verdict"] == "MATCH"
    assert rows[0]["build_ok"] == 1
    assert rows[0]["model"] == "gemma2:27b"


def test_log_attempt_handles_inference_failure(tmp_path):
    conn = open_db(tmp_path / "r.sqlite")
    rid = log_attempt(
        conn,
        decomp_result=_make_decomp(error="OllamaError: refused"),
        eval_result=None,
    )
    assert rid > 0
    rows = get_attempts(conn)
    assert rows[0]["verdict"] is None
    assert rows[0]["inference_error"] == "OllamaError: refused"


def test_get_attempts_filters(tmp_path):
    conn = open_db(tmp_path / "r.sqlite")
    log_attempt(conn, decomp_result=_make_decomp("a"), eval_result=_make_eval("MATCH"))
    log_attempt(conn, decomp_result=_make_decomp("a"), eval_result=_make_eval("FIX"))
    log_attempt(conn, decomp_result=_make_decomp("b"), eval_result=_make_eval("MATCH"))

    a_rows = get_attempts(conn, model="a")
    assert len(a_rows) == 2
    matches = get_attempts(conn, verdict="MATCH")
    assert len(matches) == 2
    a_match = get_attempts(conn, model="a", verdict="MATCH")
    assert len(a_match) == 1


def test_model_scoreboard_aggregates(tmp_path):
    conn = open_db(tmp_path / "r.sqlite")
    for verdict in ["MATCH", "MATCH", "FIX", "ESCALATE"]:
        log_attempt(
            conn,
            decomp_result=_make_decomp("gemma2:27b"),
            eval_result=_make_eval(verdict=verdict),
        )
    log_attempt(
        conn,
        decomp_result=_make_decomp("qwen2.5-coder:7b"),
        eval_result=_make_eval(verdict="MATCH"),
    )
    rows = model_scoreboard(conn)
    by_model = {r["model"]: r for r in rows}
    assert by_model["gemma2:27b"]["attempts"] == 4
    assert by_model["gemma2:27b"]["matches"] == 2
    assert by_model["gemma2:27b"]["fix"] == 1
    assert by_model["qwen2.5-coder:7b"]["matches"] == 1


# --------------------------------------------------------------------------- #
# CLI scoreboard
# --------------------------------------------------------------------------- #

def test_cli_scoreboard(tmp_path, capsys):
    db = tmp_path / "r.sqlite"
    conn = open_db(db)
    log_attempt(
        conn,
        decomp_result=_make_decomp("gemma2:27b"),
        eval_result=_make_eval(verdict="MATCH"),
    )
    conn.close()
    from phase_b.cli import main as cli_main
    rc = cli_main(["scoreboard", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemma2:27b" in out
