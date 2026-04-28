"""Tests for the Phase B iteration controller."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import phase_b.iterate as iterate_mod
from phase_b.context import ContextBundle
from phase_b.evaluate import EvaluationResult
from phase_b.iterate import iterate_one
from phase_b.ollama import OllamaResponse


SAMPLE_BUNDLE = {
    "function": {
        "id": "0x71008b92c0", "addr": 0x71008B92C0, "size": 84,
        "name": "_ZN2Lp3Utl12StateMachine5resetEv", "quality": "W", "attempts": 0,
    },
    "callers": [], "callees": [], "neighbours": [], "disasm": [],
    "features": {}, "gotchas": [], "struct_hints": [],
}


class StubClient:
    """Returns one response per call. Cycle through ``responses`` list."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = self.responses.pop(0) if self.responses else ""
        return OllamaResponse(
            text=text,
            prompt_eval_count=100, eval_count=50, total_duration_ns=int(1e9),
        )


def _make_eval(verdict, summary=None, edits=None):
    return EvaluationResult(
        function_id="0x71008b92c0",
        function_name="_ZN2Lp3Utl12StateMachine5resetEv",
        verdict=verdict,
        severity={"MATCH": "NONE", "ACCEPT_W": "COSMETIC",
                  "FIX": "LOGICAL", "ESCALATE": "LOGICAL"}.get(verdict, "LOGICAL"),
        summary=summary or {},
        build_ok=True, expected_size=84, actual_size=84,
        edits_json=str(edits or "[]"),
        candidate_path="src/auto/x.cpp", started_at="t0", finished_at="t1",
    )


@pytest.fixture
def stub_iterate(monkeypatch, tmp_path):
    """Wire stubs around iterate_one: fetch_context returns SAMPLE_BUNDLE,
    evaluate is replaced by a verdict-script. Logging is to a tmp DB."""
    eval_results = []      # populated per-test

    def stub_fetch(target, **kwargs):
        return ContextBundle(raw=SAMPLE_BUNDLE)
    def stub_evaluate(*, function_id, function_name, expected_addr,
                      expected_size, candidate_cpp, **kwargs):
        return eval_results.pop(0)

    monkeypatch.setattr(iterate_mod, "fetch_context", stub_fetch)
    monkeypatch.setattr(iterate_mod, "evaluate", stub_evaluate)
    return eval_results, tmp_path / "r.sqlite"


def test_iterate_succeeds_first_attempt(stub_iterate):
    eval_results, db = stub_iterate
    eval_results.append(_make_eval("MATCH"))
    client = StubClient(["void reset() {}"])
    outcome = iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=5, db_path=db, skip_build=True,
    )
    assert outcome.success
    assert outcome.final_verdict == "MATCH"
    assert outcome.step_count() == 1
    assert "verdict MATCH" in outcome.stop_reason


def test_iterate_runs_until_match(stub_iterate):
    eval_results, db = stub_iterate
    eval_results.extend([_make_eval("FIX"), _make_eval("FIX"), _make_eval("MATCH")])
    client = StubClient([
        "void v1() {}",
        "void v2() {}",
        "void v3() {}",
    ])
    outcome = iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=5, db_path=db, skip_build=True,
    )
    assert outcome.success
    assert outcome.step_count() == 3
    # Subsequent attempts must use iteration prompts (longer than the base).
    assert client.calls[1]["prompt"].count("Iteration feedback") == 1
    assert client.calls[2]["prompt"].count("Iteration feedback") == 1


def test_iterate_accepts_w_as_success(stub_iterate):
    eval_results, db = stub_iterate
    eval_results.append(_make_eval("ACCEPT_W"))
    client = StubClient(["void v() {}"])
    outcome = iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=5, db_path=db, skip_build=True,
    )
    assert outcome.success
    assert outcome.final_verdict == "ACCEPT_W"


def test_iterate_max_attempts_reached(stub_iterate):
    eval_results, db = stub_iterate
    # Each attempt produces a different candidate (no oscillation), all FIX.
    eval_results.extend([_make_eval("FIX") for _ in range(3)])
    client = StubClient(["void v1() {}", "void v2() {}", "void v3() {}"])
    outcome = iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=3, db_path=db, skip_build=True,
    )
    assert not outcome.success
    assert outcome.step_count() == 3
    assert "max_attempts" in outcome.stop_reason


def test_iterate_oscillation_guard(stub_iterate):
    eval_results, db = stub_iterate
    eval_results.extend([_make_eval("FIX"), _make_eval("FIX")])
    # Same candidate twice — should bail on second.
    client = StubClient(["void same() {}", "void same() {}"])
    outcome = iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=5, db_path=db, skip_build=True,
    )
    assert not outcome.success
    assert outcome.step_count() == 2
    assert "oscillation" in outcome.stop_reason


def test_iterate_inference_error_stops(stub_iterate):
    eval_results, db = stub_iterate

    class ErrClient:
        calls = []
        def generate(self, **kw):
            ErrClient.calls.append(kw)
            raise RuntimeError("ollama down")

    outcome = iterate_one(
        "0x71008b92c0", model="m", client=ErrClient(),
        max_attempts=5, db_path=db, skip_build=True,
    )
    assert not outcome.success
    assert "inference error" in outcome.stop_reason
    assert outcome.step_count() == 1


def test_iterate_logs_every_step(stub_iterate):
    """All attempts are written to the SQLite ledger, not just the final."""
    eval_results, db = stub_iterate
    eval_results.extend([_make_eval("FIX"), _make_eval("FIX"), _make_eval("MATCH")])
    client = StubClient(["a", "b", "c"])
    iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=5, db_path=db, skip_build=True,
    )
    # Re-open and count.
    from phase_b.results import open_db, get_attempts
    rows = get_attempts(open_db(db), function_id="0x71008b92c0")
    assert len(rows) == 3


def test_iteration_prompt_includes_diff_summary(stub_iterate):
    """The iteration-feedback prompt actually contains the prior diff."""
    eval_results, db = stub_iterate
    e1 = _make_eval(
        "FIX",
        summary={"constant_different": 1},
        edits='[{"kind":"replace","classification":"constant_different",'
              '"severity":"LOGICAL","note":"x mismatch","fix_recipe":"check literal",'
              '"expected":["mov w8, #1"],"actual":["mov w8, #2"]}]',
    )
    e2 = _make_eval("MATCH")
    eval_results.extend([e1, e2])
    client = StubClient(["void v1() {}", "void v2() {}"])
    iterate_one(
        "0x71008b92c0", model="m", client=client,
        max_attempts=5, db_path=db, skip_build=True,
    )
    # The 2nd prompt must mention the diff details.
    p2 = client.calls[1]["prompt"]
    assert "Iteration feedback" in p2
    assert "constant_different" in p2
    assert "mov w8, #1" in p2 or "x mismatch" in p2
