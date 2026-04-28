"""Tests for the Phase B integration shim.

We don't depend on a running Ollama server or a populated FKB; both are
stubbed so the test suite is fast, hermetic, and works in CI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from phase_b import context as phase_b_context
from phase_b import runner as phase_b_runner
from phase_b import cli as phase_b_cli
from phase_b.context import ContextBundle, FetchError, fetch_context
from phase_b.prompt import build_prompt, build_chatml, SYSTEM_PROMPT
from phase_b.ollama import OllamaResponse
from phase_b.runner import (
    DecompResult,
    decompile_one,
    strip_markdown_fences,
)
from phase_b.cli import main as cli_main


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

SAMPLE_BUNDLE = {
    "function": {
        "id": "0x00000071008b92c0",
        "addr": 0x71008B92C0,
        "size": 84,
        "name": "_ZN2Lp3Utl12StateMachine5resetEv",
        "quality": "W",
        "attempts": 0,
    },
    "callers": [
        {
            "function_id": "0x00000071017774a0",
            "name": "sub_71017774A0",
            "quality": "A",
            "kind": "call",
            "site_count": 27,
        }
    ],
    "callees": [],
    "neighbours": [
        {
            "function_id": "0x00000071008b9240",
            "addr": 0x71008B9240,
            "name": "_ZN2Lp3Utl12StateMachine10clearStateEv",
            "quality": "W",
            "delta": -128,
        }
    ],
    "disasm": [
        {"offset": 0, "bytes_hex": "08094090", "mnemonic": "ldp",
         "op_str": "w8, w9, [x0, #8]"},
        {"offset": 4, "bytes_hex": "1ff0029e", "mnemonic": "stp",
         "op_str": "wzr, w8, [x0, #0xc]"},
    ],
    "features": {
        "has_csel": True, "has_ccmp": True, "has_b": True,
        "has_bl": False, "is_leaf": True, "has_pre_index_load": False,
    },
    "gotchas": [
        {
            "title": "Critical Build Flags",
            "body": "-O2 -mllvm -enable-post-misched=false …",
        }
    ],
    "struct_hints": [],
}


class StubClient:
    """Pluggable replacement for OllamaClient used in tests."""

    def __init__(self, response_text: str = "void reset() {}\n", *,
                 prompt_eval: int = 100, eval_count: int = 50,
                 raise_exc: Exception | None = None) -> None:
        self.response_text = response_text
        self.prompt_eval = prompt_eval
        self.eval_count = eval_count
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def generate(self, *, model, prompt, system=None,
                 temperature=0.1, num_predict=4096):
        self.calls.append({
            "model": model, "prompt_chars": len(prompt),
            "system_chars": len(system or ""),
            "temperature": temperature, "num_predict": num_predict,
        })
        if self.raise_exc:
            raise self.raise_exc
        return OllamaResponse(
            text=self.response_text,
            prompt_eval_count=self.prompt_eval,
            eval_count=self.eval_count,
            total_duration_ns=int(1.5e9),
        )


@pytest.fixture
def stub_fetch(monkeypatch):
    def _stub(target, **kwargs):
        return ContextBundle(raw=SAMPLE_BUNDLE)
    monkeypatch.setattr(phase_b_context, "fetch_context", _stub)
    monkeypatch.setattr(phase_b_runner, "fetch_context", _stub)
    monkeypatch.setattr(phase_b_cli, "fetch_context", _stub)
    return _stub


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #

def test_context_bundle_accessors():
    b = ContextBundle(raw=SAMPLE_BUNDLE)
    assert b.function_name.endswith("resetEv")
    assert b.function_size == 84
    assert b.quality == "W"
    assert b.features["has_csel"] is True
    assert b.callers[0]["site_count"] == 27


def test_fetch_context_subprocess_failure(monkeypatch, tmp_path):
    """When the FKB CLI exits non-zero we raise FetchError with the stderr."""
    def _stub_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        )
    monkeypatch.setattr(subprocess, "run", _stub_run)
    monkeypatch.setattr(phase_b_context, "DEFAULT_DECOMP_REPO", tmp_path)
    with pytest.raises(FetchError, match="boom"):
        fetch_context("does_not_matter")


def test_fetch_context_invalid_json(monkeypatch, tmp_path):
    def _stub_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="not json", stderr=""
        )
    monkeypatch.setattr(subprocess, "run", _stub_run)
    monkeypatch.setattr(phase_b_context, "DEFAULT_DECOMP_REPO", tmp_path)
    with pytest.raises(FetchError, match="non-JSON"):
        fetch_context("x")


def test_fetch_context_missing_repo(tmp_path):
    bad = tmp_path / "nope"
    with pytest.raises(FetchError, match="not found"):
        fetch_context("x", decomp_repo=bad)


def test_fetch_context_happy_path(monkeypatch, tmp_path):
    def _stub_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=json.dumps(SAMPLE_BUNDLE), stderr=""
        )
    monkeypatch.setattr(subprocess, "run", _stub_run)
    monkeypatch.setattr(phase_b_context, "DEFAULT_DECOMP_REPO", tmp_path)
    bundle = fetch_context("anything")
    assert bundle.function_name.endswith("resetEv")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

def test_build_prompt_returns_system_and_prompt():
    p = build_prompt(ContextBundle(raw=SAMPLE_BUNDLE))
    assert "system" in p
    assert "prompt" in p
    assert p["system"] == SYSTEM_PROMPT
    assert "_ZN2Lp3Utl12StateMachine5resetEv" in p["prompt"]
    assert "Disassembly" in p["prompt"]
    assert "ldp" in p["prompt"]
    assert "Critical Build Flags" in p["prompt"]


def test_build_prompt_features_listed():
    p = build_prompt(ContextBundle(raw=SAMPLE_BUNDLE))
    # Feature flags rendered as comma-list of true ones.
    assert "has_csel" in p["prompt"]
    assert "is_leaf" in p["prompt"]
    # False flags should NOT appear
    assert "has_bl," not in p["prompt"] and not p["prompt"].endswith("has_bl")


def test_build_chatml_two_messages():
    msgs = build_chatml(ContextBundle(raw=SAMPLE_BUNDLE))
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Decompile" in msgs[1]["content"]


def test_build_prompt_handles_empty_callers_callees():
    bundle = dict(SAMPLE_BUNDLE)
    bundle["callers"] = []
    bundle["callees"] = []
    p = build_prompt(ContextBundle(raw=bundle))
    assert "_(none)_" in p["prompt"]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def test_strip_markdown_fences_removes_cpp_block():
    raw = "```cpp\nvoid foo() {}\n```"
    assert strip_markdown_fences(raw) == "void foo() {}"


def test_strip_markdown_fences_handles_unfenced():
    assert strip_markdown_fences("void foo() {}") == "void foo() {}"


def test_strip_markdown_fences_handles_partial():
    raw = "```c\nvoid foo() {}\n"
    assert "void foo()" in strip_markdown_fences(raw)


def test_decompile_one_writes_artifacts(stub_fetch, tmp_path):
    client = StubClient(response_text="```cpp\nvoid reset() {}\n```")
    res = decompile_one(
        "0x71008b92c0", model="gemma2:27b",
        client=client, out_dir=tmp_path,
    )
    assert isinstance(res, DecompResult)
    assert res.error is None
    assert res.candidate_cpp == "void reset() {}"
    assert res.tokens_prompt == 100
    assert res.tokens_eval == 50
    assert res.artifact_path is not None
    assert res.artifact_path.exists()
    assert res.artifact_path.read_text() == "void reset() {}"
    sidecar = res.artifact_path.with_suffix(".json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["function_id"] == "0x00000071008b92c0"
    assert payload["model"] == "gemma2:27b"


def test_decompile_one_records_error_on_client_failure(
    stub_fetch, tmp_path
):
    client = StubClient(raise_exc=RuntimeError("ollama down"))
    res = decompile_one(
        "0x71008b92c0", model="gemma2:27b",
        client=client, out_dir=tmp_path,
    )
    assert res.error is not None
    assert "ollama down" in res.error
    assert res.candidate_cpp == ""
    # No artifact should be written when nothing came back at all.
    assert res.artifact_path is None


def test_decompile_one_passes_correct_options(stub_fetch, tmp_path):
    client = StubClient()
    decompile_one(
        "0x71008b92c0", model="qwen2.5-coder:7b",
        client=client, out_dir=tmp_path,
        temperature=0.5, num_predict=2048,
    )
    assert client.calls[0]["model"] == "qwen2.5-coder:7b"
    assert client.calls[0]["temperature"] == 0.5
    assert client.calls[0]["num_predict"] == 2048
    assert client.calls[0]["system_chars"] > 0


def test_decompile_one_safe_filename(stub_fetch, tmp_path):
    """Function names with C++ mangling chars must not break the filename."""
    client = StubClient(response_text="void x() {}")
    res = decompile_one(
        "_ZN2Lp3Utl12StateMachine5resetEv", model="gemma2:27b",
        client=client, out_dir=tmp_path,
    )
    name = res.artifact_path.name
    # No path-traversal or shell-significant characters in the filename.
    for bad in "/\\:; \t":
        assert bad not in name
    assert "_ZN2Lp3Utl12StateMachine5resetEv" in name


def test_to_provenance_is_json_serialisable(stub_fetch, tmp_path):
    client = StubClient()
    res = decompile_one(
        "0x71008b92c0", model="gemma2:27b",
        client=client, out_dir=tmp_path,
    )
    s = json.dumps(res.to_provenance())
    assert "gemma2:27b" in s


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #

def test_cli_prompt_subcommand(stub_fetch, capsys):
    rc = cli_main(["prompt", "0x71008b92c0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SYSTEM" in out
    assert "PROMPT" in out
    assert "_ZN2Lp3Utl12StateMachine5resetEv" in out


def test_cli_prompt_json_subcommand(stub_fetch, capsys):
    rc = cli_main(["prompt", "0x71008b92c0", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["system"] == SYSTEM_PROMPT
    assert "Disassembly" in payload["prompt"]


def test_cli_decompile_uses_stub_client(stub_fetch, tmp_path, capsys, monkeypatch):
    """The CLI must successfully run end-to-end with a stub OllamaClient."""
    client = StubClient(response_text="void x() {}")
    monkeypatch.setattr(
        "phase_b.runner.OllamaClient",
        lambda *a, **kw: client,
    )
    rc = cli_main([
        "decompile", "0x71008b92c0",
        "--model", "gemma2:27b",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemma2:27b" in out
    assert "artifact" in out


def test_cli_decompile_returns_2_on_model_error(
    stub_fetch, tmp_path, capsys, monkeypatch
):
    client = StubClient(raise_exc=RuntimeError("network down"))
    monkeypatch.setattr(
        "phase_b.runner.OllamaClient",
        lambda *a, **kw: client,
    )
    rc = cli_main([
        "decompile", "0x71008b92c0",
        "--model", "gemma2:27b",
        "--out", str(tmp_path),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "network down" in err
