"""Tests for the GGUF export pipeline.

We mock subprocess.run so the tests don't require Unsloth, llama.cpp, or
a GPU. The point is to exercise the orchestration: timeouts surface as
``ExportError``, intermediate skips happen when files exist, llama.cpp
is cloned/built when missing, and ``run_export`` chains the three steps
in the right order with the right arguments.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from phase_b.gguf_export import (
    ExportConfig,
    ExportError,
    convert_hf_to_f16_gguf,
    ensure_llama_cpp,
    merge_lora_to_hf,
    quantize_gguf,
    run_export,
    cli_main,
)


def _cfg(tmp_path: Path, **overrides) -> ExportConfig:
    base = ExportConfig(
        lora_dir=tmp_path / "lora",
        hf_out_dir=tmp_path / "hf",
        gguf_out=tmp_path / "out-Q4_K_M.gguf",
        quant="Q4_K_M",
        llama_cpp_dir=tmp_path / "llama.cpp",
        merge_timeout=10.0,
        convert_timeout=10.0,
        quant_timeout=10.0,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# --------------------------------------------------------------------------- #
# merge_lora_to_hf
# --------------------------------------------------------------------------- #

def test_merge_invokes_subprocess_with_inherit(monkeypatch, tmp_path):
    captured = {}
    def stub_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        captured["kw"] = kw
        Path(cmd[2]).mkdir(parents=True, exist_ok=True)  # cmd[2] is hf_out_dir
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)

    cfg = _cfg(tmp_path)
    merge_lora_to_hf(cfg, max_seq_length=1024)
    assert "lora" in captured["cmd"][2] or str(cfg.lora_dir) in captured["cmd"]
    # We deliberately do NOT pass capture_output=True (would buffer-deadlock).
    assert captured["kw"].get("capture_output") in {None, False}


def test_merge_timeout_raises_export_error(monkeypatch, tmp_path):
    def stub_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1.0))
    monkeypatch.setattr(subprocess, "run", stub_run)
    cfg = _cfg(tmp_path)
    with pytest.raises(ExportError, match="timeout"):
        merge_lora_to_hf(cfg)


def test_merge_failure_raises_export_error(monkeypatch, tmp_path):
    def stub_run(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(subprocess, "run", stub_run)
    cfg = _cfg(tmp_path)
    with pytest.raises(ExportError, match="merge step failed"):
        merge_lora_to_hf(cfg)


def test_merge_skip_when_skip_merge_set(monkeypatch, tmp_path):
    called = {"n": 0}
    def stub_run(*a, **kw):
        called["n"] += 1
        return subprocess.CompletedProcess(a[0] if a else [], 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    cfg = _cfg(tmp_path, skip_merge=True)
    cfg.hf_out_dir.mkdir(parents=True, exist_ok=True)
    merge_lora_to_hf(cfg)
    assert called["n"] == 0


def test_merge_skip_errors_when_hf_dir_missing(tmp_path):
    cfg = _cfg(tmp_path, skip_merge=True)
    with pytest.raises(ExportError, match="does not exist"):
        merge_lora_to_hf(cfg)


# --------------------------------------------------------------------------- #
# ensure_llama_cpp
# --------------------------------------------------------------------------- #

def test_ensure_llama_cpp_clones_and_builds(monkeypatch, tmp_path):
    target = tmp_path / "llama.cpp"
    calls: list[list[str]] = []

    def stub_run(cmd, **kw):
        calls.append(list(cmd))
        # Materialise expected files when the corresponding command runs.
        if cmd[0] == "git":
            target.mkdir(parents=True, exist_ok=True)
            (target / "convert_hf_to_gguf.py").write_text("# stub")
        elif cmd[0] == "cmake" and len(cmd) > 2 and cmd[1] == "-B":
            (target / "build").mkdir(parents=True, exist_ok=True)
        elif cmd[0] == "cmake" and "--build" in cmd:
            (target / "build" / "bin").mkdir(parents=True, exist_ok=True)
            (target / "build" / "bin" / "llama-quantize").write_text("#!/bin/sh\nexit 0\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)

    cfg = _cfg(tmp_path)
    out = ensure_llama_cpp(cfg)
    assert out == target.resolve()
    cmds_starting = [c[0] for c in calls]
    assert "git" in cmds_starting   # cloned
    assert "cmake" in cmds_starting # configured + built


def test_ensure_llama_cpp_skips_clone_when_present(monkeypatch, tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "convert_hf_to_gguf.py").write_text("# stub")
    (target / "build" / "bin").mkdir(parents=True)
    (target / "build" / "bin" / "llama-quantize").write_text("#!/bin/sh\n")

    calls: list[list[str]] = []
    def stub_run(cmd, **kw):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    cfg = _cfg(tmp_path)
    ensure_llama_cpp(cfg)
    assert calls == []  # nothing to do


# --------------------------------------------------------------------------- #
# convert / quantize
# --------------------------------------------------------------------------- #

def test_convert_skips_when_f16_exists(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    f16 = cfg.gguf_out.with_name(cfg.gguf_out.stem.replace("-Q4_K_M","") + "-f16.gguf")
    f16.parent.mkdir(parents=True, exist_ok=True)
    f16.write_text("ggm")
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "convert_hf_to_gguf.py").write_text("# stub")
    called = {"n": 0}
    def stub_run(cmd, **kw):
        called["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    out = convert_hf_to_f16_gguf(cfg, target)
    assert out == f16
    assert called["n"] == 0


def test_convert_runs_when_missing(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "convert_hf_to_gguf.py").write_text("# stub")
    captured = {}
    def stub_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        Path(cmd[-1]).write_text("gguf")  # produce f16 file
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    out = convert_hf_to_f16_gguf(cfg, target)
    assert out.exists()
    assert "convert_hf_to_gguf.py" in " ".join(captured["cmd"])


def test_convert_timeout_raises(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "convert_hf_to_gguf.py").write_text("# stub")
    def stub_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))
    monkeypatch.setattr(subprocess, "run", stub_run)
    with pytest.raises(ExportError, match="timed out"):
        convert_hf_to_f16_gguf(cfg, target)


def test_quantize_skips_when_final_exists(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.gguf_out.write_text("done")
    target = tmp_path / "llama.cpp"
    (target / "build" / "bin").mkdir(parents=True)
    (target / "build" / "bin" / "llama-quantize").write_text("")
    called = {"n": 0}
    def stub_run(cmd, **kw):
        called["n"] += 1
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    out = quantize_gguf(cfg, target, tmp_path / "f16.gguf")
    assert out == cfg.gguf_out
    assert called["n"] == 0


def test_quantize_timeout_raises(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    target = tmp_path / "llama.cpp"
    (target / "build" / "bin").mkdir(parents=True)
    (target / "build" / "bin" / "llama-quantize").write_text("")
    f16 = tmp_path / "f.gguf"
    f16.write_text("x")
    def stub_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))
    monkeypatch.setattr(subprocess, "run", stub_run)
    with pytest.raises(ExportError, match="timed out"):
        quantize_gguf(cfg, target, f16)


# --------------------------------------------------------------------------- #
# run_export end-to-end (with full subprocess mocking)
# --------------------------------------------------------------------------- #

def test_run_export_chains_three_steps(monkeypatch, tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "convert_hf_to_gguf.py").write_text("# stub")
    (target / "build" / "bin").mkdir(parents=True)
    (target / "build" / "bin" / "llama-quantize").write_text("")

    cfg = _cfg(tmp_path)
    # The merge step's subprocess writes to hf_out_dir (cmd[2])
    def stub_run(cmd, **kw):
        if cmd[0] in {"python3", sys.executable}:
            if "convert_hf_to_gguf.py" in " ".join(cmd):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_text("gguf")
            else:
                # merge step
                Path(cmd[2]).mkdir(parents=True, exist_ok=True)
        elif cmd[0] == str(target / "build" / "bin" / "llama-quantize"):
            Path(cmd[2]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[2]).write_text("final-gguf")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    out = run_export(cfg)
    assert out == cfg.gguf_out
    assert out.exists()


def test_run_export_drops_f16_when_no_keep(monkeypatch, tmp_path):
    target = tmp_path / "llama.cpp"
    target.mkdir()
    (target / "convert_hf_to_gguf.py").write_text("# stub")
    (target / "build" / "bin").mkdir(parents=True)
    (target / "build" / "bin" / "llama-quantize").write_text("")

    cfg = _cfg(tmp_path)
    cfg.keep_f16 = False
    f16_path = cfg.gguf_out.with_name(
        cfg.gguf_out.stem.replace("-Q4_K_M","") + "-f16.gguf"
    )

    def stub_run(cmd, **kw):
        if cmd[0] in {"python3", sys.executable}:
            if "convert_hf_to_gguf.py" in " ".join(cmd):
                f16_path.parent.mkdir(parents=True, exist_ok=True)
                f16_path.write_text("f16")
            else:
                Path(cmd[2]).mkdir(parents=True, exist_ok=True)
        elif cmd[0] == str(target / "build" / "bin" / "llama-quantize"):
            cfg.gguf_out.parent.mkdir(parents=True, exist_ok=True)
            cfg.gguf_out.write_text("final")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", stub_run)
    run_export(cfg)
    assert not f16_path.exists()
    assert cfg.gguf_out.exists()


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #

def test_cli_main_returns_1_on_export_error(monkeypatch, tmp_path, capsys):
    def stub_run(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(subprocess, "run", stub_run)
    rc = cli_main([
        "--lora-dir", str(tmp_path / "lora"),
        "--hf-out-dir", str(tmp_path / "hf"),
        "--gguf-out", str(tmp_path / "out.gguf"),
        "--llama-cpp-dir", str(tmp_path / "llama.cpp"),
        "--merge-timeout", "1",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err
