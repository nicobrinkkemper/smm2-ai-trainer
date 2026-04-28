"""Robust GGUF export with subprocess timeouts and a manual-llama.cpp fallback.

Background (from ``SMM2_AUTODECOMP.md``):

    Unsloth's ``save_pretrained_gguf(quantization_method="q4_k_m")`` has a
    known bug where it deadlocks on large 27B+ models — the embedded
    llama.cpp ``subprocess.run`` overflows its stdout buffer and silently
    waits on a pseudo-sudo prompt. The workaround is to export only the
    merged f16 model from Python and then drive llama.cpp manually for the
    GGUF conversion + quantization step.

This module wraps that workaround into a single re-entrant CLI:

    python3 -m phase_b.gguf_export \\
        --lora-dir   ./lora_model_27b \\
        --hf-out-dir ./smm2-gemma-27b \\
        --gguf-out   ./smm2-gemma-27b-Q4_K_M.gguf \\
        --quant      Q4_K_M \\
        --llama-cpp-dir ./llama.cpp

Each pipeline stage runs in a subprocess with an explicit timeout. If a
stage's output already exists we skip it; runs are therefore resumable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Reasonable defaults; --timeout-merge / --timeout-convert / --timeout-quant
# override on the CLI.
DEFAULT_MERGE_TIMEOUT = 60 * 60      # Unsloth load + merge_and_unload + save
DEFAULT_CONVERT_TIMEOUT = 60 * 60    # llama.cpp convert_hf_to_gguf
DEFAULT_QUANT_TIMEOUT = 30 * 60      # llama.cpp llama-quantize


@dataclass
class ExportConfig:
    lora_dir: Path                  # PEFT/Unsloth checkpoint
    hf_out_dir: Path                # where to write the merged f16 HF model
    gguf_out: Path                  # final quantized GGUF path
    quant: str = "Q4_K_M"           # llama-quantize spec
    llama_cpp_dir: Optional[Path] = None
    f16_gguf: Optional[Path] = None # intermediate; defaults next to gguf_out
    merge_timeout: float = DEFAULT_MERGE_TIMEOUT
    convert_timeout: float = DEFAULT_CONVERT_TIMEOUT
    quant_timeout: float = DEFAULT_QUANT_TIMEOUT
    skip_merge: bool = False        # re-use an existing hf_out_dir as-is
    keep_f16: bool = True


class ExportError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Step 1 — merge LoRA into base, save as f16 HF directory.
# --------------------------------------------------------------------------- #

_MERGE_SCRIPT = """\
import os, sys
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
from unsloth import FastLanguageModel
import torch

lora_dir = sys.argv[1]
out_dir  = sys.argv[2]
max_seq_length = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

print(f'load:  lora_dir={{lora_dir}} max_seq_length={{max_seq_length}}', flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_dir,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
print('save:  merged_16bit', flush=True)
model.save_pretrained_merged(out_dir, tokenizer, save_method='merged_16bit')
print('done', flush=True)
"""


def merge_lora_to_hf(cfg: ExportConfig, *, max_seq_length: int = 1024) -> None:
    """Run a fresh Python subprocess so an Unsloth deadlock can't take down
    the orchestrator. Output streamed to stdout/stderr live (no PIPE that
    would buffer-deadlock)."""
    if cfg.skip_merge:
        if not cfg.hf_out_dir.exists():
            raise ExportError(
                f"--skip-merge set but {cfg.hf_out_dir} does not exist"
            )
        return
    cfg.hf_out_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(_MERGE_SCRIPT)
        script_path = tmp.name

    try:
        try:
            subprocess.run(
                [sys.executable, script_path,
                 str(cfg.lora_dir), str(cfg.hf_out_dir), str(max_seq_length)],
                check=True,
                timeout=cfg.merge_timeout,
                # Inherit stdout/stderr so we don't trigger the buffer-
                # overflow deadlock that's been hitting Unsloth's own export.
            )
        except subprocess.TimeoutExpired as e:
            raise ExportError(
                f"merge step exceeded {cfg.merge_timeout}s timeout — most "
                f"likely the Unsloth/llama.cpp deadlock. Manual fallback: "
                f"run convert_manual.sh after exporting f16 with another "
                f"toolchain. Underlying error: {e}"
            ) from e
        except subprocess.CalledProcessError as e:
            raise ExportError(f"merge step failed: {e}") from e
    finally:
        try:
            os.unlink(script_path)
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Step 2 — llama.cpp convert_hf_to_gguf.py to produce f16 GGUF.
# Step 3 — llama.cpp llama-quantize to produce final quantized GGUF.
# --------------------------------------------------------------------------- #

def ensure_llama_cpp(cfg: ExportConfig) -> Path:
    """Locate (or clone+build) llama.cpp. Returns the directory."""
    target = cfg.llama_cpp_dir
    if target is None:
        target = Path.cwd() / "llama.cpp"
    target = target.resolve()
    if not (target / "convert_hf_to_gguf.py").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", str(target)],
            check=True, timeout=900,
        )
    quantize_bin = target / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        if not (target / "build").exists():
            subprocess.run(
                ["cmake", "-B", "build", "-G", "Ninja"],
                cwd=str(target), check=True, timeout=600,
            )
        subprocess.run(
            ["cmake", "--build", "build", "-j"],
            cwd=str(target), check=True, timeout=3600,
        )
    return target


def convert_hf_to_f16_gguf(cfg: ExportConfig, llama_cpp: Path) -> Path:
    f16 = cfg.f16_gguf or cfg.gguf_out.with_name(
        cfg.gguf_out.stem.replace(f"-{cfg.quant}", "") + "-f16.gguf"
    )
    if f16.exists():
        print(f"skip: {f16} already exists")
        return f16
    f16.parent.mkdir(parents=True, exist_ok=True)
    convert = llama_cpp / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise ExportError(f"missing {convert}")
    try:
        subprocess.run(
            [sys.executable, str(convert), str(cfg.hf_out_dir),
             "--outfile", str(f16)],
            check=True,
            timeout=cfg.convert_timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ExportError(
            f"convert_hf_to_gguf timed out after {cfg.convert_timeout}s"
        ) from e
    except subprocess.CalledProcessError as e:
        raise ExportError(f"convert_hf_to_gguf failed: {e}") from e
    return f16


def quantize_gguf(cfg: ExportConfig, llama_cpp: Path, f16_path: Path) -> Path:
    if cfg.gguf_out.exists():
        print(f"skip: {cfg.gguf_out} already exists")
        return cfg.gguf_out
    quantize = llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize.exists():
        raise ExportError(f"missing {quantize} — was llama.cpp built?")
    cfg.gguf_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [str(quantize), str(f16_path), str(cfg.gguf_out), cfg.quant],
            check=True,
            timeout=cfg.quant_timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise ExportError(
            f"llama-quantize timed out after {cfg.quant_timeout}s"
        ) from e
    except subprocess.CalledProcessError as e:
        raise ExportError(f"llama-quantize failed: {e}") from e
    return cfg.gguf_out


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #

def run_export(cfg: ExportConfig) -> Path:
    merge_lora_to_hf(cfg)
    llama_cpp = ensure_llama_cpp(cfg)
    f16 = convert_hf_to_f16_gguf(cfg, llama_cpp)
    final = quantize_gguf(cfg, llama_cpp, f16)
    if not cfg.keep_f16 and f16 != final and f16.exists():
        f16.unlink()
    return final


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="phase_b gguf_export", description=__doc__)
    p.add_argument("--lora-dir", required=True, type=Path)
    p.add_argument("--hf-out-dir", required=True, type=Path,
                   help="where to write the merged f16 HF model")
    p.add_argument("--gguf-out", required=True, type=Path,
                   help="final quantized GGUF path")
    p.add_argument("--quant", default="Q4_K_M",
                   help="llama-quantize spec (Q4_K_M, Q5_K_M, Q8_0, …)")
    p.add_argument("--llama-cpp-dir", default=None, type=Path,
                   help="path to llama.cpp checkout (clones if missing)")
    p.add_argument("--f16-gguf", default=None, type=Path,
                   help="intermediate f16 GGUF path (defaults to alongside --gguf-out)")
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--merge-timeout",   type=float, default=DEFAULT_MERGE_TIMEOUT)
    p.add_argument("--convert-timeout", type=float, default=DEFAULT_CONVERT_TIMEOUT)
    p.add_argument("--quant-timeout",   type=float, default=DEFAULT_QUANT_TIMEOUT)
    p.add_argument("--skip-merge", action="store_true",
                   help="re-use existing --hf-out-dir as-is (don't run Unsloth)")
    p.add_argument("--no-keep-f16", action="store_true",
                   help="delete the intermediate f16 GGUF on success")
    args = p.parse_args(argv)

    cfg = ExportConfig(
        lora_dir=args.lora_dir.resolve(),
        hf_out_dir=args.hf_out_dir.resolve(),
        gguf_out=args.gguf_out.resolve(),
        quant=args.quant,
        llama_cpp_dir=args.llama_cpp_dir.resolve() if args.llama_cpp_dir else None,
        f16_gguf=args.f16_gguf.resolve() if args.f16_gguf else None,
        merge_timeout=args.merge_timeout,
        convert_timeout=args.convert_timeout,
        quant_timeout=args.quant_timeout,
        skip_merge=args.skip_merge,
        keep_f16=not args.no_keep_f16,
    )
    try:
        path = run_export(cfg)
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"\nexported: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
