"""Compile a candidate C++ snippet and structurally diff it against the
original bytes in ``main.elf``.

The evaluator is intentionally split from the runner so the iteration
controller (Task 4) can call it after a model produces a candidate, after
a permuter mutation, or after a human edit — all on the same code path.

Build mechanics
---------------
We treat ``smm2-decomp/src/<some path>.cpp`` as the placement target. If
the FKB tells us the function already has a source file (any quality W
or O), we **overwrite that file** for the attempt and restore it from
``HEAD`` afterwards. For functions with no source yet we drop a new file
under ``smm2-decomp/src/auto/_phase_b_<safe_name>.cpp`` so CMake's
auto-glob picks it up; that file is removed at end-of-attempt.

Diff mechanics
--------------
After a successful build, we read the matching symbol's bytes from
``build/Slope`` (via ``llvm-nm`` to locate, ``readelf -S`` to map VA →
file offset), read the original bytes from ``data/v3.0.3/main.elf``,
and run them through ``tools.diff_classify.build_diff``. The resulting
``StructuredDiff`` is what the SQLite ledger records.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence


# We import diff_classify lazily so this module is testable without
# capstone installed (e.g. the unit-test workhorse mocks the diff call).
def _import_diff_classify():
    from importlib import util as importlib_util
    repo = Path(os.environ.get(
        "SMM2_DECOMP_REPO", str(Path.home() / "code" / "smm2-decomp")
    ))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.diff_classify import disassemble, build_diff  # noqa: E402
    return disassemble, build_diff


def _build_diff_context(repo: Path, slope_elf: Path):
    """Build a DiffContext with FKB + nm resolvers so build_diff can
    distinguish "wrong target symbol" from "same target, different layout".
    Returns ``None`` on any failure — the caller falls back to bytes-only
    classification (the pre-resolver behaviour)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from tools.diff_classify import (  # noqa: E402
            DiffContext, fkb_resolver, nm_resolver,
        )
        import sqlite3
    except Exception:
        return None
    fkb_path = repo / "data" / "v3.0.3" / "fkb.sqlite"
    if not fkb_path.exists() or not slope_elf.exists():
        return None
    try:
        conn = sqlite3.connect(str(fkb_path))
        conn.row_factory = sqlite3.Row
        return DiffContext(
            expected_resolver=fkb_resolver(conn),
            actual_resolver=nm_resolver(slope_elf),
        )
    except Exception:
        return None


BASE_ADDR = 0x7100000000
TEXT_OFFSET_MAIN_ELF = 0x888


class EvaluationError(RuntimeError):
    pass


@dataclass
class EvaluationResult:
    function_id: str
    function_name: str
    verdict: str                       # MATCH | ACCEPT_W | FIX | ESCALATE | BUILD_FAIL | RESOLVE_FAIL
    severity: str                      # NONE | COSMETIC | PATTERN | LOGICAL
    summary: dict[str, int] = field(default_factory=dict)
    build_ok: bool = False
    build_error: Optional[str] = None
    expected_size: int = 0
    actual_size: int = 0
    edits_json: str = "[]"             # full per-edit details for debugging
    candidate_path: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Build placement / restoration
# --------------------------------------------------------------------------- #

def _find_existing_source(repo: Path, mangled_name: str) -> Optional[Path]:
    """Grep src/ for a definition or W-stub of the function. We can't rely
    on the build to tell us — viking checks the binary, not source maps."""
    src = repo / "src"
    if not src.is_dir():
        return None
    grep_for = [mangled_name, mangled_name.replace("_Z", "_Z")]
    try:
        proc = subprocess.run(
            ["grep", "-rln", "--include=*.cpp", "-e", mangled_name, str(src)],
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return None
    for line in proc.stdout.splitlines():
        p = Path(line.strip())
        if p.is_file():
            return p
    return None


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)[:80] or "anon"


@dataclass
class _Placement:
    path: Path
    pre_existing: bool
    saved_contents: Optional[str] = None  # for git-checkout-style restore


def place_candidate(
    repo: Path, function_name: str, candidate_cpp: str
) -> _Placement:
    existing = _find_existing_source(repo, function_name)
    if existing is not None:
        saved = existing.read_text()
        existing.write_text(candidate_cpp)
        return _Placement(path=existing, pre_existing=True, saved_contents=saved)
    # No existing source — drop a fresh file under src/auto/
    target = repo / "src" / "auto" / f"_phase_b_{_safe_filename(function_name)}.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(candidate_cpp)
    return _Placement(path=target, pre_existing=False)


def restore_placement(p: _Placement) -> None:
    if p.pre_existing:
        if p.saved_contents is not None:
            p.path.write_text(p.saved_contents)
    else:
        try:
            p.path.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def run_build(repo: Path, *, timeout: float = 300.0) -> tuple[bool, str]:
    """Run ``ninja -C build`` in the smm2-decomp checkout. Returns
    ``(ok, combined_output)``."""
    build_dir = repo / "build"
    if not build_dir.is_dir():
        return False, f"build dir {build_dir} not found — run setup.py first"
    try:
        proc = subprocess.run(
            ["ninja", "-C", str(build_dir)],
            capture_output=True, text=True, timeout=timeout, cwd=str(repo),
        )
    except subprocess.TimeoutExpired as e:
        return False, f"ninja timeout after {timeout}s: {e}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


# --------------------------------------------------------------------------- #
# ELF reading
# --------------------------------------------------------------------------- #

def find_symbol_addr(slope_elf: Path, mangled_name: str) -> Optional[int]:
    proc = subprocess.run(
        ["llvm-nm", str(slope_elf)],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in {"T", "t", "W", "w"} and parts[2] == mangled_name:
            return int(parts[0], 16)
    return None


def find_text_section(slope_elf: Path) -> Optional[tuple[int, int]]:
    """Return (text_va, text_off) for the .text section."""
    proc = subprocess.run(
        ["readelf", "-S", str(slope_elf)],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if " .text" in line and "PROGBITS" in line:
            hexes = [
                w for w in line.split()
                if all(c in "0123456789abcdefABCDEF" for c in w) and len(w) >= 8
            ]
            if len(hexes) >= 2:
                return int(hexes[0], 16), int(hexes[1], 16)
    return None


def read_actual_bytes(
    slope_elf: Path, mangled_name: str, expected_size: int
) -> Optional[bytes]:
    sym_addr = find_symbol_addr(slope_elf, mangled_name)
    if sym_addr is None:
        return None
    sect = find_text_section(slope_elf)
    if sect is None:
        return None
    text_va, text_off = sect
    file_off = sym_addr - text_va + text_off
    data = slope_elf.read_bytes()
    if file_off < 0 or file_off + expected_size > len(data):
        return None
    return data[file_off : file_off + expected_size]


def read_expected_bytes(main_elf: Path, addr: int, size: int) -> Optional[bytes]:
    file_off = addr - BASE_ADDR + TEXT_OFFSET_MAIN_ELF
    data = main_elf.read_bytes()
    if file_off < 0 or file_off + size > len(data):
        return None
    return data[file_off : file_off + size]


# --------------------------------------------------------------------------- #
# Top-level evaluate
# --------------------------------------------------------------------------- #

def evaluate(
    function_id: str,
    function_name: str,
    expected_addr: int,
    expected_size: int,
    candidate_cpp: str,
    *,
    decomp_repo: Optional[Path] = None,
    skip_build: bool = False,
    build_timeout: float = 300.0,
) -> EvaluationResult:
    """Place the candidate, build, diff. Always restores the source tree."""
    repo = (decomp_repo or Path(
        os.environ.get("SMM2_DECOMP_REPO", str(Path.home() / "code" / "smm2-decomp"))
    )).resolve()
    started = datetime.datetime.now(datetime.timezone.utc)

    placement = place_candidate(repo, function_name, candidate_cpp)
    candidate_path = str(placement.path.relative_to(repo))

    res = EvaluationResult(
        function_id=function_id,
        function_name=function_name,
        verdict="ESCALATE",
        severity="LOGICAL",
        candidate_path=candidate_path,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=started.isoformat(timespec="seconds"),
        expected_size=expected_size,
    )

    try:
        if skip_build:
            res.build_ok = True
        else:
            ok, output = run_build(repo, timeout=build_timeout)
            res.build_ok = ok
            if not ok:
                res.verdict = "BUILD_FAIL"
                res.severity = "LOGICAL"
                res.build_error = output[-2000:]  # tail
                return res

        slope_elf = repo / "build" / "Slope"
        main_elf = repo / "data" / "v3.0.3" / "main.elf"
        actual = read_actual_bytes(slope_elf, function_name, expected_size)
        if actual is None:
            res.verdict = "RESOLVE_FAIL"
            res.severity = "LOGICAL"
            res.build_error = (
                f"could not locate {function_name!r} in {slope_elf} "
                f"(symbol or .text section missing)"
            )
            return res

        expected = read_expected_bytes(main_elf, expected_addr, expected_size)
        if expected is None:
            res.verdict = "RESOLVE_FAIL"
            res.severity = "LOGICAL"
            res.build_error = (
                f"could not locate expected bytes at "
                f"0x{expected_addr:x}+{expected_size}"
            )
            return res

        res.actual_size = len(actual)
        disassemble, build_diff = _import_diff_classify()
        e_ins = disassemble(expected, base_addr=expected_addr)
        # Use the build/Slope absolute address so the actual_resolver
        # (nm-based) can find the right symbols at the right addresses.
        sym_addr = find_symbol_addr(slope_elf, function_name) or 0
        a_ins = disassemble(actual, base_addr=sym_addr)
        diff_ctx = _build_diff_context(repo, slope_elf)
        diff = build_diff(e_ins, a_ins, name=function_name, ctx=diff_ctx)

        res.verdict = diff.verdict()
        res.severity = diff.severity.name
        res.summary = {cls.value: n for cls, n in diff.summary().items()}
        edits_data = []
        for ed in diff.edits:
            if ed.is_equal:
                continue
            edits_data.append({
                "kind": ed.kind,
                "classification": ed.classification.value,
                "severity": ed.severity.name,
                "note": ed.note,
                "fix_recipe": ed.fix_recipe,
                "expected": [i.text for i in ed.expected],
                "actual":   [i.text for i in ed.actual],
            })
        res.edits_json = json.dumps(edits_data)
        res.finished_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds")
        return res
    finally:
        restore_placement(placement)
