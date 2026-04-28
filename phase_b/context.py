"""Pull a function context bundle from smm2-decomp's FKB CLI.

The bundle is the JSON document produced by

    python3 -m tools.fkb.cli context <target> --json

run inside ``smm2-decomp``. Phase B never reads the FKB SQLite directly —
the CLI is the stable, tested boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DECOMP_REPO = Path(
    os.environ.get("SMM2_DECOMP_REPO", str(Path.home() / "code" / "smm2-decomp"))
).resolve()


class FetchError(RuntimeError):
    """Raised when the FKB CLI fails or returns malformed JSON."""


@dataclass
class ContextBundle:
    """Lightweight wrapper around the JSON dict so callers don't sprinkle
    ``ctx["function"]["name"]`` strings everywhere."""

    raw: dict

    @property
    def function_id(self) -> str:
        return self.raw["function"]["id"]

    @property
    def function_addr(self) -> int:
        return int(self.raw["function"]["addr"])

    @property
    def function_size(self) -> int:
        return int(self.raw["function"]["size"])

    @property
    def function_name(self) -> str:
        return self.raw["function"]["name"]

    @property
    def quality(self) -> str:
        return self.raw["function"]["quality"]

    @property
    def disasm(self) -> list[dict]:
        return self.raw.get("disasm", [])

    @property
    def callees(self) -> list[dict]:
        return self.raw.get("callees", [])

    @property
    def callers(self) -> list[dict]:
        return self.raw.get("callers", [])

    @property
    def neighbours(self) -> list[dict]:
        return self.raw.get("neighbours", [])

    @property
    def features(self) -> dict:
        return self.raw.get("features", {})

    @property
    def gotchas(self) -> list[dict]:
        return self.raw.get("gotchas", [])

    @property
    def struct_hints(self) -> list[dict]:
        return self.raw.get("struct_hints", [])


def fetch_context(
    target: str,
    *,
    decomp_repo: Optional[Path] = None,
    db_path: Optional[Path] = None,
    elf_path: Optional[Path] = None,
    timeout: float = 30.0,
) -> ContextBundle:
    """Run ``tools.fkb.cli context <target> --json`` in smm2-decomp and parse.

    Raises ``FetchError`` if the process exits non-zero or the output is not
    valid JSON.
    """
    repo = (decomp_repo or DEFAULT_DECOMP_REPO).resolve()
    if not repo.is_dir():
        raise FetchError(f"smm2-decomp repo not found at {repo}")

    cmd = ["python3", "-m", "tools.fkb.cli"]
    if db_path is not None:
        cmd += ["--db", str(db_path)]
    cmd += ["context", target, "--json"]
    if elf_path is not None:
        cmd += ["--elf", str(elf_path)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise FetchError(f"FKB CLI timed out after {timeout}s: {e}") from e

    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise FetchError(
            f"FKB CLI exited {proc.returncode} for {target!r}: {stderr or '<empty>'}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise FetchError(
            f"FKB CLI emitted non-JSON for {target!r}: {e}\n--- stdout ---\n{proc.stdout[:200]}"
        ) from e

    if "function" not in data:
        raise FetchError(
            f"FKB CLI JSON missing 'function' key: keys={sorted(data)}"
        )
    return ContextBundle(raw=data)
