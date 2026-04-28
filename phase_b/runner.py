"""Top-level "decompile this function" entry point.

Pulls the FKB context bundle, builds a prompt, sends to a model, cleans
the response, writes the candidate C++ to disk. Returns a ``DecompResult``
recording everything the evaluator (Task 2) will need.

This is *one shot* — no compile, no diff, no iteration. Those live in
sibling modules (``evaluate``, ``iterate``).
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from .context import ContextBundle, fetch_context
from .ollama import OllamaClient, OllamaResponse
from .prompt import build_prompt


_FENCE_RE = re.compile(r"^```(?:cpp|c\+\+|c)?\s*$", re.MULTILINE)
_CLOSE_FENCE_RE = re.compile(r"^```\s*$", re.MULTILINE)


def strip_markdown_fences(text: str) -> str:
    """Remove ```cpp / ``` fences if the model wrapped its output."""
    text = _FENCE_RE.sub("", text)
    text = _CLOSE_FENCE_RE.sub("", text)
    return text.strip()


class _ModelClient(Protocol):
    def generate(
        self, *, model: str, prompt: str, system: str | None = None,
        temperature: float = ..., num_predict: int = ...,
    ) -> OllamaResponse: ...


@dataclass
class DecompResult:
    function_id: str
    function_name: str
    quality: str
    model: str
    prompt_chars: int
    response_chars: int
    candidate_cpp: str
    artifact_path: Optional[Path]
    started_at: str
    finished_at: str
    tokens_prompt: Optional[int] = None
    tokens_eval: Optional[int] = None
    duration_ns: Optional[int] = None
    error: Optional[str] = None
    raw_response: str = ""

    def to_provenance(self) -> dict:
        d = asdict(self)
        d["artifact_path"] = str(self.artifact_path) if self.artifact_path else None
        return d


def decompile_one(
    target: str,
    *,
    model: str,
    client: Optional[_ModelClient] = None,
    decomp_repo: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    temperature: float = 0.1,
    num_predict: int = 4096,
    db_path: Optional[Path] = None,
) -> DecompResult:
    """Run one decomp attempt against ``target`` and persist the result.

    ``out_dir`` defaults to ``<smm2-decomp>/scratch/phase_b``. Each call
    writes ``<func>__<model>__<ts>.cpp`` plus a ``.json`` provenance
    sidecar.
    """
    started = datetime.datetime.now(datetime.timezone.utc)
    ctx = fetch_context(target, decomp_repo=decomp_repo, db_path=db_path)
    p = build_prompt(ctx)

    if client is None:
        client = OllamaClient()

    err: Optional[str] = None
    candidate = ""
    raw = ""
    resp: Optional[OllamaResponse] = None
    try:
        resp = client.generate(
            model=model,
            prompt=p["prompt"],
            system=p["system"],
            temperature=temperature,
            num_predict=num_predict,
        )
        raw = resp.text
        candidate = strip_markdown_fences(raw)
    except Exception as e:  # noqa: BLE001 — we log everything
        err = f"{type(e).__name__}: {e}"

    finished = datetime.datetime.now(datetime.timezone.utc)

    artifact_path = _write_artifacts(
        ctx, model, candidate, raw, started, out_dir, decomp_repo
    )

    return DecompResult(
        function_id=ctx.function_id,
        function_name=ctx.function_name,
        quality=ctx.quality,
        model=model,
        prompt_chars=len(p["prompt"]),
        response_chars=len(raw),
        candidate_cpp=candidate,
        artifact_path=artifact_path,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        tokens_prompt=getattr(resp, "prompt_eval_count", None) if resp else None,
        tokens_eval=getattr(resp, "eval_count", None) if resp else None,
        duration_ns=getattr(resp, "total_duration_ns", None) if resp else None,
        error=err,
        raw_response=raw,
    )


def _write_artifacts(
    ctx: ContextBundle,
    model: str,
    candidate: str,
    raw: str,
    started: datetime.datetime,
    out_dir: Optional[Path],
    decomp_repo: Optional[Path],
) -> Optional[Path]:
    if not candidate and not raw:
        return None

    if out_dir is None:
        repo = decomp_repo or Path.home() / "code" / "smm2-decomp"
        out_dir = Path(repo) / "scratch" / "phase_b"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", ctx.function_name)[:80] or "anon"
    safe_model = re.sub(r"[^A-Za-z0-9_.]", "_", model)
    ts = started.strftime("%Y%m%dT%H%M%SZ")
    base = out_dir / f"{safe_name}__{safe_model}__{ts}"
    cpp_path = base.with_suffix(".cpp")
    cpp_path.write_text(candidate or raw)

    sidecar = {
        "function_id": ctx.function_id,
        "function_name": ctx.function_name,
        "model": model,
        "started_at": started.isoformat(timespec="seconds"),
        "raw_response": raw,
    }
    base.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    return cpp_path
