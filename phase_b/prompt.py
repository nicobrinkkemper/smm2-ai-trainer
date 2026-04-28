"""Format a ContextBundle into a model-friendly prompt.

Two flavours:

- ``build_prompt`` — single string suitable for completion-style endpoints
  (Ollama ``/api/generate`` with a ``system`` field, llama.cpp).
- ``build_chatml`` — list of ``{role, content}`` dicts for chat-style
  endpoints. Handy for fine-tuned models trained on the existing
  ``dataset_v3_chatml.jsonl`` format.

Both share the same body assembly so behaviour stays consistent.
"""

from __future__ import annotations

from typing import Iterable

from .context import ContextBundle


SYSTEM_PROMPT = (
    "You are an expert C++ AArch64 decompilation assistant for the "
    "Super Mario Maker 2 reverse-engineering project. The target is byte-"
    "identical Clang 8 output with -O2 and -mllvm -enable-post-misched=false. "
    "Output ONLY the C++ function body — no markdown fences, no commentary. "
    "Use volatile, inline asm, and the project's STACK_CMP_* macros only when "
    "the gotchas list says they are necessary."
)

_MAX_DISASM_LINES = 256
_MAX_CALLERS = 8
_MAX_CALLEES = 8


def _format_disasm(disasm: list[dict]) -> str:
    lines = []
    for ins in disasm[:_MAX_DISASM_LINES]:
        offset = int(ins["offset"])
        lines.append(
            f"{offset:>4x}: {ins['mnemonic']:<8} {ins['op_str']}".rstrip()
        )
    if len(disasm) > _MAX_DISASM_LINES:
        lines.append(f"... ({len(disasm) - _MAX_DISASM_LINES} more)")
    return "\n".join(lines)


def _format_calls(rows: Iterable[dict], limit: int) -> str:
    out = []
    for r in list(rows)[:limit]:
        out.append(
            f"- `{r['function_id']}` `{r['quality']}` "
            f"{r['kind']}×{r['site_count']} {r['name']}"
        )
    return "\n".join(out) if out else "_(none)_"


def _format_features(features: dict) -> str:
    flags = sorted(k for k, v in features.items() if v)
    return ", ".join(flags) if flags else "_(none)_"


def _format_gotchas(gotchas: list[dict]) -> str:
    if not gotchas:
        return ""
    return "\n\n".join(
        f"### {g['title']}\n{g['body']}" for g in gotchas
    )


def _format_struct_hints(hints: list[dict]) -> str:
    if not hints:
        return ""
    out = []
    for h in hints:
        out.append(
            f"- `{h.get('type_name', '?')}` `+{int(h['offset']):#x}` "
            f"`{h['name']}: {h.get('c_type', '?')}` "
            f"(conf {float(h.get('confidence', 1.0)):.2f})"
        )
    return "\n".join(out)


def _build_user_body(ctx: ContextBundle) -> str:
    f = ctx.raw["function"]
    parts = [
        f"# Decompile `{f['name']}`",
        "",
        f"- Address: `{f['id']}`",
        f"- Size:    {f['size']} bytes",
        f"- Quality: `{f['quality']}` (target: `O`, byte-match)",
        "",
        "## Disassembly",
        "```asm",
        _format_disasm(ctx.disasm),
        "```",
        "",
        "## Features",
        _format_features(ctx.features),
        "",
        "## Callees",
        _format_calls(ctx.callees, _MAX_CALLEES),
        "",
        "## Callers",
        _format_calls(ctx.callers, _MAX_CALLERS),
    ]
    hints = _format_struct_hints(ctx.struct_hints)
    if hints:
        parts += ["", "## Known struct fields", hints]
    gotchas = _format_gotchas(ctx.gotchas)
    if gotchas:
        parts += ["", "## Compiler gotchas (selected)", gotchas]
    parts += [
        "",
        "## Output",
        "Return the C++ source for this single function and nothing else.",
    ]
    return "\n".join(parts)


def build_prompt(ctx: ContextBundle) -> dict:
    """Return ``{"system": ..., "prompt": ...}`` for Ollama-style endpoints."""
    return {"system": SYSTEM_PROMPT, "prompt": _build_user_body(ctx)}


def build_chatml(ctx: ContextBundle) -> list[dict]:
    """Return ChatML-style messages for chat endpoints."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_body(ctx)},
    ]
