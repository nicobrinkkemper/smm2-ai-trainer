"""Iteration controller — the permuter-style autonomous loop.

For one function in ``FIX``/``ESCALATE`` state, drives a budgeted
attempt loop: each round feeds the previous diff back to the model so
it can make targeted edits. Stops when:

- verdict is ``MATCH`` or ``ACCEPT_W`` (success)
- the budget (``max_attempts`` and/or ``max_seconds``) is exhausted
- the candidate stops changing (oscillation guard)

This is the loop that lets a "not so good" pretrained Gemma compensate
for individual weak guesses with iteration.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .context import fetch_context
from .evaluate import EvaluationResult, evaluate
from .ollama import OllamaClient
from .prompt import build_prompt, build_iteration_prompt
from .results import open_db, log_attempt
from .runner import (
    DecompResult,
    decompile_one,
    strip_markdown_fences,
)


_TERMINAL_VERDICTS = {"MATCH", "ACCEPT_W"}


@dataclass
class IterationStep:
    attempt_num: int
    decomp: DecompResult
    eval: Optional[EvaluationResult]
    duration_s: float


@dataclass
class IterationOutcome:
    function_id: str
    function_name: str
    model: str
    final_verdict: Optional[str]
    success: bool
    steps: list[IterationStep] = field(default_factory=list)
    stop_reason: str = ""

    def step_count(self) -> int:
        return len(self.steps)


def _hash_candidate(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def iterate_one(
    target: str,
    *,
    model: str,
    max_attempts: int = 5,
    max_seconds: float = 600.0,
    client: Optional[OllamaClient] = None,
    decomp_repo: Optional[Path] = None,
    db_path: Optional[Path] = None,
    skip_build: bool = False,
    temperature: float = 0.3,
    num_predict: int = 4096,
    log: bool = True,
) -> IterationOutcome:
    """Run up to ``max_attempts`` rounds of decomp+evaluate with feedback.

    Each step is logged to the SQLite ledger so retrospectives can see
    *every* attempt, not just the final one. Temperature defaults to 0.3
    (vs 0.1 for one-shots) so subsequent attempts have actual variability
    to escape stuck states.
    """
    if client is None:
        client = OllamaClient()
    started = time.monotonic()
    ctx = fetch_context(target, decomp_repo=decomp_repo)
    outcome = IterationOutcome(
        function_id=ctx.function_id,
        function_name=ctx.function_name,
        model=model,
        final_verdict=None,
        success=False,
    )

    prior_candidate: Optional[str] = None
    prior_eval: Optional[EvaluationResult] = None
    seen_hashes: set[str] = set()

    for attempt in range(1, max_attempts + 1):
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            outcome.stop_reason = f"max_seconds ({max_seconds:.0f}s) exhausted"
            break

        # Build the prompt: base for first attempt, iteration-feedback otherwise.
        if attempt == 1 or prior_eval is None:
            p = build_prompt(ctx)
        else:
            try:
                edits = json.loads(prior_eval.edits_json or "[]")
            except json.JSONDecodeError:
                edits = []
            p = build_iteration_prompt(
                ctx,
                prior_candidate=prior_candidate or "",
                diff_summary=prior_eval.summary,
                diff_edits=edits,
                diff_verdict=prior_eval.verdict,
            )

        # Run inference + record cost on a synthetic DecompResult (avoids
        # re-fetching the context inside decompile_one).
        t0 = time.time()
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
            err = None
        except Exception as e:  # noqa: BLE001
            candidate = ""
            raw = ""
            resp = None
            err = f"{type(e).__name__}: {e}"

        finished = time.time()
        decomp = DecompResult(
            function_id=ctx.function_id,
            function_name=ctx.function_name,
            quality=ctx.quality,
            model=model,
            prompt_chars=len(p["prompt"]),
            response_chars=len(raw),
            candidate_cpp=candidate,
            artifact_path=None,
            started_at=datetime.datetime.fromtimestamp(
                t0, datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            finished_at=datetime.datetime.fromtimestamp(
                finished, datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            tokens_prompt=getattr(resp, "prompt_eval_count", None) if resp else None,
            tokens_eval=getattr(resp, "eval_count", None) if resp else None,
            duration_ns=getattr(resp, "total_duration_ns", None) if resp else None,
            error=err,
            raw_response=raw,
        )

        eval_res: Optional[EvaluationResult] = None
        if err is None and candidate:
            eval_res = evaluate(
                function_id=ctx.function_id,
                function_name=ctx.function_name,
                expected_addr=ctx.function_addr,
                expected_size=ctx.function_size,
                candidate_cpp=candidate,
                decomp_repo=decomp_repo,
                skip_build=skip_build,
            )

        if log:
            with open_db(db_path) as conn:
                log_attempt(conn, decomp_result=decomp, eval_result=eval_res)

        outcome.steps.append(IterationStep(
            attempt_num=attempt,
            decomp=decomp,
            eval=eval_res,
            duration_s=finished - t0,
        ))

        if err is not None:
            outcome.stop_reason = f"inference error: {err}"
            break

        if eval_res is None:
            outcome.stop_reason = "no candidate produced"
            break

        verdict = eval_res.verdict
        outcome.final_verdict = verdict
        if verdict in _TERMINAL_VERDICTS:
            outcome.success = True
            outcome.stop_reason = f"verdict {verdict}"
            return outcome

        # Oscillation guard: if we've already seen this exact candidate,
        # the model is stuck — bail.
        h = _hash_candidate(candidate)
        if h in seen_hashes:
            outcome.stop_reason = (
                f"oscillation: candidate hash {h} repeated at attempt {attempt}"
            )
            break
        seen_hashes.add(h)

        prior_candidate = candidate
        prior_eval = eval_res

    if not outcome.stop_reason:
        outcome.stop_reason = f"max_attempts ({max_attempts}) reached"
    return outcome


# --------------------------------------------------------------------------- #
# CLI integration (gets wired into phase_b.cli in __init__-style fashion)
# --------------------------------------------------------------------------- #

def cli_main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    p = argparse.ArgumentParser(prog="phase_b iterate", description=__doc__)
    p.add_argument("target")
    p.add_argument("--model", default="gemma2:27b")
    p.add_argument("--decomp-repo", default=None)
    p.add_argument("--db", default=None, help="results.sqlite path")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=600.0)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--num-predict", type=int, default=4096)
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args(argv)

    outcome = iterate_one(
        args.target,
        model=args.model,
        max_attempts=args.max_attempts,
        max_seconds=args.max_seconds,
        decomp_repo=Path(args.decomp_repo) if args.decomp_repo else None,
        db_path=Path(args.db) if args.db else None,
        skip_build=args.skip_build,
        temperature=args.temperature,
        num_predict=args.num_predict,
        log=not args.no_log,
    )

    print(f"function: {outcome.function_name}")
    print(f"  model:        {outcome.model}")
    print(f"  attempts:     {outcome.step_count()}")
    print(f"  final verdict:{outcome.final_verdict or '-'}")
    print(f"  success:      {outcome.success}")
    print(f"  stop reason:  {outcome.stop_reason}")
    print()
    for st in outcome.steps:
        v = st.eval.verdict if st.eval else "ERR"
        print(
            f"  [{st.attempt_num}] {v:<12} "
            f"prompt={st.decomp.prompt_chars:>5}c  "
            f"resp={st.decomp.response_chars:>5}c  "
            f"{st.duration_s:5.1f}s"
        )

    return 0 if outcome.success else 1


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
