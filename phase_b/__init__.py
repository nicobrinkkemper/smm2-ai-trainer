"""Phase B integration: turn smm2-decomp's Phase A context bundle + diff
classifier into a closed-loop autonomous decompilation pipeline driven by
local LLMs (Gemma, Qwen Coder, …).

The package is intentionally small. Each module has a single job:

- ``context``  — pull a function context bundle from smm2-decomp's FKB CLI
- ``prompt``   — format the bundle into a ChatML/instruct prompt
- ``ollama``   — talk to a local Ollama server (or any compatible API)
- ``runner``   — top-level "decompile this function" entry point
- ``cli``      — argparse front-end

Other tasks (evaluator, benchmark, iteration controller) are built on top of
these primitives.
"""

from .context import FetchError, fetch_context
from .prompt import build_prompt
from .ollama import OllamaClient, OllamaError
from .runner import DecompResult, decompile_one
# Note: we do NOT re-export the `evaluate` function from `.evaluate` here
# because doing so shadows the submodule (Python sees `phase_b.evaluate` as
# the rebound function, not the module). Callers should
#    from phase_b.evaluate import evaluate
# explicitly. Same logic for any other name that collides with its module.
from .evaluate import EvaluationResult, EvaluationError
from .results import open_db, log_attempt, get_attempts, model_scoreboard

__all__ = [
    "FetchError",
    "fetch_context",
    "build_prompt",
    "OllamaClient",
    "OllamaError",
    "DecompResult",
    "decompile_one",
    "EvaluationResult",
    "EvaluationError",
    "open_db",
    "log_attempt",
    "get_attempts",
    "model_scoreboard",
]
