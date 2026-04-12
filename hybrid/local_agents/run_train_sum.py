"""
run_train_sum.py — Local train-sum dispatch agent (qwen2.5:0.5b).

The LLM's only job is to parse a free-text request into a structured
{run_id} string. Python then calls plot_training_loss.run() directly.
The LLM never sees PRIVATE content — all file access is in Python.

Skill contract (matches .claude/agents/train-sum.md):
  execution:   local
  retrieves:   DERIVED-OK (run_*_metrics.jsonl — via plot_training_loss.run(), not LLM)
               PUBLIC     (train_config.json — via plot_training_loss.run(), not LLM)
  receives:    free-text request (PUBLIC)
  produces:    loss curve plot PNG (DERIVED-OK → file)
               summary table (DERIVED-OK → stdout)
               result dict (DERIVED-OK → caller)
  must_not_forward: raw epoch arrays (never returned by this module)

Usage (free-text):
    python hybrid/local_agents/run_train_sum.py "summarise training run v21_2k"
    python hybrid/local_agents/run_train_sum.py "show loss curves for v19"
    python hybrid/local_agents/run_train_sum.py "plot the latest run"

Usage (explicit args — bypass LLM):
    python hybrid/local_agents/run_train_sum.py --run_id v21_2k
    python hybrid/local_agents/run_train_sum.py --run_id latest

As a library:
    from hybrid.local_agents.run_train_sum import dispatch
    result = dispatch("summarise training run v21_2k")
    # → {"status": "PASS", "run_id": "v21_2k", "best_epoch": 312, "best_val": 0.043}
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "simulator" / "pinn"))

from hybrid.wrapper.ollama_client import LocalLLM

_PARSE_SCHEMA = "run_id (string, e.g. v21_2k, v19, v15) or the word 'latest'"
_PARSE_SYSTEM = (
    "You extract exactly one field from a user request and return valid JSON only. "
    "No explanation. No markdown. Output: {\"run_id\": \"<value>\"}. "
    "If the request says 'latest' or gives no specific run, output {\"run_id\": \"latest\"}. "
    "run_id format examples: v21_2k, v19, v15, v10. Extract the version identifier only."
)


def _parse_request(request: str, llm: LocalLLM) -> str:
    """Use 0.5B to parse free-text into run_id string."""
    parsed = llm.parse_to_json(request, schema_hint=_PARSE_SCHEMA)
    run_id = parsed.get("run_id", "latest").strip()
    return run_id if run_id else "latest"


def _run_summary(run_id: str) -> dict:
    """Call plot_training_loss.run() directly. Returns DERIVED-OK result dict."""
    from simulator.pinn.plot_training_loss import run as plot_run
    result = plot_run(run_id if run_id != "latest" else None)

    return {
        "status":     "PASS",
        "run_id":     result["run_id"],
        "best_epoch": result["best_epoch"],
        "best_val":   result["best_val"],
        "plot_path":  str(
            _REPO_ROOT / "docs" / "executive_branch_document" / "plots" / "pinn_training"
            / f"loss_curve_{result['run_id']}.png"
        ),
    }


def dispatch(request: str, llm: LocalLLM = None) -> dict:
    """Parse free-text request with 0.5B and dispatch to plot_training_loss.

    Args:
        request: free-text e.g. "summarise training run v21_2k" or "latest"
        llm:     LocalLLM instance (created with default model if None)

    Returns:
        DERIVED-OK result dict — safe to forward to plot-orchestrator.
    """
    if llm is None:
        llm = LocalLLM()

    print(f"[run_train_sum] parsing: '{request}'")
    run_id = _parse_request(request, llm)
    print(f"[run_train_sum] → run_id={run_id}")

    return _run_summary(run_id)


def dispatch_explicit(run_id: str = "latest") -> dict:
    """Bypass LLM — call plot directly with known run_id (e.g. from agents)."""
    print(f"[run_train_sum] explicit: run_id={run_id}")
    return _run_summary(run_id)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GaitSense local train-sum — dispatch via qwen2.5:0.5b"
    )
    parser.add_argument("request",   nargs="?", default=None,
                        help="Free-text request, e.g. 'show loss for v21_2k'")
    parser.add_argument("--run_id",  default=None,
                        help="Explicit run_id (bypasses LLM parsing), use 'latest' for most recent")
    args = parser.parse_args()

    if args.run_id:
        result = dispatch_explicit(args.run_id)
    elif args.request:
        result = dispatch(args.request)
    else:
        parser.error("Provide a free-text request or --run_id")

    print("\n[run_train_sum] DERIVED-OK result:")
    print(json.dumps(result, indent=2))
