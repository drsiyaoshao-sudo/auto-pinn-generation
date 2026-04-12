"""
run_model_compare.py — Local model comparison dispatch agent (qwen2.5:0.5b).

The LLM's only job is to parse a free-text request into {good_run_id, bad_run_id}.
Python then calls plot_model_compare.run() directly. The LLM never sees
checkpoint weights, walker_model code, or any PRIVATE content.

Skill contract:
  execution:   local
  retrieves:   PRIVATE (checkpoint .pt files, walker_model — via Python, not LLM)
  receives:    free-text request (PUBLIC)
  produces:    comparison plot PNG (DERIVED-OK → file)
               summary table (DERIVED-OK → stdout)
               result dict (DERIVED-OK → caller)
  must_not_forward: checkpoint weights, norm stats, walker_model equations

Usage (free-text):
    python hybrid/local_agents/run_model_compare.py "compare good v1 vs bad random"
    python hybrid/local_agents/run_model_compare.py "show bad model v3 against good v1"
    python hybrid/local_agents/run_model_compare.py "show the untrained model vs v1"

Usage (explicit args — bypass LLM):
    python hybrid/local_agents/run_model_compare.py --good v1 --bad random
    python hybrid/local_agents/run_model_compare.py --good v1 --bad v3

As a library:
    from hybrid.local_agents.run_model_compare import dispatch
    result = dispatch("compare good v1 against untrained")
    # → {"good_run_id": "v1", "bad_run_id": "random", "good_passes": 4, ...}
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

VALID_RUN_IDS = [
    "v1", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v21_2k",
    "random",
]

_PARSE_SCHEMA = (
    "good_run_id (trained PINN run_id, e.g. v1, v21_2k), "
    "bad_run_id (untrained/random or an early run_id, e.g. random, v3)"
)
_PARSE_SYSTEM = (
    "You extract exactly two fields from a user request and return valid JSON only. "
    "No explanation. No markdown. "
    "Output: {\"good_run_id\": \"<value>\", \"bad_run_id\": \"<value>\"}. "
    "If the user says 'untrained', 'random', or 'random init', use 'random' for bad_run_id. "
    "If no good run_id is mentioned, default to 'v1'. "
    "If no bad run_id is mentioned, default to 'random'."
)


def _parse_request(request: str, llm: LocalLLM) -> tuple:
    """Use 0.5B to parse free-text into (good_run_id, bad_run_id)."""
    parsed       = llm.parse_to_json(request, schema_hint=_PARSE_SCHEMA)
    good_run_id  = parsed.get("good_run_id", "v1").strip()
    bad_run_id   = parsed.get("bad_run_id",  "random").strip()
    # Normalise unknown values to defaults
    if good_run_id not in VALID_RUN_IDS:
        good_run_id = "v1"
    if bad_run_id not in VALID_RUN_IDS:
        bad_run_id = "random"
    return good_run_id, bad_run_id


def _run_compare(good_run_id: str, bad_run_id: str) -> dict:
    """Call plot_model_compare.run() directly. Returns DERIVED-OK result dict."""
    from simulator.pinn.plot_model_compare import run as compare_run
    result = compare_run(good_run_id, bad_run_id)
    result["status"] = "PASS"
    return result


def dispatch(request: str, llm: LocalLLM = None) -> dict:
    """Parse free-text with 0.5B and dispatch to plot_model_compare.

    Args:
        request: free-text e.g. "compare good v1 against untrained"
        llm:     LocalLLM instance (created with default model if None)

    Returns:
        DERIVED-OK result dict — safe to forward to plot-orchestrator or Justice.
    """
    if llm is None:
        llm = LocalLLM()

    print(f"[run_model_compare] parsing: '{request}'")
    good_run_id, bad_run_id = _parse_request(request, llm)
    print(f"[run_model_compare] → good={good_run_id}  bad={bad_run_id}")

    return _run_compare(good_run_id, bad_run_id)


def dispatch_explicit(good_run_id: str = "v1", bad_run_id: str = "random") -> dict:
    """Bypass LLM — call comparison directly with known args (e.g. from agents)."""
    print(f"[run_model_compare] explicit: good={good_run_id}  bad={bad_run_id}")
    return _run_compare(good_run_id, bad_run_id)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GaitSense hybrid demo — good vs bad PINN comparison"
    )
    parser.add_argument("request", nargs="?", default=None,
                        help="Free-text request, e.g. 'compare good v1 vs untrained'")
    parser.add_argument("--good",  default=None,
                        help="Good checkpoint run_id (bypasses LLM)")
    parser.add_argument("--bad",   default=None,
                        help="Bad checkpoint run_id or 'random' (bypasses LLM)")
    args = parser.parse_args()

    if args.good:
        result = dispatch_explicit(args.good, args.bad or "random")
    elif args.request:
        result = dispatch(args.request)
    else:
        # Default demo run
        result = dispatch_explicit("v1", "random")

    print("\n[run_model_compare] DERIVED-OK result:")
    # Print summary without raw arrays
    safe_result = {k: v for k, v in result.items() if k != "summary"}
    print(json.dumps(safe_result, indent=2))
