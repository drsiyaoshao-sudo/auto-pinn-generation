"""
run_plotter.py — Local plotter dispatch agent (qwen2.5:0.5b).

The LLM's only job is to parse a free-text request into a structured
{profile, mode} dict. Python then calls plot_signal_check.run() directly.
The LLM never sees PRIVATE content — all file access is in Python.

Skill contract (matches .claude/agents/plotter.md):
  execution:   local
  retrieves:   PRIVATE (walker_model.py — via plot_signal_check.run(), not LLM)
               PUBLIC  (amendments.md — via plot_signal_check.run(), not LLM)
  receives:    free-text request (PUBLIC)
  produces:    plot PNG (DERIVED-OK → file)
               data table (DERIVED-OK → stdout)
               result dict (DERIVED-OK → caller)
  must_not_forward: raw signal arrays (never returned by this module)

Usage (free-text):
    python hybrid/local_agents/run_plotter.py "plot the flat profile"
    python hybrid/local_agents/run_plotter.py "show pathological bad_wear"
    python hybrid/local_agents/run_plotter.py "stairs healthy"

Usage (explicit args — bypass LLM):
    python hybrid/local_agents/run_plotter.py --profile flat
    python hybrid/local_agents/run_plotter.py --profile slope --mode pathological

As a library:
    from hybrid.local_agents.run_plotter import dispatch
    result = dispatch("plot the flat profile")
    # → {"status": "PASS", "profile": "flat", "mode": "healthy", "plot_path": "..."}
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

VALID_PROFILES = ["flat", "bad_wear", "stairs", "slope"]
VALID_MODES    = ["healthy", "pathological"]

_PARSE_SCHEMA = "profile (one of: flat, bad_wear, stairs, slope), mode (one of: healthy, pathological)"
_PARSE_SYSTEM = (
    "You extract exactly two fields from a user request and return valid JSON only. "
    "No explanation. No markdown. Output: {\"profile\": \"<value>\", \"mode\": \"<value>\"}. "
    f"Valid profiles: flat, bad_wear, stairs, slope. Valid modes: healthy, pathological. "
    "Default mode is healthy if not mentioned."
)


def _parse_request(request: str, llm: LocalLLM) -> tuple:
    """Use 0.5B to parse free-text into (profile, mode). Returns (profile, mode)."""
    parsed = llm.parse_to_json(request, schema_hint=_PARSE_SCHEMA)
    profile = parsed.get("profile", "").lower().strip()
    mode    = parsed.get("mode",    "healthy").lower().strip()

    if profile not in VALID_PROFILES:
        raise ValueError(
            f"LLM returned invalid profile '{profile}'. "
            f"Valid: {VALID_PROFILES}. Raw: {parsed}"
        )
    if mode not in VALID_MODES:
        mode = "healthy"

    return profile, mode


def _run_plot(profile: str, mode: str) -> dict:
    """Call plot_signal_check.run() directly. Returns DERIVED-OK result dict."""
    from simulator.pinn.plot_signal_check import run as plot_run
    plot_run(profile, mode)

    # Derive output path (mirrors what plot_signal_check.py writes)
    plots_dir = _REPO_ROOT / "docs" / "executive_branch_document" / "plots"
    suffix    = "" if mode == "healthy" else f"_{mode}"
    plot_path = str(plots_dir / f"{profile}{suffix}_signal_check.png")

    return {
        "status":    "PASS",
        "profile":   profile,
        "mode":      mode,
        "plot_path": plot_path,
    }


def dispatch(request: str, llm: LocalLLM = None) -> dict:
    """Parse free-text request with 0.5B and dispatch to plot_signal_check.

    Args:
        request: free-text e.g. "plot the flat profile" or "stairs pathological"
        llm:     LocalLLM instance (created with default model if None)

    Returns:
        DERIVED-OK result dict — safe to forward to plot-orchestrator.
    """
    if llm is None:
        llm = LocalLLM()

    print(f"[run_plotter] parsing: '{request}'")
    profile, mode = _parse_request(request, llm)
    print(f"[run_plotter] → profile={profile}  mode={mode}")

    return _run_plot(profile, mode)


def dispatch_explicit(profile: str, mode: str = "healthy") -> dict:
    """Bypass LLM — call plot directly with known profile/mode (e.g. from agents)."""
    print(f"[run_plotter] explicit: profile={profile}  mode={mode}")
    return _run_plot(profile, mode)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GaitSense local plotter — dispatch via qwen2.5:0.5b"
    )
    parser.add_argument("request",  nargs="?", default=None,
                        help="Free-text request, e.g. 'plot the flat profile'")
    parser.add_argument("--profile", choices=VALID_PROFILES,
                        help="Explicit profile (bypasses LLM parsing)")
    parser.add_argument("--mode",    choices=VALID_MODES, default="healthy",
                        help="healthy (default) or pathological")
    args = parser.parse_args()

    if args.profile:
        result = dispatch_explicit(args.profile, args.mode)
    elif args.request:
        result = dispatch(args.request)
    else:
        parser.error("Provide a free-text request or --profile")

    print("\n[run_plotter] DERIVED-OK result:")
    print(json.dumps(result, indent=2))
