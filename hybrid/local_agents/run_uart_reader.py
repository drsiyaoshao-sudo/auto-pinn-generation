"""
run_uart_reader.py — Local uart-reader dispatch agent (qwen2.5:0.5b).

The LLM's only job is to parse a free-text request into a structured
{log_path} string. Python then reads the log and prints the UART table.
The LLM never sees PRIVATE content — all file access is in Python.

Skill contract (matches .claude/agents/uart-reader.md):
  execution:   local
  retrieves:   PRIVATE (UART log files — via _read_log(), not LLM)
  receives:    free-text request (PUBLIC)
  produces:    uart_table (DERIVED-OK → stdout)
               session_summary (DERIVED-OK → caller)
  must_not_forward: raw UART binary stream (never returned by this module)

Usage (free-text):
    python hybrid/local_agents/run_uart_reader.py "read gait_uart.log"
    python hybrid/local_agents/run_uart_reader.py "show UART output from /tmp/gait_uart.log"
    python hybrid/local_agents/run_uart_reader.py "parse the latest simulation log"

Usage (explicit args — bypass LLM):
    python hybrid/local_agents/run_uart_reader.py --log_path /tmp/gait_uart.log
    python hybrid/local_agents/run_uart_reader.py --log_path latest

As a library:
    from hybrid.local_agents.run_uart_reader import dispatch
    result = dispatch("read gait_uart.log")
    # → {"status": "PASS", "log_path": "...", "steps": 20, "snapshots": 4, ...}
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIMULATOR_DIR = _REPO_ROOT / "simulator"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SIMULATOR_DIR))

from hybrid.wrapper.ollama_client import LocalLLM

_PARSE_SCHEMA = "log_path (file path string, or the word 'latest' if not specified)"
_PARSE_SYSTEM = (
    "You extract exactly one field from a user request and return valid JSON only. "
    "No explanation. No markdown. Output: {\"log_path\": \"<value>\"}. "
    "If the request names a specific file path, extract it exactly. "
    "If no path is mentioned or the user says 'latest', output {\"log_path\": \"latest\"}."
)

# Default UART log locations (searched when 'latest' is requested)
_DEFAULT_LOG_PATHS = [
    Path("/tmp/gait_uart.log"),
    _SIMULATOR_DIR / "logs" / "gait_uart.log",
]


def _find_latest_log() -> str:
    """Return path of the most recently modified UART log, or raise if none found."""
    candidates = []
    for p in _DEFAULT_LOG_PATHS:
        if p.exists():
            candidates.append(p)
    # Also search simulator/logs/ for any .log files
    logs_dir = _SIMULATOR_DIR / "logs"
    if logs_dir.exists():
        candidates.extend(logs_dir.glob("*.log"))

    if not candidates:
        raise FileNotFoundError(
            "No UART log found. Run the simulation first, or specify --log_path."
        )
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _parse_request(request: str, llm: LocalLLM) -> str:
    """Use 0.5B to parse free-text into log_path string."""
    parsed = llm.parse_to_json(request, schema_hint=_PARSE_SCHEMA)
    log_path = parsed.get("log_path", "latest").strip()
    return log_path if log_path else "latest"


def _read_log(log_path: str) -> dict:
    """Read UART log and return DERIVED-OK summary dict."""
    # Resolve 'latest' keyword
    if log_path == "latest":
        log_path = _find_latest_log()

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"UART log not found: {log_path}")

    with open(log_path) as f:
        text = f.read()

    # Import signal_analysis from simulator directory
    from signal_analysis import parse_uart_log
    steps, snaps, ends = parse_uart_log(text)

    # ── Print the standard UART table (uart-reader agent format) ──────────────
    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n─────────────────────────────────────")
    print(f"UART SESSION — {log_path}  [{ts}]")
    print(f"─────────────────────────────────────")

    print("\nSTEP lines:")
    for s in steps:
        print(f"  #{s.step_index}  ts={s.ts_ms:.0f}ms  acc={s.peak_acc_mag:.1f}  "
              f"gyr_y={s.peak_gyr_y:.1f}  cadence={s.cadence_spm:.1f} spm")

    print("\nSNAPSHOT lines:")
    for sn in snaps:
        print(f"  snap={sn.anchor_step}  si_stance={sn.si_stance_pct:.1f}%  "
              f"si_swing={sn.si_swing_pct:.1f}%  cadence={sn.mean_cadence_spm:.1f} spm")

    print("\nSESSION_END:")
    if ends:
        for e in ends:
            print(f"  total_steps={e.total_steps}")
    else:
        print("  (not found)")

    final_si      = snaps[-1].si_stance_pct if snaps else None
    final_cadence = snaps[-1].mean_cadence_spm if snaps else None

    print("\nSUMMARY:")
    print(f"  Steps detected : {len(steps)}")
    print(f"  Snapshots      : {len(snaps)}")
    print(f"  Final SI       : {final_si:.1f}%" if final_si is not None else "  Final SI       : N/A")
    print(f"  Final cadence  : {final_cadence:.1f} spm" if final_cadence is not None else "  Final cadence  : N/A")
    print(f"─────────────────────────────────────")

    return {
        "status":         "PASS",
        "log_path":       log_path,
        "steps":          len(steps),
        "snapshots":      len(snaps),
        "session_end":    bool(ends),
        "final_si":       float(final_si) if final_si is not None else None,
        "final_cadence":  float(final_cadence) if final_cadence is not None else None,
    }


def dispatch(request: str, llm: LocalLLM = None) -> dict:
    """Parse free-text request with 0.5B and dispatch to UART log reader.

    Args:
        request: free-text e.g. "read gait_uart.log" or "show the latest UART output"
        llm:     LocalLLM instance (created with default model if None)

    Returns:
        DERIVED-OK result dict — safe to forward to plot-orchestrator.
    """
    if llm is None:
        llm = LocalLLM()

    print(f"[run_uart_reader] parsing: '{request}'")
    log_path = _parse_request(request, llm)
    print(f"[run_uart_reader] → log_path={log_path}")

    return _read_log(log_path)


def dispatch_explicit(log_path: str = "latest") -> dict:
    """Bypass LLM — call log reader directly with known path (e.g. from agents)."""
    print(f"[run_uart_reader] explicit: log_path={log_path}")
    return _read_log(log_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GaitSense local uart-reader — dispatch via qwen2.5:0.5b"
    )
    parser.add_argument("request",    nargs="?", default=None,
                        help="Free-text request, e.g. 'read gait_uart.log'")
    parser.add_argument("--log_path", default=None,
                        help="Explicit log file path (bypasses LLM parsing), use 'latest' for most recent")
    args = parser.parse_args()

    if args.log_path:
        result = dispatch_explicit(args.log_path)
    elif args.request:
        result = dispatch(args.request)
    else:
        parser.error("Provide a free-text request or --log_path")

    print("\n[run_uart_reader] DERIVED-OK result:")
    print(json.dumps(result, indent=2))
