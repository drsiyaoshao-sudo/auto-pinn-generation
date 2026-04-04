"""
Local plotter agent — PoC for hybrid local/cloud LLM architecture.

Skill contract:
  execution:   local (Qwen2.5-Coder:7b via Ollama)
  retrieves:   PRIVATE (walker_model.py, signal arrays)
  receives:    profile (PUBLIC), mode (PUBLIC)
  produces:    signal_plot (DERIVED-OK → file)
               data_table  (DERIVED-OK → stdout)
  may_forward: plot path + peak scalars to plot-orchestrator
  must_not_forward: raw signal arrays

Usage:
  python hybrid/local_agents/run_plotter.py --profile flat
  python hybrid/local_agents/run_plotter.py --profile stairs --mode pathological
  python hybrid/local_agents/run_plotter.py --profile flat --dry-run
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import requests

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
WALKER_PATH = REPO_ROOT / "simulator" / "walker_model.py"
PLOTS_DIR   = REPO_ROOT / "docs" / "executive_branch_document" / "plots"
AMEND_PATH  = REPO_ROOT / "docs" / "gaitsense_code" / "amendments.md"

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "qwen2.5-coder:7b"

VALID_PROFILES = ["flat", "bad_wear", "stairs", "slope"]
VALID_MODES    = ["healthy", "pathological"]


def extract_walker_interface(walker_src: str) -> str:
    """
    Extract only the public interface from walker_model.py — the profile
    names, generate_imu_sequence signature, and PROFILES dict.
    PRIVATE derivation formulas are not needed by the plotter.
    Keeps prompt small enough for 7B to complete in time.
    """
    lines = walker_src.split("\n")
    keep = []
    in_profiles = False
    for i, line in enumerate(lines):
        # keep imports, constants, function signatures, profile names
        if any(x in line for x in [
            "import", "PROFILES", "def generate_imu_sequence",
            "WalkerProfile", "ODR_HZ", "flat", "bad_wear", "stairs", "slope",
            "cadence_spm", "step_length_m", "vertical_oscillation_cm",
            "si_stance_true_pct", "n_steps", "si_override", "return"
        ]):
            keep.append(line)
        if "PROFILES" in line:
            in_profiles = True
        if in_profiles:
            keep.append(line)
            if line.strip() == "}":
                in_profiles = False
    # deduplicate while preserving order
    seen = set()
    result = []
    for l in keep:
        if l not in seen:
            seen.add(l)
            result.append(l)
    return "\n".join(result[:80])  # hard cap


def build_prompt(profile: str, mode: str, walker_interface: str, amend_snippet: str) -> str:
    """
    Build the prompt sent to the local LLM.
    Only the walker interface (not full source) is included — minimises
    prompt size for 7B and avoids sending full derivation chain to model.
    """
    si_line = "si_override=25.0" if mode == "pathological" else "si_override=None"
    return textwrap.dedent(f"""
    You are the GaitSense plotter agent. Write a complete runnable Python script.
    Output ONLY the script. No explanation. No markdown fences.

    TASK: Generate a 3-panel IMU diagnostic plot for profile='{profile}', mode='{mode}'.

    WALKER MODEL INTERFACE (available at {REPO_ROOT}/simulator/walker_model.py):
    {walker_interface}

    AMENDMENT 11 CONTEXT:
    {amend_snippet}

    SCRIPT REQUIREMENTS:
    import sys
    sys.path.insert(0, '{REPO_ROOT}/simulator')
    from walker_model import generate_imu_sequence, PROFILES
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import butter, filtfilt

    profile = PROFILES['{profile}']
    data = generate_imu_sequence(profile, n_steps=20, {si_line})
    # data shape: (N, 6) columns: ax, ay, az, gx, gy, gz in SI units

    # 15 Hz lowpass on az (column 2), fs=208 Hz — firmware matched (Amendment 11)
    b, a = butter(4, 15/(208/2), btype='low')
    az_filt = filtfilt(b, a, data[:, 2])

    # 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # Panel 1: az raw + filtered
    # Panel 2: gy raw (column 4)
    # Panel 3: az filtered with peak markers

    # Save to: {PLOTS_DIR}/{profile}_signal_check.png  dpi=150
    out_path = '{PLOTS_DIR}/{profile}_signal_check.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Print data table then plot path
    az_peak_raw  = float(np.max(np.abs(data[:, 2])))
    az_peak_filt = float(np.max(np.abs(az_filt)))
    gy_peak      = float(np.max(np.abs(data[:, 4])))
    print(f"acc_z peak raw:      {{az_peak_raw:.2f}} m/s2")
    print(f"acc_z peak filtered: {{az_peak_filt:.2f}} m/s2")
    print(f"gyr_y peak:          {{gy_peak:.2f}} dps")
    print(f"PLOT_PATH: {{out_path}}")

    Complete the script above. Fill in the 3 panel plotting code.
    Output ONLY executable Python. No markdown.
    """).strip()


def extract_amendment_snippet(amend_text: str) -> str:
    """Extract Amendment 11 text only — PUBLIC, safe to include in prompt."""
    lines = amend_text.split("\n")
    snippet = []
    in_amend11 = False
    for line in lines:
        if "Amendment 11" in line:
            in_amend11 = True
        elif in_amend11 and line.startswith("### Amendment"):
            break
        if in_amend11:
            snippet.append(line)
    return "\n".join(snippet[:30])  # cap at 30 lines


def call_ollama(prompt: str) -> str:
    """Stream response tokens — avoids read timeout on long code generation."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    # (connect_timeout, read_timeout) — long read to allow prompt tokenisation
    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(10, 600))
    resp.raise_for_status()
    chunks = []
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        chunk = json.loads(raw_line)
        token = chunk.get("message", {}).get("content", "")
        chunks.append(token)
        print(token, end="", flush=True)
        if chunk.get("done"):
            break
    print()  # newline after streaming
    return "".join(chunks)


def strip_fences(code: str) -> str:
    """Remove markdown code fences if the model wraps output anyway."""
    lines = code.strip().split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def run(profile: str, mode: str, dry_run: bool = False) -> dict:
    """
    Main entry point. Returns a DERIVED-OK result dict safe to forward upstream.
    Raw signal arrays are never returned — only scalars and file paths.
    """
    print(f"[plotter] profile={profile}  mode={mode}  model={MODEL}")

    # ── read PRIVATE sources locally ──────────────────────────────────────────
    walker_src       = WALKER_PATH.read_text()
    walker_interface = extract_walker_interface(walker_src)   # trimmed — interface only
    amend_text       = AMEND_PATH.read_text() if AMEND_PATH.exists() else ""
    amend_snippet    = extract_amendment_snippet(amend_text)  # PUBLIC subset only

    prompt = build_prompt(profile, mode, walker_interface, amend_snippet)

    if dry_run:
        print("[plotter] dry-run: prompt built, skipping Ollama call")
        print(f"[plotter] prompt length: {len(prompt)} chars")
        print(f"[plotter] walker interface lines: {len(walker_interface.splitlines())}")
        return {"status": "dry-run", "profile": profile, "mode": mode}

    # ── call local LLM ────────────────────────────────────────────────────────
    print(f"[plotter] calling {MODEL} via Ollama...")
    raw_code  = call_ollama(prompt)
    plot_code = strip_fences(raw_code)

    # ── write generated script ────────────────────────────────────────────────
    script_dir  = REPO_ROOT / "hybrid" / "generated"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"plot_{profile}_{mode}.py"
    script_path.write_text(plot_code)
    print(f"[plotter] generated script → {script_path}")

    # ── execute generated script ──────────────────────────────────────────────
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=60
    )

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"[plotter] SCRIPT FAILED:\n{stderr}")
        return {
            "status": "FAIL",
            "profile": profile,
            "mode": mode,
            "error": stderr[-500:],   # truncated — no raw data in error
        }

    print(stdout)

    # ── extract DERIVED-OK output — path and scalars only ─────────────────────
    plot_path = None
    for line in stdout.split("\n"):
        if line.startswith("PLOT_PATH:"):
            plot_path = line.split("PLOT_PATH:")[-1].strip()

    # DERIVED-OK result — safe to forward to plot-orchestrator
    return {
        "status": "PASS",
        "profile": profile,
        "mode": mode,
        "plot_path": plot_path,
        "script_path": str(script_path),
        # raw signal arrays are NOT included here
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaitSense local plotter agent")
    parser.add_argument("--profile", required=True, choices=VALID_PROFILES)
    parser.add_argument("--mode",    default="healthy", choices=VALID_MODES)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompt only, skip Ollama call")
    args = parser.parse_args()

    result = run(args.profile, args.mode, dry_run=args.dry_run)
    print("\n[plotter] DERIVED-OK result (safe to forward):")
    print(json.dumps(result, indent=2))
