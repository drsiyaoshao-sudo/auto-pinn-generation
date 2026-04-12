"""
run_physics_insight.py — Cloud physics insight agent (claude-sonnet-4-6).

Step 2 of the hybrid pipeline. Receives the DERIVED-OK comparison plot from
the local agent and uses the Claude API (vision + reasoning) to produce
physics-grounded insight text. The cloud model never sees PRIVATE content.

Hybrid contract:
  execution:   cloud (claude-sonnet-4-6)
  retrieves:   DERIVED-OK (comparison plot PNG — image bytes only, no equations)
               PUBLIC     (Article I physics primer — gait primitives definition)
  receives:    plot_path (DERIVED-OK), physics primer (PUBLIC)
  produces:    physics insight text (PUBLIC → stdout)
  must_not_forward: nothing — all inputs are DERIVED-OK or PUBLIC

Privacy derivation chain:
  walker_model.py equations        PRIVATE  (never leaves local)
       │  run locally → produce waveform shapes
       ▼
  Comparison PNG (shapes only)     DERIVED-OK  (sent to cloud)
       │  Claude sees shapes, NOT equations
       ▼
  Physics insight text             PUBLIC  (no IP content — only clinical observation)

Usage:
    python hybrid/cloud_agents/run_physics_insight.py --plot docs/.../model_compare_v1_vs_random.png
    python hybrid/cloud_agents/run_physics_insight.py  # uses latest comparison plot

As a library:
    from hybrid.cloud_agents.run_physics_insight import run_insight
    result = run_insight(plot_path="docs/.../model_compare_v1_vs_random.png")
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ── Load .env if present (ANTHROPIC_API_KEY) ─────────────────────────────────
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Early API key check — fail before accounting block, not after ─────────────
def _check_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print()
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print()
        print("Add it to the repo .env file (git-ignored, never committed):")
        print(f"  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> {_REPO_ROOT}/.env")
        print()
        print("Or export it for this session:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print()
        sys.exit(1)

_check_api_key()

# ── PRIVATE corpus token counts (pre-measured; never sent) ───────────────────
_PRIVATE_CORPUS = [
    ("simulator/walker_model.py",              3878),
    ("simulator/pinn/physics_loss.py",         3215),
    ("simulator/pinn/pinn_model.py",           1611),
    ("simulator/pinn/train_pinn.py",           3774),
    ("simulator/pinn/generate_training_data.py", 4009),
    ("simulator/gait_algorithm.py",            5199),
    ("simulator/pipeline.py",                  2187),
]
_TOTAL_SHIELDED = sum(t for _, t in _PRIVATE_CORPUS)

# ── PUBLIC physics primer (Article I) — safe to send to cloud ─────────────────
_PHYSICS_PRIMER = """
ARTICLE I PHYSICS PRIMER (public — no proprietary equations)
─────────────────────────────────────────────────────────────
The three and only three first-order primitives of walking gait are:

  1. Vertical Oscillation (cm)  — amplitude of centre-of-mass vertical
                                   movement per step (drives az waveform)
  2. Cadence (steps/min)        — fundamental temporal frequency of gait
                                   (sets the oscillation period of gy and az)
  3. Step Length (m)            — spatial extent of each step
                                   (coupled to cadence via walking speed)

IMU axis physical meaning (sensor mounted on ankle):
  gy (dps)  — gyroscope Y axis, ankle angular velocity.
               The push-off phase produces a large positive peak (toe-off).
               Swing phase produces a smaller negative trough.
               Peak gy amplitude is a direct function of Cadence × Vertical Oscillation.
               A well-trained PINN must reproduce: timing of the push-off peak,
               amplitude proportional to cadence, and the zero-crossing at mid-stance.

  az (m/s²) — accelerometer Z axis (vertical), dominated by gravity (≈9.81 m/s²)
               modulated by Vertical Oscillation each step.
               A heel-strike produces a characteristic spike above 1g.
               Mid-stance (stance phase) produces a local minimum as the foot loads.
               A well-trained PINN must reproduce: the stance/swing modulation pattern
               and the gravity offset that shows the sensor is correctly oriented.

SIGNAL SHAPE CRITERIA for gait algorithm compatibility:
  gy: Must produce a push-off peak ≥ 80 dps to trigger the step detector.
      Must cross zero during mid-stance (step phase segmentation relies on this).
  az: Must show a toe-off dip below 2.94 m/s² to confirm foot-off.
      Must modulate around 9.81 m/s² (gravity) — flat signal indicates no walking.

WHAT A BADLY TRAINED MODEL LOOKS LIKE (physical interpretation):
  gy flat near 0  → the model learned no ankle angular velocity structure;
                    it cannot generate the push-off event the step detector needs.
  az flat at 0    → the model does not encode gravity orientation;
                    it would appear as if the sensor were in free-fall, not walking.

DERIVATION PRIVACY NOTE (for your analysis):
  You are seeing only the OUTPUT SHAPES of the physical model — not the equations.
  The walker_model.py derivations (which encode how cadence, step length, and
  vertical oscillation couple into the IMU signal) remain local and undisclosed.
  Your analysis is based on observable biomechanical principles only.
"""

_CLOUD_MODEL = "claude-sonnet-4-6"


def _load_plot_as_b64(plot_path: str) -> str:
    """Read PNG and encode as base64 for the Anthropic vision API."""
    with open(plot_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _count_primer_tokens() -> int:
    """Approximate token count for the physics primer (chars/4)."""
    return len(_PHYSICS_PRIMER) // 4


def _print_hybrid_header(plot_path: str, primer_tokens: int):
    """Print the hybrid accounting block before the cloud call."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          HYBRID INTELLIGENCE PIPELINE — BOUNDARY ACCOUNTING         ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  STEP 1 (already complete)  LOCAL agent                             ║")
    print("║    Model : qwen2.5:0.5b  (parse only — no PRIVATE data seen)        ║")
    print("║    Reads : walker_model.py + checkpoint .pt  [PRIVATE]              ║")
    print("║    Writes: comparison PNG                    [DERIVED-OK]           ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  STEP 2 (running now)       CLOUD agent                             ║")
    print(f"║    Model : {_CLOUD_MODEL:<59}║")
    print("║    Reads : comparison PNG (image)            [DERIVED-OK]           ║")
    print("║    Reads : Article I physics primer          [PUBLIC]               ║")
    print("║    Writes: physics insight text              [PUBLIC]               ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  PRIVACY DERIVATION — what stays local                              ║")
    for fname, toks in _PRIVATE_CORPUS:
        label = os.path.basename(fname)
        print(f"║    {label:<45} ~{toks:>5} tok  SHIELDED ║")
    print(f"║    {'Training data arrays + checkpoint weights':<45} binary  SHIELDED ║")
    print(f"║  {'─'*68}║")
    print(f"║    Total PRIVATE tokens never sent to cloud:      ~{_TOTAL_SHIELDED:>6} tokens    ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  WHAT IS SENT TO CLOUD (DERIVED-OK + PUBLIC)                        ║")
    png_kb = os.path.getsize(plot_path) // 1024
    print(f"║    Comparison PNG (shapes only, no equations)    {png_kb:>4} KB image      ║")
    print(f"║    Article I physics primer                      ~{primer_tokens:>4} tokens        ║")
    print(f"║  {'─'*68}║")
    print(f"║    Cloud model sees: SHAPES + PUBLIC DEFINITIONS only              ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  WHY THIS PROTECTS PATENTABLE KNOWLEDGE                             ║")
    print("║    The derivation chain (how cadence + step_length + vertical_      ║")
    print("║    oscillation couple into gy/az waveforms) is encoded ONLY in      ║")
    print("║    walker_model.py. That derivation is the IP.                      ║")
    print("║    Sending it to any third-party API would:                         ║")
    print("║      1. Disclose the formula structure (patent risk)                ║")
    print("║      2. Expose the parameter distributions (trade secret risk)      ║")
    print("║      3. Transfer the derivation into a cloud log (GDPR risk)        ║")
    print("║    The hybrid boundary ensures only OUTCOMES (shapes) reach cloud.  ║")
    print("║    Claude reasons about biomechanics — not about our equations.     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def run_insight(plot_path: str = None, model: str = _CLOUD_MODEL) -> dict:
    """Send DERIVED-OK plot + PUBLIC primer to Claude for physics insight.

    Args:
        plot_path: path to comparison PNG (DERIVED-OK). If None, uses latest.
        model:     Claude model ID.

    Returns:
        {
          "model":          cloud model used,
          "insight":        physics insight text (PUBLIC),
          "input_tokens":   tokens used in cloud call,
          "output_tokens":  tokens produced by cloud,
          "shielded_tokens": tokens that stayed local (never sent),
          "plot_path":      path to the plot that was analysed,
        }
    """
    import anthropic

    # ── Resolve plot path ─────────────────────────────────────────────────────
    if plot_path is None:
        plots_dir = _REPO_ROOT / "docs" / "executive_branch_document" / "plots"
        candidates = sorted(plots_dir.glob("model_compare_*.png"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                "No model_compare_*.png found. Run run_model_compare.py first."
            )
        plot_path = str(candidates[0])

    primer_tokens = _count_primer_tokens()
    _print_hybrid_header(plot_path, primer_tokens)

    # ── Build API request — DERIVED-OK + PUBLIC only ──────────────────────────
    image_b64 = _load_plot_as_b64(plot_path)

    system_prompt = (
        "You are a physics reviewer for a gait analysis wearable device. "
        "You receive signal comparison plots and provide concise physics-grounded insights. "
        "You have access only to the plot image and a public physics primer. "
        "You do NOT have access to source code, derivation equations, or model weights. "
        "Ground all observations in the three gait primitives: "
        "Vertical Oscillation, Cadence, and Step Length."
    )

    user_message = (
        f"Physics Primer:\n{_PHYSICS_PRIMER}\n\n"
        "---\n"
        "Review the attached comparison plot. It shows:\n"
        "  - GT (blue dashed): walker_model ground truth signal for each profile\n"
        "  - Good PINN (green): trained neural network prediction\n"
        "  - Bad model (red): randomly initialized neural network (untrained)\n\n"
        "For each of the 4 gait profiles (flat, stairs, slope, bad_wear), "
        "provide a brief physics insight (2–3 sentences) explaining:\n"
        "  1. What specific gait primitive is NOT encoded in the bad model's signal\n"
        "  2. What the good model partially learned (even if amplitude is off)\n"
        "  3. Which clinical measurement would be wrong if the bad model were deployed\n\n"
        "End with one paragraph on what the shape difference tells us about "
        "the learning objective — what physics constraint was missing in the untrained model."
    )

    client = anthropic.Anthropic()

    print(f"[cloud agent] calling {model} — sending DERIVED-OK image + PUBLIC primer...")
    t0 = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": user_message,
                },
            ],
        }],
    )

    elapsed = time.time() - t0
    insight_text = response.content[0].text
    in_tok  = response.usage.input_tokens
    out_tok = response.usage.output_tokens

    # ── Print insight ──────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  CLOUD PHYSICS INSIGHT  [{model}]")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(insight_text)
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  TOKEN ACCOUNTING SUMMARY                                           ║")
    print(f"║    Cloud model:            {model:<43}║")
    print(f"║    Call time:              {elapsed:>5.1f}s                                   ║")
    print(f"║    Input tokens (sent):    {in_tok:>6}  (image + primer)              ║")
    print(f"║    Output tokens:          {out_tok:>6}  (insight text)                ║")
    print(f"║    ─────────────────────────────────────────────────────           ║")
    print(f"║    PRIVATE tokens shielded: {_TOTAL_SHIELDED:>6}  (never left local)          ║")
    ratio = _TOTAL_SHIELDED / max(in_tok, 1)
    print(f"║    Shield ratio:           {ratio:>5.1f}×  (private/sent)               ║")
    print(f"║    ─────────────────────────────────────────────────────           ║")
    print(f"║    IP protection: walker_model derivation stayed LOCAL             ║")
    print(f"║    Cloud received: SHAPES only — no equations, no weights          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    return {
        "model":           model,
        "insight":         insight_text,
        "input_tokens":    in_tok,
        "output_tokens":   out_tok,
        "shielded_tokens": _TOTAL_SHIELDED,
        "shield_ratio":    round(ratio, 1),
        "call_time_s":     round(elapsed, 2),
        "plot_path":       plot_path,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid demo step 2 — cloud physics insight from DERIVED-OK plot"
    )
    parser.add_argument("--plot",  default=None,
                        help="Path to comparison PNG (default: latest model_compare_*.png)")
    parser.add_argument("--model", default=_CLOUD_MODEL,
                        help=f"Claude model ID (default: {_CLOUD_MODEL})")
    args = parser.parse_args()

    result = run_insight(plot_path=args.plot, model=args.model)

    print("\n[cloud agent] DERIVED-OK result dict:")
    safe = {k: v for k, v in result.items() if k != "insight"}
    print(json.dumps(safe, indent=2))
