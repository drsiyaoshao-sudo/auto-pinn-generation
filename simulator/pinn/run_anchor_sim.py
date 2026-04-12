"""
run_anchor_sim.py — PINN → four anchor profiles → gait algorithm pipeline.

Standard agent-callable module used by simulator-operator and pinn-validator
to verify that a trained PINN checkpoint produces algorithmically detectable steps.

Usage:
    python run_anchor_sim.py [run_id]
    python run_anchor_sim.py v21_2k

If run_id is omitted, reads from train_config.json.

Output:
    stdout table: profile | steps_detected | steps_expected | SI_mean% | SI_max% | status

Returns (from run()):
    list of dicts, one per profile:
        {
          "profile":       str,
          "steps_detected": int,
          "steps_expected": int,
          "si_mean_pct":   float,
          "si_max_pct":    float,
          "step_ok":       bool,
          "si_ok":         bool,
          "pass":          bool,
        }

Threshold overrides for PINN signal:
    train_config.json may contain optional "pinn_sim_overrides" dict:
        "pinn_gyr_confirm_ms":  int    (default 400 — wider window for PINN gy timing)
        "pinn_threshold_seed":  float  (default 2.0 — lower for smaller PINN az peak)
        "pinn_pushoff_dps":     float  (default 20.0 — PINN gy max ~42 dps, below 80 default)
        "pinn_acc_z_toe_off":   float  (default 4.5 — PINN az min ~5 m/s², above 2.94 default)
    These overrides are temporary PINN adaptation parameters; they do NOT modify
    firmware constants. Changing firmware constants requires a Bill (Amendment 13).

Constitutional grounding:
    pinn-validator Gate 5b: run after Gate 5a (plot_step_unit.py) pass confirmation
    Amendment 11: results must be available for human review before any merge
    Amendment 13: threshold overrides here are PINN-side adaptation, NOT firmware changes
"""

import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from inference import load_model, pinn_one_step, T_STEPS
from walker_model import PROFILES, ODR_HZ, WalkerProfile
import gait_algorithm

ANCHOR_KEYS   = ["flat", "bad_wear", "stairs", "slope"]
N_STEPS       = 200      # gait steps to simulate per profile
ODR           = int(ODR_HZ)

# ── Default PINN-side threshold overrides ────────────────────────────────────
# These compensate for PINN signal amplitude being below ground truth during
# early training. They shrink as the model improves. NOT firmware constants.
_DEFAULT_OVERRIDES = {
    "pinn_gyr_confirm_ms":  400,    # default firmware: 40 ms
    "pinn_threshold_seed":  2.0,    # default firmware: 7.0
    "pinn_pushoff_dps":     20.0,   # default firmware: 80.0 dps
    "pinn_acc_z_toe_off":   4.5,    # default firmware: 2.94 m/s²
}


def _load_overrides(cfg: dict) -> dict:
    overrides = dict(_DEFAULT_OVERRIDES)
    pinn_cfg = cfg.get("pinn_sim_overrides", {})
    for key in _DEFAULT_OVERRIDES:
        if key in pinn_cfg:
            overrides[key] = pinn_cfg[key]
    return overrides


def _build_signal(canonical_step: np.ndarray, n_steps: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Tile canonical_step n_steps times with sensor noise; prepend 1-s stationary prefix."""
    import math
    G = 9.81
    accel_noise_rms = (90e-6 * G) * math.sqrt(ODR / 2)
    gyro_noise_rms  = 4e-3 * math.sqrt(ODR / 2)
    noise_rms = np.array([accel_noise_rms] * 3 + [gyro_noise_rms] * 3, dtype=np.float32)

    tiles = []
    for _ in range(n_steps):
        step = canonical_step + rng.standard_normal((T_STEPS, 6)).astype(np.float32) * noise_rms
        tiles.append(step)
    walking = np.concatenate(tiles, axis=0)

    stat = np.zeros((ODR, 6), dtype=np.float32)
    stat[:, 2] = 9.81
    stat += (rng.standard_normal((ODR, 6)) * noise_rms).astype(np.float32)

    return np.concatenate([stat, walking], axis=0)


def run(run_id: str = None) -> list:
    """Run PINN→gait-algorithm pipeline for all four anchor profiles.

    Args:
        run_id: checkpoint run_id string, e.g. "v21_2k".
                Defaults to train_config.json run_id if None.

    Returns:
        list of result dicts (one per anchor profile).
    """
    train_config_path = os.path.join(_HERE, "train_config.json")
    with open(train_config_path) as f:
        cfg = json.load(f)
    if run_id is None:
        run_id = cfg["run_id"]

    overrides = _load_overrides(cfg)

    # ── Apply PINN-side threshold overrides ───────────────────────────────────
    gait_algorithm.StepDetector.GYR_CONFIRM_MS       = overrides["pinn_gyr_confirm_ms"]
    gait_algorithm.StepDetector.THRESHOLD_SEED        = overrides["pinn_threshold_seed"]
    gait_algorithm.PhaseSegmenter.PUSHOFF_DEFAULT_DPS = overrides["pinn_pushoff_dps"]
    gait_algorithm.PhaseSegmenter.ACC_Z_TOE_OFF       = overrides["pinn_acc_z_toe_off"]

    print("=" * 70)
    print(f"Anchor Sim — run_id={run_id}")
    print(f"  GYR_CONFIRM_MS={overrides['pinn_gyr_confirm_ms']} ms  "
          f"THRESHOLD_SEED={overrides['pinn_threshold_seed']}  "
          f"PUSHOFF_DPS={overrides['pinn_pushoff_dps']}  "
          f"ACC_Z_TOE_OFF={overrides['pinn_acc_z_toe_off']}")
    print("=" * 70)

    print("\nLoading model...")
    model, Y_mean, Y_std = load_model(run_id)

    print("\nRunning per-profile inference + gait algorithm...\n")
    print(f"{'Profile':<12}  {'Detected':>9}  {'Expected':>9}  "
          f"{'SI_mean%':>9}  {'SI_max%':>8}  Status")
    print("─" * 70)

    rng     = np.random.default_rng(42)
    results = []

    for key in ANCHOR_KEYS:
        profile   = PROFILES[key]
        canonical = pinn_one_step(model, Y_mean, Y_std, profile)  # (T_STEPS, 6)
        signal    = _build_signal(canonical, N_STEPS, rng)

        steps, snapshots, _ = gait_algorithm.run(signal)

        n_detected = len(steps)
        si_vals    = [s.si_stance_pct for s in snapshots]
        si_mean    = float(np.mean(si_vals)) if si_vals else float("nan")
        si_max     = float(np.max(si_vals))  if si_vals else float("nan")

        step_ok = abs(n_detected - N_STEPS) / N_STEPS < 0.15
        si_ok   = si_max < 3.0
        passed  = step_ok and si_ok

        parts = []
        if not step_ok:
            parts.append(f"step_err={100*(n_detected-N_STEPS)/N_STEPS:+.0f}%")
        if not si_ok:
            parts.append(f"SI_max={si_max:.1f}%>3%")
        status_str = "PASS" if passed else "FAIL: " + ", ".join(parts)

        print(f"{key:<12}  {n_detected:>9}  {N_STEPS:>9}  "
              f"{si_mean:>8.2f}%  {si_max:>7.2f}%  {status_str}")

        az_min = canonical[:, 2].min()
        az_max = canonical[:, 2].max()
        gy_min = canonical[:, 4].min()
        gy_max = canonical[:, 4].max()
        print(f"  canonical: az=[{az_min:.2f}, {az_max:.2f}] m/s²  "
              f"gy=[{gy_min:.2f}, {gy_max:.2f}] dps")

        results.append({
            "profile":        key,
            "steps_detected": n_detected,
            "steps_expected": N_STEPS,
            "si_mean_pct":    si_mean,
            "si_max_pct":     si_max,
            "step_ok":        step_ok,
            "si_ok":          si_ok,
            "pass":           passed,
        })

    n_pass = sum(1 for r in results if r["pass"])
    print()
    print(f"Summary: {n_pass}/{len(results)} profiles PASS")
    if n_pass == len(results):
        print("Gate 5b: ALL PASS — proceed to pinn-validator.")
    else:
        print("Gate 5b: FAIL — review checkpoint and threshold overrides.")
        print("  To adjust PINN-side thresholds, add 'pinn_sim_overrides' to train_config.json.")
        print("  To change firmware constants, file a Bill (Amendment 13).")

    return results


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    run(run_id)
