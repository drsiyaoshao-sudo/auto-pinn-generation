"""
PINN checkpoint → four anchor profiles → Python simulation pipeline.

Loads a PINN checkpoint, generates one canonical step per profile,
tiles it n_steps times with light sensor noise, prepends a 1-second
stationary prefix, then runs gait_algorithm.run().

Reports:
  - step count vs expected
  - mean/max SI_stance (expected ≈ 0 for all four profiles)

PINN_GYR_CONFIRM_MS: overrides GYR_CONFIRM_MS for PINN runs.
  The PINN signal has a broader az peak and gy zero-crossing is
  further from the az peak than in walker_model, so a wider window
  is needed. Set to 10× the default (40ms → 400ms) as a permissive
  trial to confirm whether step shape is sufficient for detection.
"""

import sys
import os
import numpy as np
import torch

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMULATOR_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, SIMULATOR_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pinn_model import PINNModel
from walker_model import PROFILES, ODR_HZ, WalkerProfile
import gait_algorithm
from gait_algorithm import run as run_algorithm

CKPT_DIR  = os.path.join(SCRIPT_DIR, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "best_v21.pt")
NORM_PATH = os.path.join(CKPT_DIR, "Y_norm_stats_v20.npy")

# ── Relaxed thresholds for PINN signal (broader, lower-amplitude vs walker_model) ──
# GYR_CONFIRM_MS: gy zero-crossing window after az peak.
# THRESHOLD_SEED: initial az detection threshold. PINN az LP peak is smaller → lower to 2.0.
# PUSHOFF_DEFAULT_DPS: PhaseSegmenter terminal→toe_off gy threshold. Default 80 dps;
#   v21 gy max is ~42 dps so lower to 20 (half of v21 max) to allow phase progression.
# ACC_Z_TOE_OFF: fallback toe-off az threshold. Default 2.94 m/s²; v21 az min ~5 m/s²
#   so raise to 4.5 so the fallback can also trigger.
PINN_GYR_CONFIRM_MS  = 400    # default 40ms
PINN_THRESHOLD_SEED  = 2.0    # default 7.0
PINN_PUSHOFF_DPS     = 20.0   # default 80.0  — below v21's ~42 dps gy max
PINN_ACC_Z_TOE_OFF   = 4.5    # default 2.94  — below v21's ~5 m/s² az min

TERRAIN_INT = {"flat": 0, "slope": 1, "stairs": 2}
T_STEPS     = 208          # samples per step (= ODR_HZ / ~1 Hz cadence window)
N_STEPS     = 200          # gait steps to simulate
ODR         = int(ODR_HZ)  # 208

ANCHOR_KEYS = ["flat", "bad_wear", "stairs", "slope"]


def profile_to_x(profile: WalkerProfile) -> np.ndarray:
    return np.array([
        profile.cadence_spm,
        profile.step_length_m,
        profile.vertical_oscillation_cm,
        profile.slope_deg,
        profile.stance_frac,
        profile.si_stance_true_pct,
        profile.mounting_offset_deg,
        profile.loose_fit_attenuation,
        profile.step_variability_ms,
        TERRAIN_INT[profile.terrain],
    ], dtype=np.float32)


def load_model():
    model = PINNModel(hidden_dim=16, fourier_dim=24, n_layers=4, use_fourier=False,
                      use_cheby=True, cheby_degree=5)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {CKPT_PATH}  ({model.count_parameters():,} params)")
    return model


def pinn_canonical_step(model, x_vec: np.ndarray, Y_mean, Y_std_safe) -> np.ndarray:
    """
    Run PINN forward for one step → (T_STEPS, 6) denormalised IMU in physical units.

    x_vec: (10,) float32  conditioning vector
    Returns: np.ndarray shape (T_STEPS, 6)  [ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]
    """
    t   = torch.linspace(0, 1, T_STEPS)          # (T_STEPS,)
    x   = torch.from_numpy(x_vec).unsqueeze(0)    # (1, 10)
    x   = x.expand(T_STEPS, -1)                  # (T_STEPS, 10)

    with torch.no_grad():
        pred_norm = model(x, t)                   # (T_STEPS, 6)  normalised

    pred_norm = pred_norm.numpy()
    # Denormalise: Y_raw = pred_norm * Y_std_safe + Y_mean
    pred_raw  = pred_norm * Y_std_safe + Y_mean   # (T_STEPS, 6)
    return pred_raw.astype(np.float32)


def build_signal(canonical_step: np.ndarray, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Tile canonical_step n_steps times with sensor-level noise.
    Prepend a 1-second (ODR samples) zero-mean stationary prefix.
    Returns: (ODR + n_steps*T_STEPS, 6) float32
    """
    # Sensor noise levels (LSM6DS3TR-C, from walker_model.py)
    import math
    G = 9.81
    accel_noise_rms = (90e-6 * G) * math.sqrt(ODR / 2)
    gyro_noise_rms  = 4e-3 * math.sqrt(ODR / 2)
    noise_rms = np.array([accel_noise_rms]*3 + [gyro_noise_rms]*3, dtype=np.float32)

    walking_tiles = []
    for _ in range(n_steps):
        step = canonical_step + rng.standard_normal((T_STEPS, 6)).astype(np.float32) * noise_rms
        walking_tiles.append(step)

    walking = np.concatenate(walking_tiles, axis=0)  # (n_steps*T_STEPS, 6)

    # Stationary prefix: only gravity on az ≈ 9.81 m/s²
    stat = np.zeros((ODR, 6), dtype=np.float32)
    stat[:, 2] = 9.81  # az gravity
    stat += (rng.standard_normal((ODR, 6)) * noise_rms).astype(np.float32)

    return np.concatenate([stat, walking], axis=0)


def expected_steps(profile: WalkerProfile) -> float:
    """Approx steps expected from n_steps=N_STEPS (each step is one canonical window)."""
    return N_STEPS


def main():
    # Inject relaxed thresholds for PINN signal
    gait_algorithm.StepDetector.GYR_CONFIRM_MS        = PINN_GYR_CONFIRM_MS
    gait_algorithm.StepDetector.THRESHOLD_SEED         = PINN_THRESHOLD_SEED
    gait_algorithm.PhaseSegmenter.PUSHOFF_DEFAULT_DPS  = PINN_PUSHOFF_DPS
    gait_algorithm.PhaseSegmenter.ACC_Z_TOE_OFF        = PINN_ACC_Z_TOE_OFF

    print("=" * 65)
    print("v21 PINN → Four Anchor Profiles → Python Gait Algorithm")
    print(f"  GYR_CONFIRM_MS={PINN_GYR_CONFIRM_MS}ms  THRESHOLD_SEED={PINN_THRESHOLD_SEED}")
    print("=" * 65)

    # Load model and normalisation stats
    print("\n[1] Loading model and norm stats...")
    model = load_model()
    norm_stats  = np.load(NORM_PATH)    # (2, 6): [Y_mean, Y_std_safe]
    Y_mean      = norm_stats[0]         # (6,)
    Y_std_safe  = norm_stats[1]         # (6,)
    print(f"  Y_mean:     {Y_mean}")
    print(f"  Y_std_safe: {Y_std_safe}")

    print()
    print("[2] Running per-profile inference + gait algorithm...")
    print()
    print(f"{'Profile':<12} {'Steps(PINN)':<14} {'Steps(Exp)':<12} {'SI_mean%':<11} {'SI_max%':<10} {'Status'}")
    print("-" * 65)

    rng = np.random.default_rng(42)

    for key in ANCHOR_KEYS:
        profile = PROFILES[key]
        x_vec   = profile_to_x(profile)

        # Generate canonical step from PINN
        canonical = pinn_canonical_step(model, x_vec, Y_mean, Y_std_safe)

        # Build multi-step signal
        signal = build_signal(canonical, N_STEPS, rng)   # (ODR + N_STEPS*T_STEPS, 6)

        # Run gait algorithm
        steps, snapshots, records = run_algorithm(signal)

        n_detected = len(steps)
        si_vals    = [s.si_stance_pct for s in snapshots]
        si_mean    = sum(si_vals) / len(si_vals) if si_vals else float("nan")
        si_max     = max(si_vals) if si_vals else float("nan")

        step_ok = abs(n_detected - N_STEPS) / N_STEPS < 0.15
        si_ok   = si_max < 3.0

        status = ""
        if step_ok and si_ok:
            status = "PASS"
        else:
            parts = []
            if not step_ok:
                parts.append(f"step_count_err={100*(n_detected-N_STEPS)/N_STEPS:+.0f}%")
            if not si_ok:
                parts.append(f"SI_max={si_max:.1f}%>3%")
            status = "FAIL: " + ", ".join(parts)

        print(f"{key:<12} {n_detected:<14} {N_STEPS:<12} {si_mean:>7.2f}%    {si_max:>6.2f}%    {status}")

        # Canonical step stats for diagnostics
        print(f"  canonical step stats:  az=[{canonical[:,2].min():.2f}, {canonical[:,2].max():.2f}] m/s²"
              f"   gy=[{canonical[:,4].min():.2f}, {canonical[:,4].max():.2f}] dps")

    print()
    print("[3] Done.")


if __name__ == "__main__":
    main()
