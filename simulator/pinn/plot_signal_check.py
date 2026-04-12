"""
plot_signal_check.py — walker_model ground truth signal diagnostic.

Called by the plotter agent via /plot-profile and at Amendment 11 gates.
Shows raw and filtered IMU signals from walker_model for a single profile.
No PINN involvement — pure physics model output.

Usage:
    python plot_signal_check.py <profile> [mode]
    python plot_signal_check.py flat
    python plot_signal_check.py bad_wear pathological

Arguments:
    profile:  flat | stairs | slope | bad_wear
    mode:     healthy (default) | pathological

Output:
    docs/executive_branch_document/plots/<profile>_signal_check.png

3 panels:
    1. az (m/s²) — raw + 15 Hz LP filtered
    2. gy (dps)  — raw + zero-crossing markers
    3. ax (m/s²) — raw (lateral sway)

Constitutional grounding:
    Amendment 11: mandatory after any change to walker_model.py or algorithm parameters
    Bureaucracy Signal Plotting Standing Order: no Bill or hearing required
    /plot-profile skill: dispatches this module via plotter agent
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

from walker_model import PROFILES, WalkerProfile, generate_imu_sequence, ODR_HZ

PLOTS_DIR = os.path.join(_HERE, "../../docs/executive_branch_document/plots")

# ── Firmware-matched filter (15 Hz LP, matching phase_segmenter) ─────────────
def _lp15(signal: np.ndarray, fs: float = ODR_HZ) -> np.ndarray:
    b, a = butter(2, 15.0 / (fs / 2), btype="low")
    return filtfilt(b, a, signal)

# ── Channel indices (matches walker_model output order) ──────────────────────
COL = {"ax": 0, "ay": 1, "az": 2, "gx": 3, "gy": 4, "gz": 5}

N_STEPS    = 6     # generate 6 steps, show middle 4 (avoid edge effects)
SKIP_STEPS = 1     # skip first step (stationary prefix artefact)


def _make_pathological(profile: WalkerProfile) -> WalkerProfile:
    """Return a copy of profile with si_stance_true_pct=25 (pathological walker)."""
    import dataclasses
    return dataclasses.replace(profile, si_stance_true_pct=25.0)


def run(profile_name: str, mode: str = "healthy"):
    if profile_name not in PROFILES:
        print(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")
        sys.exit(1)

    profile = PROFILES[profile_name]
    if mode == "pathological":
        profile = _make_pathological(profile)

    rng = np.random.default_rng(42)
    seq = generate_imu_sequence(profile, n_steps=N_STEPS + 1, rng=rng)

    # strip stationary prefix (1 s = ODR_HZ samples) then skip first step
    step_samples = int(ODR_HZ)
    walking = seq[step_samples + SKIP_STEPS * step_samples:]
    n_show  = min(4 * step_samples, len(walking))
    data    = walking[:n_show]

    t_axis = np.arange(n_show) / ODR_HZ   # seconds

    az_raw = data[:, COL["az"]]
    gy_raw = data[:, COL["gy"]]
    ax_raw = data[:, COL["ax"]]
    az_lp  = _lp15(az_raw)

    # zero-crossings in gy
    zc_idx = np.where(np.diff(np.sign(gy_raw)))[0]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"Signal Check — profile={profile_name}  mode={mode}\n"
        f"cadence={profile.cadence_spm:.0f} spm  "
        f"step_len={profile.step_length_m:.2f} m  "
        f"vert_osc={profile.vertical_oscillation_cm:.1f} cm  "
        f"terrain={profile.terrain}",
        fontsize=11, fontweight="bold"
    )

    # ── Panel 1: az ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_axis, az_raw, color="lightsteelblue", lw=0.8, alpha=0.7, label="az raw")
    ax.plot(t_axis, az_lp,  color="steelblue",      lw=1.5, label="az 15 Hz LP")
    ax.axhline(9.81, color="gray", lw=0.6, ls=":", label="1g")
    ax.set_ylabel("az (m/s²)")
    ax.set_title(f"az — range [{az_raw.min():.2f}, {az_raw.max():.2f}] m/s²")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: gy ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_axis, gy_raw, color="tomato", lw=1.3, label="gy raw")
    ax.scatter(t_axis[zc_idx], np.zeros(len(zc_idx)),
               s=18, color="black", zorder=5, label=f"zero-crossings ({len(zc_idx)})")
    ax.axhline(0,    color="gray",   lw=0.6, ls=":")
    ax.axhline( 80,  color="red",    lw=1.0, ls="-.", alpha=0.6,
                label="PUSHOFF_DEFAULT_DPS=80 (PhaseSegmenter)")
    ax.axhline(-80,  color="red",    lw=1.0, ls="-.", alpha=0.3)
    ax.set_ylabel("gy (dps)")
    ax.set_title(f"gy — range [{gy_raw.min():.1f}, {gy_raw.max():.1f}] dps  "
                 f"zero-crossings={len(zc_idx)}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: ax ──────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_axis, ax_raw, color="mediumpurple", lw=1.2, label="ax raw (lateral sway)")
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("ax (m/s²)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"ax lateral sway — range [{ax_raw.min():.2f}, {ax_raw.max():.2f}] m/s²")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    suffix = "" if mode == "healthy" else f"_{mode}"
    out = os.path.join(PLOTS_DIR, f"{profile_name}{suffix}_signal_check.png")
    plt.savefig(out, dpi=150)
    plt.close()

    # ── stdout table ─────────────────────────────────────────────────────────
    print(f"\nSignal Check — profile={profile_name}  mode={mode}")
    print(f"  az: min={az_raw.min():.3f}  max={az_raw.max():.3f}  mean={az_raw.mean():.3f} m/s²")
    print(f"  gy: min={gy_raw.min():.1f}  max={gy_raw.max():.1f}  zero-crossings={len(zc_idx)}")
    print(f"  ax: min={ax_raw.min():.3f}  max={ax_raw.max():.3f} m/s²")
    print(f"\nPlot saved: {out}")

    if os.environ.get("GAITSENSE_DEMO") == "1":
        os.system(f"open {out}")


if __name__ == "__main__":
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "flat"
    mode         = sys.argv[2] if len(sys.argv) > 2 else "healthy"
    run(profile_name, mode)
