"""
Fixed plotter script — no LLM required.

Skill contract:
  execution:         local (deterministic — no model call)
  retrieves PRIVATE: simulator/walker_model.py, signal arrays
  receives PUBLIC:   profile name, mode
  produces DERIVED-OK: signal plot PNG (file), data table (stdout)
  must_not_forward:  raw signal arrays

Usage:
  python hybrid/scripts/plot_profile.py --profile flat
  python hybrid/scripts/plot_profile.py --profile stairs --mode pathological
  python hybrid/scripts/plot_profile.py --profile bad_wear --open
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
SIM_DIR    = REPO_ROOT / "simulator"
PLOTS_DIR  = REPO_ROOT / "docs" / "executive_branch_document" / "plots"
sys.path.insert(0, str(SIM_DIR))

from walker_model import generate_imu_sequence, PROFILES  # PRIVATE — stays local

# ── firmware-matched filter constants (Amendment 11) ─────────────────────────
ODR_HZ        = 208.0
LP_CUTOFF_HZ  = 15.0   # lowpass on acc_z — matches firmware phase_segmenter
HP_CUTOFF_HZ  = 0.5    # highpass on gyr_y — matches firmware step_detector
ACC_THRESHOLD = 5.0    # m/s² adaptive detection threshold (Amendment 15)
GYR_THRESHOLD = 30.0   # dps gyr_y zero-crossing gate (Amendment 15)


def butter_filter(data: np.ndarray, cutoff: float, fs: float,
                  btype: str, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype=btype)
    return filtfilt(b, a, data)


def plot_profile(profile_name: str, mode: str, open_after: bool = False) -> dict:
    """
    Generate 3-panel IMU diagnostic plot. Returns DERIVED-OK result dict.
    Raw signal arrays are not included in the return value.
    """
    if profile_name not in PROFILES:
        print(f"[plot_profile] ERROR: unknown profile '{profile_name}'")
        print(f"[plot_profile] valid: {list(PROFILES.keys())}")
        sys.exit(1)

    profile = PROFILES[profile_name]

    # pathological mode: override si_stance_true_pct on a copy of the profile
    if mode == "pathological":
        from dataclasses import replace
        profile = replace(profile, si_stance_true_pct=25.0)

    # ── generate signal (PRIVATE — not forwarded) ─────────────────────────────
    data = generate_imu_sequence(profile, n_steps=20)
    # data: (N, 6)  cols: ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps

    t_ms = np.arange(len(data)) * (1000.0 / ODR_HZ)

    acc_z_raw  = data[:, 2]
    gyr_y_raw  = data[:, 4]

    acc_z_filt = butter_filter(acc_z_raw, LP_CUTOFF_HZ, ODR_HZ, "low")
    gyr_y_filt = butter_filter(gyr_y_raw, HP_CUTOFF_HZ, ODR_HZ, "high")

    # simple peak detector on acc_z_filt for step markers
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(acc_z_filt, height=ACC_THRESHOLD, distance=int(ODR_HZ * 0.3))

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"GaitSense IMU Diagnostic — profile={profile_name}  mode={mode}",
        fontsize=13, fontweight="bold"
    )

    # Panel 1 — acc_z raw + filtered
    ax1 = axes[0]
    ax1.plot(t_ms, acc_z_raw,  color="lightgrey", lw=0.8, label="acc_z raw")
    ax1.plot(t_ms, acc_z_filt, color="steelblue", lw=1.2, label="acc_z filtered (15 Hz LP)")
    ax1.axhline(ACC_THRESHOLD,  color="red",    lw=0.8, ls="--",
                label=f"{ACC_THRESHOLD} m/s² threshold (Amendment 15)")
    ax1.axhline(-ACC_THRESHOLD, color="red",    lw=0.8, ls="--")
    ax1.set_ylabel("m/s²")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title("acc_z — vertical acceleration (Amendment 11: 15 Hz LP)")

    # Panel 2 — gyr_y raw + HP filtered
    ax2 = axes[1]
    ax2.plot(t_ms, gyr_y_raw,  color="wheat",  lw=0.8, label="gyr_y raw")
    ax2.plot(t_ms, gyr_y_filt, color="darkorange", lw=1.2, label="gyr_y filtered (0.5 Hz HP)")
    ax2.axhline(GYR_THRESHOLD,  color="purple", lw=0.8, ls="--",
                label=f"{GYR_THRESHOLD} dps gate (Amendment 15)")
    ax2.axhline(-GYR_THRESHOLD, color="purple", lw=0.8, ls="--")
    ax2.set_ylabel("dps")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.set_title("gyr_y — sagittal angular velocity (Amendment 11: 0.5 Hz HP)")

    # Panel 3 — acc_z filtered with step markers
    ax3 = axes[2]
    ax3.plot(t_ms, acc_z_filt, color="steelblue", lw=1.2)
    ax3.scatter(t_ms[peaks], acc_z_filt[peaks],
                color="red", s=40, zorder=5, label=f"{len(peaks)} steps detected")
    ax3.axhline(ACC_THRESHOLD, color="red", lw=0.8, ls="--",
                label=f"{ACC_THRESHOLD} m/s² (Amendment 15)")
    ax3.set_ylabel("m/s²")
    ax3.set_xlabel("time (ms)")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_title("Step detection — acc_z filtered + peak markers")

    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"{profile_name}_signal_check.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # ── DERIVED-OK data table (stdout) ────────────────────────────────────────
    az_peak_raw  = float(np.max(np.abs(acc_z_raw)))
    az_peak_filt = float(np.max(np.abs(acc_z_filt)))
    gy_peak      = float(np.max(np.abs(gyr_y_raw)))

    print("─────────────────────────────────────────────────────")
    print(f"SIGNAL DATA — profile={profile_name}  mode={mode}")
    print("─────────────────────────────────────────────────────")
    print(f"acc_z peak (raw):      {az_peak_raw:.2f} m/s²")
    print(f"acc_z peak (filtered): {az_peak_filt:.2f} m/s²")
    print(f"gyr_y peak:            {gy_peak:.2f} dps")
    print(f"steps detected:        {len(peaks)}")
    print(f"cadence (spm):         {profile.cadence_spm:.1f}")
    print(f"vert. oscillation:     {profile.vertical_oscillation_cm:.1f} cm")
    print("─────────────────────────────────────────────────────")
    print(f"PLOT_PATH: {out_path}")

    if open_after or os.environ.get("GAITSENSE_DEMO") == "1":
        import subprocess
        subprocess.run(["open", str(out_path)])

    # DERIVED-OK — safe to forward to plot-orchestrator
    return {
        "status":         "PASS",
        "profile":        profile_name,
        "mode":           mode,
        "az_peak_raw":    az_peak_raw,
        "az_peak_filt":   az_peak_filt,
        "gy_peak":        gy_peak,
        "steps_detected": int(len(peaks)),
        "plot_path":      str(out_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaitSense fixed plotter — no LLM")
    parser.add_argument("--profile", required=True,
                        choices=list(PROFILES.keys()))
    parser.add_argument("--mode",    default="healthy",
                        choices=["healthy", "pathological"])
    parser.add_argument("--open",    action="store_true",
                        help="Open plot in Preview after saving (macOS)")
    parser.add_argument("--json",    action="store_true",
                        help="Print DERIVED-OK result as JSON at end")
    args = parser.parse_args()

    result = plot_profile(args.profile, args.mode, open_after=args.open)

    if args.json:
        print("\n[plot_profile] DERIVED-OK result:")
        print(json.dumps(result, indent=2))
