"""
plot_model_compare.py — Good vs Bad PINN model comparison (hybrid demo).

Demonstrates the hybrid cloud-local intelligence architecture:
  - Local agent accesses PRIVATE sources: walker_model, checkpoint .pt files
  - Outputs DERIVED-OK: plot PNG (no raw formulas, no norm stats, no weights)
  - LLM parsed the request; Python did all PRIVATE file access

Comparison:
  "Good" model  — trained checkpoint (default: best_v1.pt, 490 epochs, val≈1.17)
  "Bad"  model  — randomly initialized PINNModel (0 epochs, pure noise)

Both are evaluated against the walker_model ground truth for all 4 profiles.
The plot shows gy (step push-off angular velocity) and az (vertical acceleration)
for one canonical step — the shape that the gait algorithm needs to detect steps.

Usage:
    python plot_model_compare.py
    python plot_model_compare.py good_run_id=v1 bad_run_id=random

Output:
    docs/executive_branch_document/plots/model_compare_<good>_vs_<bad>.png

Constitutional grounding:
    Amendment 11: signal plots mandatory at each stage gate
    Article II:   human reviews DERIVED-OK plot — no LLM interprets raw signals
    hybrid demo:  local agent → PRIVATE in → DERIVED-OK plot out → cloud/human
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from pinn_model import PINNModel, HIDDEN_DIM, N_LAYERS, FOURIER_DIM, INPUT_DIM
from walker_model import PROFILES, generate_imu_sequence

CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
PLOTS_DIR      = os.path.join(_HERE, "../../docs/executive_branch_document/plots")

PROFILES_ORDER = ["flat", "stairs", "slope", "bad_wear"]
T_STEPS        = 208
TERRAIN_INT    = {"flat": 0, "slope": 1, "stairs": 2, "bad_wear": 0}
OUTPUT_COLS    = ["ax_ms2", "ay_ms2", "az_ms2", "gx_dps", "gy_dps", "gz_dps"]
COL            = {name: i for i, name in enumerate(OUTPUT_COLS)}

# Gait algorithm thresholds (Amendment 13 — change requires a Bill)
PUSHOFF_DEFAULT_DPS = 80.0
ACC_Z_TOE_OFF       = 2.94


def _compute_norm_stats() -> tuple:
    """Derive Y_mean and Y_std from walker_model GT output across all 4 profiles.

    This is a local computation — accesses PRIVATE walker_model.
    Returns: (Y_mean, Y_std_safe) as np.ndarray (6,) each.
    """
    rng = np.random.default_rng(42)
    all_seqs = []
    for key in PROFILES_ORDER:
        profile = PROFILES[key]
        seq = generate_imu_sequence(profile, n_steps=20, rng=rng)
        stationary = T_STEPS
        all_seqs.append(seq[stationary:stationary + T_STEPS * 10])
    data = np.vstack(all_seqs).astype(np.float32)
    Y_mean     = data.mean(axis=0)
    Y_std_safe = np.where(data.std(axis=0) > 1e-6, data.std(axis=0), 1.0)
    return Y_mean, Y_std_safe


def _load_good_model(run_id: str):
    """Load trained checkpoint. Returns (model, Y_mean, Y_std_safe)."""
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"best_{run_id}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Try loading norm stats from file first; fall back to computing from GT
    norm_path = os.path.join(CHECKPOINT_DIR, f"Y_norm_stats_{run_id}.npy")
    if os.path.exists(norm_path):
        norm       = np.load(norm_path)
        Y_mean     = norm[0]
        Y_std_safe = norm[1]
    else:
        Y_mean, Y_std_safe = _compute_norm_stats()

    model = PINNModel(use_fourier=True)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, Y_mean, Y_std_safe


def _make_bad_model():
    """Return a randomly initialized (untrained) PINNModel with identity norms."""
    torch.manual_seed(0)
    model = PINNModel(use_fourier=True)
    model.eval()
    Y_mean     = np.zeros(6, dtype=np.float32)
    Y_std_safe = np.ones(6, dtype=np.float32)
    return model, Y_mean, Y_std_safe


def _pinn_predict(model, Y_mean, Y_std_safe, profile) -> np.ndarray:
    """Run model for one canonical step. Returns (T_STEPS, 6) in physical units."""
    x_vec = np.array([
        profile.cadence_spm,
        profile.step_length_m,
        profile.vertical_oscillation_cm,
        profile.slope_deg,
        profile.stance_frac,
        profile.si_stance_true_pct,
        profile.mounting_offset_deg,
        profile.loose_fit_attenuation,
        profile.step_variability_ms,
        TERRAIN_INT.get(profile.terrain, 0),
    ], dtype=np.float32)

    t = torch.linspace(0, 1, T_STEPS)
    x = torch.from_numpy(x_vec).unsqueeze(0).expand(T_STEPS, -1)
    with torch.no_grad():
        pred_norm = model(x, t).numpy()
    return pred_norm * Y_std_safe + Y_mean


def _gt_predict(profile, seed: int = 42) -> np.ndarray:
    """Ground truth from walker_model for one canonical step."""
    rng = np.random.default_rng(seed)
    seq = generate_imu_sequence(profile, n_steps=2, rng=rng)
    stationary = T_STEPS
    return seq[stationary:stationary + T_STEPS]


def run(good_run_id: str = "v1", bad_run_id: str = "random") -> dict:
    """Generate PINN good-vs-bad comparison plot.

    Args:
        good_run_id: run_id for the trained checkpoint (default: v1)
        bad_run_id:  "random" for untrained model, or a run_id for a checkpoint

    Returns:
        DERIVED-OK result dict (plot path + pass/fail table).
    """
    # ── Load models (PRIVATE access — local only) ─────────────────────────────
    good_model, good_mean, good_std = _load_good_model(good_run_id)

    if bad_run_id == "random":
        bad_model, bad_mean, bad_std = _make_bad_model()
        bad_label = "Untrained (random init)"
    else:
        bad_model, bad_mean, bad_std = _load_good_model(bad_run_id)
        bad_label = f"PINN {bad_run_id}"

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_profiles = len(PROFILES_ORDER)
    fig, axes  = plt.subplots(n_profiles, 2,
                              figsize=(14, 3.8 * n_profiles),
                              sharex=True)
    norm_note = ("GT norm stats (approx)" if not os.path.exists(
        os.path.join(CHECKPOINT_DIR, f"Y_norm_stats_{good_run_id}.npy")) else "saved norm stats")
    fig.suptitle(
        f"Hybrid Demo — PINN Model Comparison  [{norm_note}]\n"
        f"Good: PINN {good_run_id}  |  Bad: {bad_label}  |  GT: walker_model ground truth\n"
        f"Left: gy (dps)  |  Right: az (m/s²)  "
        f"[Threshold lines show gait algorithm detection boundaries — Amendment 13]",
        fontsize=11, fontweight="bold"
    )

    t_axis = np.linspace(0, 1, T_STEPS)
    summary_rows = []

    for row, key in enumerate(PROFILES_ORDER):
        profile = PROFILES[key]
        gt      = _gt_predict(profile)
        good    = _pinn_predict(good_model, good_mean, good_std, profile)
        bad     = _pinn_predict(bad_model,  bad_mean,  bad_std,  profile)

        gy_gt   = gt[:, COL["gy_dps"]]
        gy_good = good[:, COL["gy_dps"]]
        gy_bad  = bad[:,  COL["gy_dps"]]

        az_gt   = gt[:, COL["az_ms2"]]
        az_good = good[:, COL["az_ms2"]]
        az_bad  = bad[:,  COL["az_ms2"]]

        ax_gy = axes[row, 0]
        ax_az = axes[row, 1]

        # ── gy panel ─────────────────────────────────────────────────────────
        ax_gy.plot(t_axis, gy_gt,   color="steelblue",  lw=1.2, ls="--", alpha=0.8, label="GT (walker_model)")
        ax_gy.plot(t_axis, gy_good, color="green",      lw=1.8,          label=f"Good (PINN {good_run_id})")
        ax_gy.plot(t_axis, gy_bad,  color="red",        lw=1.2, alpha=0.7, label=bad_label)
        ax_gy.axhline( PUSHOFF_DEFAULT_DPS, color="darkorange", lw=1.0, ls="-.",
                       label=f"PUSHOFF threshold ±{PUSHOFF_DEFAULT_DPS:.0f} dps")
        ax_gy.axhline(-PUSHOFF_DEFAULT_DPS, color="darkorange", lw=1.0, ls="-.", alpha=0.5)
        ax_gy.axhline(0, color="gray", lw=0.5, ls=":")
        ax_gy.set_ylabel(f"{key}\ngy (dps)")
        ax_gy.legend(fontsize=7, loc="upper right")

        good_gy_ok = gy_good.max() >= PUSHOFF_DEFAULT_DPS
        bad_gy_ok  = gy_bad.max()  >= PUSHOFF_DEFAULT_DPS
        ax_gy.set_title(
            f"{key}  gy  |  Good peak: {gy_good.max():.1f} dps [{('PASS' if good_gy_ok else 'FAIL')}]"
            f"  |  Bad peak: {gy_bad.max():.1f} dps [{('PASS' if bad_gy_ok else 'FAIL')}]",
            fontsize=8,
            color="green" if (good_gy_ok and not bad_gy_ok) else
                  "orange" if (good_gy_ok and bad_gy_ok) else "red"
        )

        # ── az panel ─────────────────────────────────────────────────────────
        ax_az.plot(t_axis, az_gt,   color="steelblue",  lw=1.2, ls="--", alpha=0.8, label="GT (walker_model)")
        ax_az.plot(t_axis, az_good, color="green",      lw=1.8,          label=f"Good (PINN {good_run_id})")
        ax_az.plot(t_axis, az_bad,  color="red",        lw=1.2, alpha=0.7, label=bad_label)
        ax_az.axhline(ACC_Z_TOE_OFF, color="darkorange", lw=1.0, ls="-.",
                      label=f"ACC_Z_TOE_OFF={ACC_Z_TOE_OFF:.2f} m/s²")
        ax_az.axhline(9.81, color="gray", lw=0.5, ls=":", label="1g")
        ax_az.set_ylabel("az (m/s²)")
        ax_az.legend(fontsize=7, loc="upper right")

        good_az_ok = az_good.min() <= ACC_Z_TOE_OFF
        bad_az_ok  = az_bad.min()  <= ACC_Z_TOE_OFF
        ax_az.set_title(
            f"{key}  az  |  Good min: {az_good.min():.2f} m/s² [{('PASS' if good_az_ok else 'FAIL')}]"
            f"  |  Bad min: {az_bad.min():.2f} m/s² [{('PASS' if bad_az_ok else 'FAIL')}]",
            fontsize=8,
            color="green" if (good_az_ok and not bad_az_ok) else
                  "orange" if (good_az_ok and bad_az_ok) else "red"
        )

        summary_rows.append({
            "profile":      key,
            "good_gy_peak": float(gy_good.max()),
            "bad_gy_peak":  float(gy_bad.max()),
            "gt_gy_peak":   float(gy_gt.max()),
            "good_gy_ok":   good_gy_ok,
            "bad_gy_ok":    bad_gy_ok,
            "good_az_min":  float(az_good.min()),
            "bad_az_min":   float(az_bad.min()),
            "good_az_ok":   good_az_ok,
            "bad_az_ok":    bad_az_ok,
        })

    axes[-1, 0].set_xlabel("Normalised step time t  [0 = heel-strike, 1 = next heel-strike]")
    axes[-1, 1].set_xlabel("Normalised step time t  [0 = heel-strike, 1 = next heel-strike]")

    plt.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_name = f"model_compare_{good_run_id}_vs_{bad_run_id}.png"
    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    # ── stdout table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"HYBRID DEMO — Model Comparison: PINN {good_run_id}  vs  {bad_label}")
    print(f"{'═'*76}")
    print(f"{'Profile':<12}  {'GT gy':>8}  {'Good gy':>9} {'':>5}  {'Bad gy':>8} {'':>5}  "
          f"{'Good az':>9} {'':>5}  {'Bad az':>8} {'':>5}")
    print("─" * 76)
    for r in summary_rows:
        g_gy = "PASS" if r["good_gy_ok"] else "FAIL"
        b_gy = "PASS" if r["bad_gy_ok"]  else "FAIL"
        g_az = "PASS" if r["good_az_ok"] else "FAIL"
        b_az = "PASS" if r["bad_az_ok"]  else "FAIL"
        print(f"{r['profile']:<12}  {r['gt_gy_peak']:>8.1f}  {r['good_gy_peak']:>9.1f} {g_gy:>5}  "
              f"{r['bad_gy_peak']:>8.1f} {b_gy:>5}  {r['good_az_min']:>9.2f} {g_az:>5}  "
              f"{r['bad_az_min']:>8.2f} {b_az:>5}")
    print(f"{'═'*76}")

    good_passes = sum(r["good_gy_ok"] and r["good_az_ok"] for r in summary_rows)
    bad_passes  = sum(r["bad_gy_ok"]  and r["bad_az_ok"]  for r in summary_rows)
    print(f"\nGood model thresholds met: {good_passes}/{len(summary_rows)} profiles")
    print(f"Bad  model thresholds met: {bad_passes}/{len(summary_rows)} profiles")
    print(f"\nHybrid boundary: local agent accessed PRIVATE (walker_model, checkpoint)")
    print(f"Output tier: DERIVED-OK (plot PNG — safe to forward to Justice/cloud)")
    print(f"\nPlot saved: {out_path}")

    if os.environ.get("GAITSENSE_DEMO") == "1":
        os.system(f"open {out_path}")

    return {
        "good_run_id":  good_run_id,
        "bad_run_id":   bad_run_id,
        "good_passes":  int(good_passes),
        "bad_passes":   int(bad_passes),
        "n_profiles":   len(summary_rows),
        "plot_path":    out_path,
        "summary":      summary_rows,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--good", default="v1",     help="Good checkpoint run_id")
    parser.add_argument("--bad",  default="random", help="Bad checkpoint run_id, or 'random'")
    args = parser.parse_args()
    run(args.good, args.bad)
