"""
plot_step_unit.py — Gate 5a signal sanity plot.

Shows PINN-predicted gy and az for one canonical step per profile,
with gait algorithm threshold lines overlaid. Used by the plotter agent
at Gate 5a to let the Justice verify that PINN signal amplitudes are
compatible with algorithm thresholds BEFORE any simulation run.

Usage:
    python plot_step_unit.py [run_id]
    python plot_step_unit.py v21_2k

If run_id is omitted, reads from train_config.json.

Output:
    docs/executive_branch_document/plots/step_unit_<run_id>.png

Constitutional grounding:
    Gate 5a (model-train.md): mandatory after every training run, before pinn-validator
    Amendment 11: signal plots mandatory at each stage gate
    Amendment 13: threshold values are algorithm parameters — any change requires a Bill
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from inference import load_model, pinn_one_step, gt_one_step, T_STEPS, COL
from walker_model import PROFILES

# ── Gait algorithm thresholds (from PhaseSegmenter defaults) ──────────────────
# Change here requires a firmware Bill (Amendment 13)
PUSHOFF_DEFAULT_DPS  = 80.0    # PhaseSegmenter.PUSHOFF_DEFAULT_DPS
ACC_Z_TOE_OFF        = 2.94    # PhaseSegmenter.ACC_Z_TOE_OFF (m/s²)

PROFILES_ORDER = ["flat", "stairs", "slope", "bad_wear"]
PLOTS_DIR = os.path.join(_HERE, "../../docs/executive_branch_document/plots")

COLORS = {"pinn": "tomato", "gt": "steelblue"}


def _threshold_label(val, name, source):
    return f"{name}={val} ({source})"


def run(run_id: str | None = None):
    model, Y_mean, Y_std = load_model(run_id)

    # resolve actual run_id for filename
    if run_id is None:
        with open(os.path.join(_HERE, "train_config.json")) as f:
            run_id = json.load(f)["run_id"]

    t_axis = np.linspace(0, 1, T_STEPS)

    fig, axes = plt.subplots(len(PROFILES_ORDER), 2,
                             figsize=(14, 3.8 * len(PROFILES_ORDER)),
                             sharex=True)
    fig.suptitle(
        f"Gate 5a — Signal Sanity: PINN step unit vs algorithm thresholds\n"
        f"run_id={run_id}  |  left: gy (dps)  |  right: az (m/s²)",
        fontsize=12, fontweight="bold"
    )

    sanity_rows = []

    for row, key in enumerate(PROFILES_ORDER):
        profile = PROFILES[key]
        pred    = pinn_one_step(model, Y_mean, Y_std, profile)
        gt      = gt_one_step(profile)

        gy_pred = pred[:, COL["gy_dps"]]
        az_pred = pred[:, COL["az_ms2"]]
        gy_gt   = gt[:, COL["gy_dps"]]
        az_gt   = gt[:, COL["az_ms2"]]

        ax_gy = axes[row, 0]
        ax_az = axes[row, 1]

        # ── gy panel ──
        ax_gy.plot(t_axis, gy_pred, color=COLORS["pinn"], lw=1.5, label="PINN")
        ax_gy.plot(t_axis, gy_gt,   color=COLORS["gt"],   lw=1.0, ls="--", alpha=0.6, label="GT")
        ax_gy.axhline( PUSHOFF_DEFAULT_DPS, color="red", lw=1.2, ls="-.",
                       label=_threshold_label(PUSHOFF_DEFAULT_DPS, "PUSHOFF_DPS", "PhaseSegmenter"))
        ax_gy.axhline(-PUSHOFF_DEFAULT_DPS, color="red", lw=1.2, ls="-.", alpha=0.4)
        ax_gy.axhline(0, color="gray", lw=0.5, ls=":")
        ax_gy.set_ylabel(f"{key}\ngy (dps)")
        ax_gy.legend(fontsize=7, loc="upper right")

        gy_max = gy_pred.max()
        gy_ok  = gy_max >= PUSHOFF_DEFAULT_DPS
        status = "PASS" if gy_ok else f"WARN — peak {gy_max:.1f} < threshold {PUSHOFF_DEFAULT_DPS}"
        ax_gy.set_title(
            f"{key} gy  PINN:[{gy_pred.min():.1f}, {gy_pred.max():.1f}]  GT:[{gy_gt.min():.1f}, {gy_gt.max():.1f}]"
            f"  [{status}]",
            fontsize=8,
            color="green" if gy_ok else "red"
        )

        # ── az panel ──
        ax_az.plot(t_axis, az_pred, color=COLORS["pinn"], lw=1.5, label="PINN")
        ax_az.plot(t_axis, az_gt,   color=COLORS["gt"],   lw=1.0, ls="--", alpha=0.6, label="GT")
        ax_az.axhline(ACC_Z_TOE_OFF, color="orange", lw=1.2, ls="-.",
                      label=_threshold_label(ACC_Z_TOE_OFF, "ACC_Z_TOE_OFF", "PhaseSegmenter"))
        ax_az.axhline(9.81, color="gray", lw=0.5, ls=":", label="1g")
        ax_az.set_ylabel("az (m/s²)")
        ax_az.legend(fontsize=7, loc="upper right")

        az_min = az_pred.min()
        az_ok  = az_min <= ACC_Z_TOE_OFF
        az_status = "PASS" if az_ok else f"WARN — min {az_min:.2f} > threshold {ACC_Z_TOE_OFF}"
        ax_az.set_title(
            f"{key} az  PINN:[{az_pred.min():.2f}, {az_pred.max():.2f}]  GT:[{az_gt.min():.2f}, {az_gt.max():.2f}]"
            f"  [{az_status}]",
            fontsize=8,
            color="green" if az_ok else "red"
        )

        sanity_rows.append({
            "profile":         key,
            "gy_pinn_max":     float(gy_max),
            "gy_threshold":    PUSHOFF_DEFAULT_DPS,
            "gy_ok":           gy_ok,
            "az_pinn_min":     float(az_min),
            "az_threshold":    ACC_Z_TOE_OFF,
            "az_ok":           az_ok,
        })

    axes[-1, 0].set_xlabel("Normalised step time t")
    axes[-1, 1].set_xlabel("Normalised step time t")
    plt.tight_layout()

    # ── save ──
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"step_unit_{run_id}.png")
    plt.savefig(out, dpi=150)
    plt.close()

    # ── stdout table ──
    print(f"\nGate 5a — Signal Sanity Table  (run_id={run_id})")
    print(f"{'Profile':<12}  {'gy_PINN_max':>12}  {'gy_thresh':>10}  {'gy':>6}  "
          f"{'az_PINN_min':>12}  {'az_thresh':>10}  {'az':>6}")
    print("─" * 78)
    all_pass = True
    for r in sanity_rows:
        gy_s = "PASS" if r["gy_ok"] else "WARN"
        az_s = "PASS" if r["az_ok"] else "WARN"
        if not r["gy_ok"] or not r["az_ok"]:
            all_pass = False
        print(f"{r['profile']:<12}  {r['gy_pinn_max']:>12.1f}  {r['gy_threshold']:>10.1f}  {gy_s:>6}  "
              f"{r['az_pinn_min']:>12.2f}  {r['az_threshold']:>10.2f}  {az_s:>6}")

    print()
    if all_pass:
        print("GATE 5a: ALL PASS — PINN amplitudes compatible with algorithm thresholds.")
        print("Justice may proceed to pinn-validator.")
    else:
        print("GATE 5a: WARN — one or more profiles have threshold incompatibilities.")
        print("Review plot. If threshold adaptation is needed, propose a Bill (Amendment 13)")
        print("before running pinn-validator or any simulation.")
    print(f"\nPlot saved: {out}")

    if os.environ.get("GAITSENSE_DEMO") == "1":
        os.system(f"open {out}")

    return sanity_rows


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    run(run_id)
