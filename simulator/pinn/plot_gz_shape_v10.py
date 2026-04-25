"""
Plot gz step shape: PINN v10 prediction vs walker_model.py ground truth (noise floor).
Bureaucracy Standing Order: Signal Plotting (Amendment 11).

gz in walker_model.py is pure gyro sensor noise — no physics signal.
gx is identical. Only gy carries the gait waveform.
This plot shows what the PINN outputs for gz vs the actual data anchor.
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_pinn_dir  = Path(__file__).resolve().parent
_repo_root = _pinn_dir.parent.parent
sys.path.insert(0, str(_repo_root))

from simulator.pinn.pinn_model import PINNModel

DEVICE = "cpu"
N_T    = 208
TERRAIN_INT = {"flat": 0, "slope": 1, "stairs": 2}

with open(_pinn_dir / "training_data" / "anchor_profiles.json") as f:
    ANCHORS = json.load(f)

PROFILE_ORDER = ["flat", "stairs", "slope", "bad_wear"]
LABELS = {
    "flat":     "Flat — healthy adult",
    "stairs":   "Stairs — ascending",
    "slope":    "Slope 10° — ascending",
    "bad_wear": "Bad wear — loose fit",
}
COLORS = {"flat": "#2196F3", "stairs": "#FF5722", "slope": "#4CAF50", "bad_wear": "#9C27B0"}
ANCHORS_NPY = {k: _pinn_dir / "training_data" / f"anchor_{k}.npy" for k in PROFILE_ORDER}


def profile_to_x(p: dict) -> torch.Tensor:
    terrain_int = float(TERRAIN_INT[p["terrain"]])
    return torch.tensor([
        p["cadence_spm"], p["step_length_m"], p["vertical_oscillation_cm"],
        p["slope_deg"], p["stance_frac"], p["si_stance_true_pct"],
        p["mounting_offset_deg"], p["loose_fit_attenuation"],
        p["step_variability_ms"], terrain_int,
    ], dtype=torch.float32)


def main():
    ckpt_path = _pinn_dir / "checkpoints" / "warmup_best_v10.pt"
    model = PINNModel()
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path.name}")

    t_np = np.linspace(0.0, 1.0, N_T, dtype=np.float32)
    t    = torch.from_numpy(t_np)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()

    for ax, key in zip(axes, PROFILE_ORDER):
        p = ANCHORS[key]
        x_vec = profile_to_x(p)
        x_rep = x_vec.unsqueeze(0).expand(N_T, -1)
        t_rep = t.unsqueeze(1)

        with torch.no_grad():
            y_pred = model(x_rep, t_rep)
        gz_pred = y_pred[:, 5].numpy()   # col 5 = gz_dps

        # Ground truth from anchor npy: shape (208, 6), col 5 = gz
        gz_true = np.load(ANCHORS_NPY[key])[:, 5]

        c = COLORS[key]
        ax.plot(t_np, gz_true, color="gray", lw=1.0, alpha=0.6, label="walker_model.py (noise floor)")
        ax.plot(t_np, gz_pred, color=c,      lw=2.0,              label="PINN v10 prediction")

        ax.axhline(0, color="k", lw=0.5, ls=":")
        pred_rms = float(np.sqrt(np.mean(gz_pred ** 2)))
        true_rms = float(np.sqrt(np.mean(gz_true ** 2)))
        ax.set_title(
            f"{LABELS[key]}\npred_RMS={pred_rms:.3f} dps  |  true_RMS={true_rms:.3f} dps"
            f"\n(gz has no physics signal — only gyro noise σ≈0.04 dps)",
            fontsize=8,
        )
        ax.set_ylabel("gz (dps)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1)

    for ax in axes[-2:]:
        ax.set_xlabel("Normalised step time t")

    fig.suptitle(
        "PINN v10 warmup_best (epoch 93) — gz step shape\n"
        "gz = pure sensor noise in walker_model.py (σ≈0.04 dps, no gait physics)",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = _repo_root / "docs" / "executive_branch_document" / "plots" / "gz_step_shape_v10_warmup.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
