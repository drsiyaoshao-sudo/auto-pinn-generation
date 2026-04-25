"""
Plot v9 PINN predicted gy step shape vs physics target for all 4 anchor profiles.
Bureaucracy Standing Order: Signal Plotting (Amendment 11 compliance).
"""

import sys
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
from simulator.pinn.physics_loss import _build_gy_target

DEVICE = "cpu"
N_T    = 208  # same resolution as training data

# X columns: [cadence_spm, step_length_m, vert_osc_cm, slope_deg, stance_frac,
#              si_stance_true_pct, mounting_offset_deg, loose_fit_attenuation,
#              step_variability_ms, terrain_int]
TERRAIN_INT = {"flat": 0, "slope": 1, "stairs": 2}

import json
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

G = 9.81


def profile_to_x(p: dict) -> torch.Tensor:
    terrain_int = float(TERRAIN_INT[p["terrain"]])
    return torch.tensor([
        p["cadence_spm"],
        p["step_length_m"],
        p["vertical_oscillation_cm"],
        p["slope_deg"],
        p["stance_frac"],
        p["si_stance_true_pct"],
        p["mounting_offset_deg"],
        p["loose_fit_attenuation"],
        p["step_variability_ms"],
        terrain_int,
    ], dtype=torch.float32)


def physics_gy_target(p: dict, t: torch.Tensor) -> torch.Tensor:
    v_walk = (p["cadence_spm"] / 60.0) * p["step_length_m"]
    slope_rad = p["slope_deg"] * (math.pi / 180.0)
    slope_factor = 1.0 + 0.4 * math.sin(slope_rad)
    stairs_factor = 1.5 if p["terrain"] == "stairs" else 1.0
    peak = (100.0 + 65.0 * v_walk) * slope_factor * stairs_factor
    sf = torch.full_like(t, p["stance_frac"])
    peak_t = torch.full_like(t, peak)
    return _build_gy_target(t, peak_t, sf)


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
        x_vec = profile_to_x(p)                   # (10,)
        x_rep = x_vec.unsqueeze(0).expand(N_T, -1) # (N_T, 10)
        t_rep = t.unsqueeze(1)                      # (N_T, 1)

        with torch.no_grad():
            y_pred = model(x_rep, t_rep)            # (N_T, 6)
        gy_pred = y_pred[:, 4].numpy()              # col 4 = gy_dps

        gy_tgt = physics_gy_target(p, t).numpy()

        c = COLORS[key]
        ax.plot(t_np, gy_tgt,  color="gray",  lw=1.5, ls="--", label="Physics target")
        ax.plot(t_np, gy_pred, color=c,        lw=2.0,          label="PINN v9 prediction")

        peak_tgt  = float(np.max(np.abs(gy_tgt)))
        peak_pred = float(np.max(np.abs(gy_pred)))
        residual  = float(np.sqrt(np.mean((gy_pred - gy_tgt) ** 2)))
        ax.set_title(
            f"{LABELS[key]}\npeak_tgt={peak_tgt:.0f} dps  |  peak_pred={peak_pred:.0f} dps  |  RMS err={residual:.1f} dps",
            fontsize=9,
        )
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_ylabel("gy (dps)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1)

    for ax in axes[-2:]:
        ax.set_xlabel("Normalised step time t")

    fig.suptitle(
        "PINN v10 warmup_best (epoch 93) — Predicted vs Physics-target gy step shape\n"
        "Two-phase checkpoint fix: warmup saved at best raw_physics (not ramp-scaled val_total)",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = _repo_root / "docs" / "executive_branch_document" / "plots" / "gy_step_shape_v10_warmup.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
