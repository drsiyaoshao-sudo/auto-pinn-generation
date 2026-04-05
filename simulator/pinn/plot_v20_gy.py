"""
Plot v20 PINN gy (and az) for flat and stairs, alongside walker_model ground truth.
One step window (208 samples, t=[0,1]).
"""

import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from pinn_model import PINNModel
from walker_model import PROFILES, generate_imu_sequence, ODR_HZ

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKPT  = os.path.join(os.path.dirname(__file__), "checkpoints/best_v21.pt")
NORM  = os.path.join(os.path.dirname(__file__), "checkpoints/Y_norm_stats_v20.npy")
OUT   = os.path.join(os.path.dirname(__file__), "../../docs/executive_branch_document/plots/v21_gy_az_comparison.png")
TERRAIN_INT = {"flat": 0, "slope": 1, "stairs": 2}
T_STEPS = 208
KEYS = ["flat", "stairs", "slope", "bad_wear"]

# ── load model ────────────────────────────────────────────────────────────────
model = PINNModel(hidden_dim=16, fourier_dim=24, n_layers=4, use_fourier=False, use_cheby=True, cheby_degree=5)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()

norm       = np.load(NORM)
Y_mean     = norm[0]   # (6,)
Y_std_safe = norm[1]   # (6,)

# ── generate ground truth (one step window, same as training data) ────────────
def gt_one_step(profile, seed=42):
    rng = np.random.default_rng(seed)
    seq = generate_imu_sequence(profile, n_steps=2, rng=rng)
    STAT = T_STEPS   # stationary prefix = 1s at 208 Hz
    walking = seq[STAT:]
    return walking[:T_STEPS]   # (208, 6)

# ── PINN inference ─────────────────────────────────────────────────────────────
def pinn_one_step(profile):
    x_vec = np.array([
        profile.cadence_spm, profile.step_length_m, profile.vertical_oscillation_cm,
        profile.slope_deg, profile.stance_frac, profile.si_stance_true_pct,
        profile.mounting_offset_deg, profile.loose_fit_attenuation,
        profile.step_variability_ms, TERRAIN_INT[profile.terrain],
    ], dtype=np.float32)
    t = torch.linspace(0, 1, T_STEPS)
    x = torch.from_numpy(x_vec).unsqueeze(0).expand(T_STEPS, -1)
    with torch.no_grad():
        pred_norm = model(x, t).numpy()
    return pred_norm * Y_std_safe + Y_mean   # (208, 6)

# ── plot ──────────────────────────────────────────────────────────────────────
t_axis = np.linspace(0, 1, T_STEPS)
fig, axes = plt.subplots(len(KEYS), 2, figsize=(13, 3.5 * len(KEYS)), sharex=True)
fig.suptitle("v21 PINN — one canonical step per profile (Cheby T1–T5, 4-layer)\n(left: gy_dps, right: az_ms2)", fontsize=13)

for row, key in enumerate(KEYS):
    profile = PROFILES[key]
    gt   = gt_one_step(profile)
    pred = pinn_one_step(profile)

    ax_gy = axes[row, 0]
    ax_az = axes[row, 1]

    ax_gy.plot(t_axis, pred[:, 4], color="tomato", lw=1.4, label="PINN v20")
    ax_gy.axhline(0, color="gray", lw=0.6, ls=":")
    ax_gy.set_ylabel(f"{key}\ngy (dps)")
    ax_gy.legend(fontsize=8, loc="upper right")
    ax_gy.set_title(f"{key} — gy  PINN:[{pred[:,4].min():.1f},{pred[:,4].max():.1f}]", fontsize=9)

    ax_az.plot(t_axis, pred[:, 2], color="tomato", lw=1.4, label="PINN v20")
    ax_az.axhline(9.81, color="gray", lw=0.6, ls=":")
    ax_az.set_ylabel("az (m/s²)")
    ax_az.set_title(f"{key} — az  PINN:[{pred[:,2].min():.1f},{pred[:,2].max():.1f}]", fontsize=9)

axes[-1, 0].set_xlabel("Normalised step time t")
axes[-1, 1].set_xlabel("Normalised step time t")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=120)
print(f"Saved: {OUT}")

# Print channel-level MSE breakdown
print("\nChannel MSE (v20 PINN vs walker_model, one canonical step):")
print(f"{'Profile':<10}  {'ax':>7}  {'ay':>7}  {'az':>7}  {'gx':>7}  {'gy':>7}  {'gz':>7}")
CHAN_NAMES = ["ax","ay","az","gx","gy","gz"]
for key in KEYS:
    gt   = gt_one_step(PROFILES[key])
    pred = pinn_one_step(PROFILES[key])
    mse  = ((gt - pred)**2).mean(axis=0)
    print(f"{key:<10}  " + "  ".join(f"{v:>7.2f}" for v in mse))
