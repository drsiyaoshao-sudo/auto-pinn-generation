"""
Generate physics_balance_v1.png — 4-panel physics loss balance plot.
physics-reviewer agent, 2026-04-03.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "/Users/siyaoshao/auto-pinn-generation/docs/executive_branch_document/plots/pinn_training"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "physics_balance_v1.png")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("GaitSense PINN — Physics Loss Balance Review v1\n2026-04-03  |  EVIDENCE FOR HUMAN REVIEW",
             fontsize=12, fontweight="bold", y=0.98)

# ─── Panel 1 — Weighted Loss Balance at Initialisation ───────────────────────
ax1 = axes[0, 0]
profiles = ["flat", "bad_wear", "stairs", "slope"]
ode_vals   = [2.39, 2.39, 0.39, 1.71]
vel_vals   = [4.94, 4.94, 0.31, 3.04]
phase_vals = [0.79, 0.79, 1.77, 1.13]

x = np.arange(len(profiles))
w = 0.25
bars_ode   = ax1.bar(x - w,   ode_vals,   w, label="λ_ODE · L_ODE",   color="#2196F3")
bars_vel   = ax1.bar(x,       vel_vals,   w, label="λ_vel · L_vel",   color="#4CAF50")
bars_phase = ax1.bar(x + w,   phase_vals, w, label="λ_phase · L_phase", color="#FF9800")

ax1.axhline(0.1,  color="red",  linestyle="--", linewidth=1.0, label="Balance bounds (0.1 / 10)")
ax1.axhline(10.0, color="red",  linestyle="--", linewidth=1.0)
ax1.set_xticks(x)
ax1.set_xticklabels(profiles)
ax1.set_ylabel("λ · L value at epoch 0")
ax1.set_title("Weighted Loss Balance at Initialisation")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 6.5)

# ─── Panel 2 — Physics Warmup Ramp ───────────────────────────────────────────
ax2 = axes[0, 1]
epochs = np.arange(0, 201)
warmup_end = 100
ramp = np.where(epochs <= warmup_end, epochs / warmup_end, 1.0)

ax2.plot(epochs, ramp, color="#9C27B0", linewidth=2)
ax2.axvline(warmup_end, color="gray", linestyle="--", linewidth=1.2)
ax2.text(warmup_end + 2, 0.5, "warmup complete", color="gray", fontsize=9, va="center")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("physics_weight")
ax2.set_title("Physics Loss Warmup Ramp (epochs 0–200)")
ax2.set_xlim(0, 200)
ax2.set_ylim(-0.05, 1.15)
ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

# ─── Panel 3 — λ Range Ratios Across Profiles ────────────────────────────────
ax3 = axes[1, 0]
terms = ["L_ODE", "L_vel", "L_phase"]
ratios = [6.2, 16.1, 2.3]
colors = ["green" if r <= 10 else "orange" for r in ratios]

bars = ax3.barh(terms, ratios, color=colors, height=0.5)
ax3.axvline(10, color="red", linestyle="--", linewidth=1.2, label="10× soft threshold")
ax3.text(10.3, 2.35, "10× soft threshold", color="red", fontsize=9, va="center")

# annotate each bar
for bar, ratio, term in zip(bars, ratios, terms):
    ax3.text(ratio + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{ratio}×", va="center", fontsize=10, fontweight="bold")

ax3.set_xlabel("Range ratio (max_profile / min_profile)")
ax3.set_title("λ Range Ratios Across Profiles")
ax3.set_xlim(0, 21)
green_patch  = mpatches.Patch(color="green",  label="≤ 10× (within threshold)")
orange_patch = mpatches.Patch(color="orange", label="> 10× (exceeds threshold)")
ax3.legend(handles=[green_patch, orange_patch], fontsize=8, loc="lower right")

# ─── Panel 4 — Training Set Terrain Distribution ─────────────────────────────
ax4 = axes[1, 1]
terrain_labels = ["flat\n208", "slope\n74", "stairs\n68"]
terrain_sizes  = [208, 74, 68]   # from dataset_manifest.json terrain_counts_train
terrain_colors = ["#42A5F5", "#66BB6A", "#FFA726"]
explode = (0.03, 0.03, 0.03)

wedges, texts, autotexts = ax4.pie(
    terrain_sizes,
    labels=terrain_labels,
    autopct="%1.1f%%",
    colors=terrain_colors,
    explode=explode,
    startangle=140,
    textprops={"fontsize": 10},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")

ax4.set_title("Training Set Terrain Distribution (N=350)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
