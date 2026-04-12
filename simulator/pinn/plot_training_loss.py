"""
plot_training_loss.py — 4-panel PINN training loss diagnostic.

Called by the plotter agent via /plot-training and by train-sum after each run.
Reads the .jsonl metrics log produced by pinn-monitor during training.

Usage:
    python plot_training_loss.py [run_id]
    python plot_training_loss.py v21_2k

If run_id is omitted, uses the most recent .jsonl in checkpoints/.

Output:
    docs/executive_branch_document/plots/pinn_training/loss_curve_<run_id>.png

4 panels:
    1. Total loss — train + val curves
    2. Data loss — train + val (supervised fitting component)
    3. Physics loss components — val_ode, val_vel, val_phase separately
    4. Learning rate schedule + early stop marker (if applicable)

Constitutional grounding:
    Amendment 11: training plots mandatory after each run
    Amendment 20: plot must show physics vs data loss balance
    /plot-training skill: dispatches this module via plotter agent
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(_HERE, "checkpoints")
PLOTS_DIR      = os.path.join(_HERE, "../../docs/executive_branch_document/plots/pinn_training")


def _find_latest_metrics() -> str:
    logs = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "run_*_metrics.jsonl")),
                  key=os.path.getmtime, reverse=True)
    if not logs:
        raise FileNotFoundError(f"No metrics logs found in {CHECKPOINT_DIR}")
    return logs[0]


def _load_metrics(run_id) :
    if run_id is None:
        path = _find_latest_metrics()
        run_id = os.path.basename(path).replace("run_", "").replace("_metrics.jsonl", "")
    else:
        path = os.path.join(CHECKPOINT_DIR, f"run_{run_id}_metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics log not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return run_id, rows


def run(run_id = None):
    run_id, rows = _load_metrics(run_id)

    epochs        = np.array([r["epoch"]       for r in rows])
    val_total     = np.array([r.get("val_total", r.get("val_data", float("nan"))) for r in rows])
    val_data      = np.array([r.get("val_data",  float("nan")) for r in rows])
    val_ode       = np.array([r.get("val_ode",   float("nan")) for r in rows])
    val_vel       = np.array([r.get("val_vel",   float("nan")) for r in rows])
    val_phase     = np.array([r.get("val_phase", float("nan")) for r in rows])
    lr            = np.array([r.get("lr",        float("nan")) for r in rows])

    best_idx  = int(np.nanargmin(val_total))
    best_ep   = epochs[best_idx]
    best_val  = val_total[best_idx]

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"PINN Training Diagnostics — run_id={run_id}\n"
                 f"Best val loss: {best_val:.4f} at epoch {best_ep}",
                 fontsize=12, fontweight="bold")

    # ── Panel 1: Total loss ───────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(epochs, val_total, color="steelblue",  lw=1.5, label="val total")
    ax.axvline(best_ep, color="gold", lw=1.2, ls="--", label=f"best epoch {best_ep}")
    ax.set_ylabel("Total loss (log)")
    ax.set_title("Panel 1 — Total loss")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel 2: Data loss ────────────────────────────────────────────────────
    ax = axes[1]
    ax.semilogy(epochs, val_data, color="tomato", lw=1.5, label="val data loss")
    ax.axvline(best_ep, color="gold", lw=1.2, ls="--")
    ax.set_ylabel("Data loss (log)")
    ax.set_title("Panel 2 — Data loss (supervised fitting)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel 3: Physics loss components ─────────────────────────────────────
    ax = axes[2]
    physics_any = False
    if not np.all(np.isnan(val_ode)) and np.nanmax(val_ode) > 0:
        ax.semilogy(epochs, val_ode,   color="darkorange",  lw=1.3, label="val ODE residual")
        physics_any = True
    if not np.all(np.isnan(val_vel)) and np.nanmax(val_vel) > 0:
        ax.semilogy(epochs, val_vel,   color="mediumpurple", lw=1.3, label="val velocity")
        physics_any = True
    if not np.all(np.isnan(val_phase)) and np.nanmax(val_phase) > 0:
        ax.semilogy(epochs, val_phase, color="seagreen",    lw=1.3, label="val phase")
        physics_any = True
    if not physics_any:
        ax.text(0.5, 0.5, "All physics λ = 0 this run\n(data-only training)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="gray")
    ax.axvline(best_ep, color="gold", lw=1.2, ls="--")
    ax.set_ylabel("Physics loss (log)")
    ax.set_title("Panel 3 — Physics loss components (Amendment 20)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    # ── Panel 4: Learning rate ────────────────────────────────────────────────
    ax = axes[3]
    ax.semilogy(epochs, lr, color="dimgray", lw=1.3, label="learning rate")
    ax.axvline(best_ep, color="gold", lw=1.2, ls="--", label=f"best epoch {best_ep}")
    ax.set_ylabel("LR (log)")
    ax.set_xlabel("Epoch")
    ax.set_title("Panel 4 — Learning rate schedule")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"loss_curve_{run_id}.png")
    plt.savefig(out, dpi=150)
    plt.close()

    # ── stdout summary ────────────────────────────────────────────────────────
    print(f"\nTraining Loss Summary — run_id={run_id}")
    print(f"  Epochs logged:       {len(rows)}")
    print(f"  Best val total loss: {best_val:.6f}  at epoch {best_ep}")
    print(f"  Final val total:     {val_total[-1]:.6f}  (epoch {epochs[-1]})")
    print(f"  Final LR:            {lr[-1]:.2e}")
    physics_active = any(
        not np.all(np.isnan(a)) and np.nanmax(a) > 0
        for a in [val_ode, val_vel, val_phase]
    )
    print(f"  Physics loss active: {'YES' if physics_active else 'NO (all λ=0)'}")
    print(f"\nPlot saved: {out}")

    if os.environ.get("GAITSENSE_DEMO") == "1":
        os.system(f"open {out}")

    return {"run_id": run_id, "best_epoch": int(best_ep), "best_val": float(best_val)}


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else None
    run(run_id)
