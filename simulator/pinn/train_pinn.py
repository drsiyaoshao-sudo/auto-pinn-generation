"""
PINN Training Script — written by pinn-executor agent.
Bill reference: bill_train_config_v1
Date: 2026-04-03

Trains PINNModel using PhysicsLoss against training data in training_data/.
All hyperparameters are read from train_config.json — no hardcoded values.

Bureaucracy Standing Order: Simulation Execution class.
This script is deterministic and fully reversible (checkpoint files are additive output).
"""

import json
import math
import time
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── path setup: allow both `python simulator/pinn/train_pinn.py` from repo root
#    and direct execution from the pinn/ directory
_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from simulator.pinn.pinn_model import PINNModel
from simulator.pinn.physics_loss import PhysicsLoss


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train PINN for GaitSense IMU signal generation")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override epochs_max from train_config.json (use for dry runs)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    pinn_dir   = Path(__file__).resolve().parent
    config_path = pinn_dir / "train_config.json"
    data_dir    = pinn_dir / "training_data"
    ckpt_dir    = pinn_dir / "checkpoints"

    # ── Load config ──────────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        cfg = json.load(f)

    lr_initial         = cfg["lr_initial"]
    lr_scheduler_eta_min = cfg["lr_scheduler_eta_min"]
    epochs_max         = cfg["epochs_max"]
    warmup_epochs      = cfg["physics_loss_warmup_epochs"]
    batch_size         = cfg["batch_size"]
    early_stop_patience  = cfg["early_stop_patience"]
    early_stop_min_epoch = cfg["early_stop_min_epoch"]
    grad_clip_norm     = cfg["grad_clip_norm"]
    seed               = cfg["seed"]
    log_every          = cfg["log_every"]
    lambda_ode         = cfg["lambda_ode"]
    lambda_vel         = cfg["lambda_vel"]
    lambda_phase       = cfg["lambda_phase"]
    run_id             = cfg.get("run_id", "v1")

    # CLI override
    if args.max_epochs is not None:
        print(
            f"[DRY RUN] --max-epochs={args.max_epochs} overrides epochs_max={epochs_max} from config. "
            f"This is a dry-run test, not full training."
        )
        epochs_max = args.max_epochs

    checkpoint_path = ckpt_dir / f"best_{run_id}.pt"
    metrics_path    = ckpt_dir / f"run_{run_id}_metrics.jsonl"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    X_train = torch.from_numpy(np.load(data_dir / "X_train.npy")).float().to(device)
    Y_train = torch.from_numpy(np.load(data_dir / "Y_train.npy")).float().to(device)
    t_train = torch.from_numpy(np.load(data_dir / "t_train.npy")).float().to(device)

    X_val   = torch.from_numpy(np.load(data_dir / "X_val.npy")).float().to(device)
    Y_val   = torch.from_numpy(np.load(data_dir / "Y_val.npy")).float().to(device)
    t_val   = torch.from_numpy(np.load(data_dir / "t_val.npy")).float().to(device)

    N_train, T_steps, out_dim = Y_train.shape
    N_val   = Y_val.shape[0]

    print(f"Training samples: {N_train},  Val samples: {N_val},  Time steps: {T_steps},  Output dim: {out_dim}")
    print(f"Config: lr={lr_initial}, epochs={epochs_max}, batch={batch_size}, warmup={warmup_epochs}")
    print(f"Lambda: ode={lambda_ode}, vel={lambda_vel}, phase={lambda_phase}")

    # ── DataLoader ───────────────────────────────────────────────────────────
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    model = PINNModel().to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ── Optimizer and Scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_max,
        eta_min=lr_scheduler_eta_min,
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    physics_loss_fn = PhysicsLoss().to(device)
    data_criterion  = nn.MSELoss()

    # ── Training state ────────────────────────────────────────────────────────
    best_val              = float("inf")
    best_epoch            = 0
    last_improvement_epoch = 0
    consecutive_increases  = 0
    prev_val_total        = float("inf")
    metrics_log           = []

    print(f"\nStarting training — run_id={run_id}, checkpoint={checkpoint_path}")
    print("-" * 90)

    train_start = time.time()

    for epoch in range(1, epochs_max + 1):

        # Physics warmup ramp: 0 → 1 over warmup_epochs
        physics_weight = min(epoch / max(warmup_epochs, 1), 1.0)

        # ── Train one epoch ──────────────────────────────────────────────────
        model.train()
        epoch_loss_sum = 0.0
        n_batches      = 0

        for X_batch, Y_batch in train_loader:
            B = X_batch.shape[0]

            # Expand t to (B*T_steps,) — same time vector for every profile in batch
            t_batch = t_train.unsqueeze(0).expand(B, -1).reshape(-1)    # (B*T_steps,)

            # Expand X to (B*T_steps, 10) — each time point uses its profile params
            X_exp = X_batch.unsqueeze(1).expand(-1, T_steps, -1).reshape(-1, 10)  # (B*T_steps, 10)

            pred = model(X_exp, t_batch)     # (B*T_steps, 6) — attached, requires_grad=True

            # Data loss
            Y_flat    = Y_batch.reshape(-1, 6)        # (B*T_steps, 6)
            loss_data = data_criterion(pred, Y_flat)

            # Per-profile primitives extracted from X_batch (one scalar per profile)
            cadence     = X_batch[:, 0]    # (B,) spm
            step_length = X_batch[:, 1]    # (B,) m
            vert_osc    = X_batch[:, 2]    # (B,) cm
            stance_frac = X_batch[:, 4]    # (B,) dimensionless
            step_period = 60.0 / (cadence + 1e-6)  # (B,) s — guard against /0

            # Expand per-profile scalars to (B*T_steps,)
            def expand_scalar(s: torch.Tensor) -> torch.Tensor:
                return s.unsqueeze(1).expand(-1, T_steps).reshape(-1)

            cadence_exp     = expand_scalar(cadence)
            step_length_exp = expand_scalar(step_length)
            vert_osc_exp    = expand_scalar(vert_osc)
            stance_frac_exp = expand_scalar(stance_frac)
            step_period_exp = expand_scalar(step_period)

            phys_dict = physics_loss_fn.total_loss(
                pred=pred,
                t=t_batch,
                cadence_spm=cadence_exp,
                step_length_m=step_length_exp,
                vert_osc_cm=vert_osc_exp,
                stance_frac=stance_frac_exp,
                step_period_s=step_period_exp,
                lambda_ode=lambda_ode,
                lambda_vel=lambda_vel,
                lambda_phase=lambda_phase,
                physics_weight_ramp=physics_weight,
            )

            total_loss = loss_data + phys_dict["physics"]

            # NaN/Inf guard
            if not torch.isfinite(total_loss):
                print(
                    f"\nFATAL: NaN or Inf loss detected at epoch {epoch}, batch {n_batches+1}. "
                    f"loss_data={loss_data.item():.6f}, "
                    f"l_ode={phys_dict['l_ode'].item():.6f}, "
                    f"l_vel={phys_dict['l_vel'].item():.6f}, "
                    f"l_phase={phys_dict['l_phase'].item():.6f}. "
                    "HALTING — corrupt gradients. Escalate to human."
                )
                sys.exit(1)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_loss_sum += total_loss.item()
            n_batches      += 1

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        # Always compute val loss every epoch for early-stop tracking;
        # only print at log_every intervals.
        #
        # NOTE on grad requirement: PhysicsLoss.l_vel() asserts ax_pred.requires_grad.
        # torch.no_grad() suppresses grad on all tensors produced inside the block,
        # so the val physics pass MUST run outside no_grad (model.eval() alone is
        # sufficient to disable dropout/BN — no_grad is a separate context).
        model.eval()

        # Pre-compute val expansions once (used in both data and physics passes)
        B_val     = N_val
        t_val_exp = t_val.unsqueeze(0).expand(B_val, -1).reshape(-1)
        X_val_exp = X_val.unsqueeze(1).expand(-1, T_steps, -1).reshape(-1, 10)
        Y_val_flat = Y_val.reshape(-1, 6)

        cadence_v     = X_val[:, 0]
        step_length_v = X_val[:, 1]
        vert_osc_v    = X_val[:, 2]
        stance_frac_v = X_val[:, 4]
        step_period_v = 60.0 / (cadence_v + 1e-6)

        def expand_val(s):
            return s.unsqueeze(1).expand(-1, T_steps).reshape(-1)

        # Data loss — no_grad is fine here (no autograd needed for MSE)
        with torch.no_grad():
            pred_val_nograd = model(X_val_exp, t_val_exp)
            val_data = data_criterion(pred_val_nograd, Y_val_flat).item()

        # Physics loss — MUST run outside no_grad so ax_pred.requires_grad is True
        pred_val_grad = model(X_val_exp, t_val_exp)   # fresh forward with grad attached

        phys_val = physics_loss_fn.total_loss(
            pred=pred_val_grad,
            t=t_val_exp,
            cadence_spm=expand_val(cadence_v),
            step_length_m=expand_val(step_length_v),
            vert_osc_cm=expand_val(vert_osc_v),
            stance_frac=expand_val(stance_frac_v),
            step_period_s=expand_val(step_period_v),
            lambda_ode=lambda_ode,
            lambda_vel=lambda_vel,
            lambda_phase=lambda_phase,
            physics_weight_ramp=physics_weight,
        )

        val_ode   = phys_val["l_ode"].item()
        val_vel   = phys_val["l_vel"].item()
        val_phase = phys_val["l_phase"].item()
        val_total = val_data + phys_val["physics"].item()

        # ── Three-strike / 10-epoch divergence rule (Amendment 7) ────────────
        if val_total < prev_val_total:
            consecutive_increases = 0
        else:
            consecutive_increases += 1

        if consecutive_increases == 3:
            print(f"  [WARNING] Val loss has increased for 3 consecutive epochs (epoch {epoch}).")

        if epoch >= early_stop_min_epoch and consecutive_increases >= 10:
            print(
                f"\n[ESCALATION — Amendment 7] Val loss has increased for 10 consecutive epochs "
                f"after warmup (epoch {epoch}). Stopping and escalating to human. "
                f"Current val_total={val_total:.6f}, best_val={best_val:.6f}."
            )
            break

        prev_val_total = val_total

        # ── Checkpoint on improvement ─────────────────────────────────────────
        if val_total < best_val:
            best_val   = val_total
            best_epoch = epoch
            last_improvement_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        # ── Early stopping ────────────────────────────────────────────────────
        if epoch >= early_stop_min_epoch and (epoch - last_improvement_epoch) >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {early_stop_patience} epochs)."
            )
            break

        # ── Logging ───────────────────────────────────────────────────────────
        current_lr = scheduler.get_last_lr()[0]

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - train_start
            print(
                f"Epoch {epoch:4d}/{epochs_max}"
                f" | data={val_data:.4f}"
                f" | ode={val_ode:.4f}"
                f" | vel={val_vel:.4f}"
                f" | phase={val_phase:.4f}"
                f" | ramp={physics_weight:.2f}"
                f" | lr={current_lr:.2e}"
                f" | elapsed={elapsed:.1f}s"
            )

            metric_entry = {
                "epoch":        epoch,
                "val_data":     val_data,
                "val_ode":      val_ode,
                "val_vel":      val_vel,
                "val_phase":    val_phase,
                "val_total":    val_total,
                "physics_weight": physics_weight,
                "lr":           current_lr,
                "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            metrics_log.append(metric_entry)

            with open(metrics_path, "a") as mf:
                mf.write(json.dumps(metric_entry) + "\n")

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("Training complete.")
    print(f"  Best val loss: {best_val:.6f} at epoch {best_epoch}")
    print(f"  Checkpoint: simulator/pinn/checkpoints/best_{run_id}.pt")
    print(f"  Metrics:    simulator/pinn/checkpoints/run_{run_id}_metrics.jsonl")


if __name__ == "__main__":
    main()
