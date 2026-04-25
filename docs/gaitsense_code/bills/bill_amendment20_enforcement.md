### BILL: Enforce Amendment 20 — Zero data weight during physics warmup

Proposed by: human + main session
Date drafted: 2026-04-19
Change type: software (train_pinn.py)
Status: RATIFIED — 2026-04-19

---

## Problem Statement

Amendment 20 (PINN Physics-First Training Order) requires:
  "physics loss >= 80% of total loss during warmup with verified downward trend
   in all physics terms before data fitting may dominate."

The current train_pinn.py implementation computes:
    total_loss = loss_data + physics_weight_ramp * physics_loss

With loss_data always at weight 1.0, at epoch 1 (physics_weight=0.01):
    physics fraction = 0.01 × 1.46 / (517 + 0.01 × 1.46) ≈ 0.003%

This is a direct violation of Amendment 20. The data loss dominates from epoch 1,
preventing the physics terms from establishing a learnable gradient signal.
v4/v5/v6 training all showed val_gyy plateau or divergence caused by this error.

## Proposed Change

During `physics_loss_warmup_epochs`, set `data_weight = 0.0`:
    total_loss = data_weight * loss_data + phys_dict["physics"]
    data_weight = 0.0  during warmup  (epochs 1 → physics_loss_warmup_epochs)
    data_weight = 1.0  after warmup   (epochs > physics_loss_warmup_epochs)

physics_weight_ramp is unchanged (0→1 linear over warmup_epochs) — it controls
the relative weight of physics terms against each other during warmup.

After warmup: data enters at full weight. The model has spent 100 epochs fitting
only physics (gy waveform, az baseline, vel, phase) and can now use data to refine
without data overriding the physics-aligned state.

Amendment 20 compliance with this fix:
  Warmup epoch 1:   physics = 100% of total loss  (data_weight = 0)
  Warmup epoch 50:  physics = 100% of total loss  (data_weight = 0)
  Warmup epoch 100: physics = 100% of total loss  (data_weight = 0)
  Post-warmup:      data enters; physics competes on its own merit

## Files Changed

- simulator/pinn/train_pinn.py — data_weight logic in training loop and validation

## Article/Amendment Grounding

- Amendment 20: physics-dominant warmup is now enforced structurally, not just
  by intent. The data term is excluded during warmup by setting data_weight=0.
- Article II: this change is a bug fix restoring the known-correct behaviour
  mandated by Amendment 20. No new behaviour is introduced.
