### BILL: Physics Loss Weight Recalibration v2 — lambda_gyy warmup balance fix

Proposed by: main session
Date drafted: 2026-04-19
Status: RATIFIED — 2026-04-19

---

## Problem Statement

v7 and v8 training both showed val_gyy plateau during pure-physics warmup (data_weight=0).
Root cause: lambda_az × L_az dominates warmup gradient budget entirely.

At epoch 1, warmup gradient budget:
  lambda_az  × L_az  = 9.443 × 87.6  = 827  (99.86% of physics gradient)
  lambda_gyy × L_gyy = 3.651e-04 × 3230 = 1.18  (0.14% of physics gradient)

l_az converges rapidly (87 → 0.07 in 80 epochs), leaving lambda_gyy starved.
val_gyy moved only 0.67% across 100 warmup epochs — insufficient for convergence.

## Derivation (Amendment 17)

Target: lambda_gyy × L_gyy ≈ lambda_az × L_az at warmup epoch 1, so both
physics terms receive comparable gradient budget from the start.

  lambda_gyy_new = lambda_az × L_az_epoch1 / L_gyy_epoch1
                 = 9.443 × 87.6 / 3230
                 = 825.8 / 3230
                 = 0.2556

Rounded to 3 significant figures: lambda_gyy = 0.256

All other lambda values unchanged:
  lambda_az    = 9.443   (unchanged)
  lambda_vel   = 1.126   (unchanged)
  lambda_phase = 1069    (unchanged)

## Expected Outcome

At warmup epoch 1:
  lambda_gyy × L_gyy = 0.256 × 3230 ≈ 827 (equal to lambda_az × L_az)
  Both terms compete equally for gradient budget.
  As l_az converges, lambda_gyy × L_gyy becomes the dominant physics term.
  val_gyy should show clear downward trend within 50 warmup epochs.

## Files Changed

- simulator/pinn/train_config.json: lambda_gyy 3.651e-04 → 0.256, run_id v8 → v9
- docs/gaitsense_code/bills/bill_loss_recalibration_v2.md: this document
