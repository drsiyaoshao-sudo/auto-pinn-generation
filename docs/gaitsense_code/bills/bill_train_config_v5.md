# Bill: Training Hyperparameter Configuration — v5

**Proposed by:** pinn-compiler agent
**Date:** 2026-04-19
**Status:** RATIFIED

## Change Type

Lambda recalibration only. All training hyperparameters are identical to bill_train_config_v4.
Hyperparameter justifications are not repeated here — see bill_train_config_v4 for the full
per-parameter rationale. This Bill exists solely to record the updated physics loss weights
as required by the constitutional rule that every change to train_config.json must trace to
a ratified Bill.

## Source Bill

`bill_loss_recalibration_v1` — ratified 2026-04-19.
The four lambda values below are copied verbatim from that Bill. They were derived by the
loss-setter agent from the v4 training run and represent recalibration to bring each physics
loss component to the same order of magnitude as the data loss at the end of the warmup
period.

## Changed Fields vs v4

| Field | v4 Value | v5 Value | Change Direction |
|---|---|---|---|
| `run_id` | "v4" | "v5" | identifier increment |
| `lambda_gyy` | 2.994e-05 | 3.651e-04 | +12.2x (gyroscope y-axis loss was underweighted) |
| `lambda_az` | 6.054e-03 | 9.443 | +1560x (vertical acceleration loss was severely underweighted) |
| `lambda_vel` | 4.908 | 1.126 | -4.4x (velocity loss was overweighted relative to others) |
| `lambda_phase` | 73.625 | 1069 | +14.5x (gait phase loss was underweighted) |

## Unchanged Fields vs v4

All other fields carry over without modification: optimizer, lr_initial, lr_scheduler,
lr_scheduler_T_max, lr_scheduler_eta_min, epochs_max, physics_loss_warmup_epochs,
batch_size, early_stop_patience, early_stop_min_epoch, val_fraction, grad_clip_norm,
seed, checkpoint_every, log_every, load_checkpoint, _approx_parameters_from_architecture.

## Amendment Grounding

Amendment 17 — physics loss weights must be set by a ratified Bill and must trace to
empirical evidence from a completed training run. The lambda values in this Bill are
sourced from bill_loss_recalibration_v1, which carries that empirical grounding.

## Amendment 13 Compliance Declaration

No hyperparameter was changed without engineering justification traceable to the v4
training run. The lambda adjustments correct component imbalance observed during v4
training; they are not arbitrary. If training diverges under these values, a new Bill
must be filed — the configuration may not be adjusted mid-run.

## Proposed train_config.json

```json
{
  "run_id": "v5",
  "optimizer": "Adam",
  "lr_initial": 0.001,
  "lr_scheduler": "CosineAnnealingLR",
  "lr_scheduler_T_max": 500,
  "lr_scheduler_eta_min": 1e-05,
  "epochs_max": 500,
  "physics_loss_warmup_epochs": 100,
  "batch_size": 256,
  "early_stop_patience": 100,
  "early_stop_min_epoch": 200,
  "val_fraction": 0.15,
  "grad_clip_norm": 1.0,
  "seed": 42,
  "checkpoint_every": 25,
  "log_every": 5,
  "lambda_gyy": 3.651e-04,
  "lambda_az": 9.443,
  "lambda_vel": 1.126,
  "lambda_phase": 1069,
  "load_checkpoint": null,
  "_approx_parameters_from_architecture": 166582,
  "_created_by": "pinn-compiler",
  "_bill": "bill_train_config_v5",
  "_date": "2026-04-19",
  "bill_ref": "bill_train_config_v5",
  "ratified_date": "2026-04-19"
}
```

## Expected Training Behaviour

- Physics loss components should be same order of magnitude as data loss after the
  100-epoch warmup period (this is what the recalibrated lambdas are designed to achieve).
- Early stopping must not fire before epoch 200.
- Loss should decrease monotonically after the warmup period.

## What Constitutes a Failed Run Requiring a New Bill

- Total loss does not decrease in 100 consecutive epochs after the warmup period.
- Any physics loss component grows relative to data loss after the ramp-up (indicates
  the recalibrated weight is still miscalibrated — file a new loss recalibration Bill).
- NaN or Inf appears in any loss component — escalate to human immediately.
