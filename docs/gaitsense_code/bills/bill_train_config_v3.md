# Bill: Training Hyperparameter Configuration — v3

**Proposed by:** pinn-compiler agent (Legislature branch)
**Date:** 2026-04-03
**Status:** RATIFIED — Justice directive issued under Article II, 2026-04-03

---

## Justice's Directive

This Bill is issued in response to a direct Justice ruling following the PINN Data Loss Dominance hearing (Article II — human approval granted). The Justice has identified that the ODE residual `val_ode` plateaued at ~40 across all 500 v2 epochs without convergence, indicating the physics ODE was never learned under the v1/v2 lambda regime. The directive mandates a 100-epoch diagnostic training session with physics dominant from epoch 1 and a fine-tuning learning rate loaded from the best existing checkpoint.

This Bill supersedes `bill_train_config_v2` (RATIFIED 2026-04-03) for the parameters listed. It does not supersede the architecture or loss weight derivation standard — Amendment 17 compliance requires all lambda values to be re-derived from first principles.

---

## Problem Statement

**Training runs v1 and v2: physics ODE never converged.**

Evidence from `simulator/pinn/checkpoints/run_v1_metrics.jsonl` (v2 run, lines 27–76):

| Epoch | val_data | val_ode | val_vel | val_phase | physics_weight |
|---|---|---|---|---|---|
| 1  | 14.95 | 464.74 | 0.857 | 0.0143 | 0.003 |
| 30 | 0.352 | 43.16  | 0.345 | 0.0038 | 0.100 |
| 100| 0.346 | 40.77  | 0.179 | 0.0043 | 0.333 |
| 200| 0.391 | 40.79  | 0.149 | 0.0010 | 0.667 |
| 300| 0.429 | 40.83  | 0.131 | 0.0005 | 1.000 |
| 400| 0.433 | 40.62  | 0.131 | 0.0005 | 1.000 |
| 490| 0.426 | 40.43  | 0.133 | 0.0046 | 1.000 |

The ODE residual `val_ode` does not decrease below ~35–43 in any of the 490 logged epochs. It drops from 464 at epoch 1 to ~40 by epoch 30 and then **stagnates permanently**. Under the v2 lambda regime (`lambda_ode = 1.3425e-4`), the weighted ODE contribution to total loss at epoch 300 (full physics) is:

```
Weighted_ODE = lambda_ode * val_ode = 1.3425e-4 * 40.83 = 0.005479
```

Compare this to the data loss contribution at the same epoch:

```
L_data = 1.0 * val_data = 1.0 * 0.4292 = 0.4292
```

ODE fraction of total loss = 0.005479 / (0.4292 + 0.005479 + ...) ≈ 1.2%

The optimizer was receiving a gradient signal that was 1.2% physics (ODE) and ~98% data-driven at full physics weighting. The ODE was never the dominant signal — it was a perturbation on the data loss gradient. This is precisely the failure mode Amendment 17 was ratified to prevent: a lambda that cannot be traced to a physical scale ratio.

**Root cause:** `lambda_ode` in v1/v2 was sized to produce a weighted ODE contribution comparable in magnitude to the data loss, not dominant over it. When val_ode stagnated at ~40 (raw scale much larger than val_data ~0.31), the lambda was too small to give the ODE residual sufficient gradient weight to drive network weights away from the data minimum.

**Physical interpretation (Article I):** The ODE `d²z/dt² + ω²z = F_contact` governs vertical centre-of-mass displacement. The ω² factor — derived from cadence_spm (Article I primitive 2) as `ω = cadence_spm × π / 30` — multiplies the displacement term. The raw ODE residual at R_ode ≈ 40 m²/s⁴ is physically large because it is a second-derivative quantity (acceleration squared, dimensionally). A lambda of 1.3425e-4 maps this to a weighted loss contribution of ~5e-3, which is 100x smaller than the data loss. The cadence primitive was effectively unweighted in the optimizer's gradient computation.

---

## Proposed Changes

This is a diagnostic configuration — 100 epochs only, fine-tuning from the best existing checkpoint, with physics dominant from epoch 1. The purpose is to verify the model CAN close the ODE residual when given sufficient gradient priority.

### Full Proposed train_config.json

```json
{
  "run_id": "v3",
  "optimizer": "Adam",
  "lr_initial": 0.0001,
  "lr_scheduler": "CosineAnnealingLR",
  "lr_scheduler_T_max": 100,
  "lr_scheduler_eta_min": 1e-05,
  "epochs_max": 100,
  "physics_loss_warmup_epochs": 0,
  "batch_size": 256,
  "early_stop_patience": 100,
  "early_stop_min_epoch": 100,
  "val_fraction": 0.15,
  "grad_clip_norm": 1.0,
  "seed": 42,
  "checkpoint_every": 10,
  "log_every": 5,
  "lambda_ode": 0.1031,
  "lambda_vel": 4.908,
  "lambda_phase": 73.625,
  "load_checkpoint": "simulator/pinn/checkpoints/best_v1.pt",
  "_checkpoint_note": "best_v1.pt contains v2 weights per manifest.json note — run_id bug in train_pinn.py (now fixed)",
  "_data_config_present_at_bill_time": true,
  "_batch_size_source": "mini_batch — data_config.json present, >20 profiles",
  "_approx_parameters_from_architecture": 330246,
  "_created_by": "pinn-compiler",
  "_bill": "bill_train_config_v3",
  "_date": "2026-04-03"
}
```

---

## Justification Per Parameter

### Parameters Changing from v2

| Parameter | v2 Value | v3 Value | Justification |
|---|---|---|---|
| `run_id` | "v2" | "v3" | Bookkeeping — distinguishes this diagnostic run in manifest.json per Amendment 16 |
| `lr_initial` | 3e-4 | **1e-4** | Fine-tuning rate for loading from a pre-trained checkpoint; 3e-4 is a cold-start rate and would cause catastrophic forgetting of the data-fitting structure already learned in v2 |
| `lr_scheduler_T_max` | 2000 | **100** | T_max must equal epochs_max for CosineAnnealingLR to span the full training run; 2000 would hold LR near initial for all 100 epochs |
| `epochs_max` | 2000 | **100** | Diagnostic session only — sufficient to verify whether physics-dominant gradient drives val_ode below the ~40 plateau; not a production training run |
| `physics_loss_warmup_epochs` | 300 | **0** | Justice directive: physics must be active from epoch 1; the diagnostic question is specifically whether physics can be learned when forced dominant — warmup would defer this signal |
| `early_stop_patience` | 100 | **100** | Unchanged in absolute value; with epochs_max=100, this effectively disables early stopping — correct for a 100-epoch diagnostic where we want to observe the full trajectory |
| `early_stop_min_epoch` | 500 | **100** | Set equal to epochs_max to disable early stopping during this diagnostic run; any stopping before epoch 100 would truncate the diagnostic evidence |
| `checkpoint_every` | 50 | **10** | 10-epoch granularity over a 100-epoch run provides 10 checkpoint snapshots vs. 2 in v2 — necessary for observing the physics convergence trajectory in a short run |
| `log_every` | 10 | **5** | 5-epoch logging over 100 epochs gives 20 log entries — adequate diagnostic resolution for a short run |
| `lambda_ode` | 1.3425e-4 | **0.1031** | See §1 — Lambda Derivation below |
| `lambda_vel` | 2.8691 | **4.908** | See §1 — Lambda Derivation below |
| `lambda_phase` | 78.472 | **73.625** | See §1 — Lambda Derivation below |
| `load_checkpoint` | (absent) | **"simulator/pinn/checkpoints/best_v1.pt"** | Justice directive: load v2 weights (stored in best_v1.pt per manifest bug note) to begin from the existing data-fitting minimum rather than re-learning from scratch |

### Parameters Unchanged from v2

| Parameter | Value | Justification |
|---|---|---|
| `optimizer` | Adam | Adaptive moment estimation handles the heterogeneous loss scales between lambda_ode*R_ode (~4.1) and lambda_phase*R_phase (~0.59) without manual per-parameter LR tuning |
| `lr_scheduler` | CosineAnnealingLR | Smooth monotone decay from lr_initial to lr_min over T_max epochs — appropriate for fine-tuning where we do not want LR oscillation |
| `lr_scheduler_eta_min` | 1e-5 | Below 1e-5 Adam epsilon (1e-8) dominates the effective step size in float32, making further decay meaningless |
| `batch_size` | 256 | data_config.json present with >20 profiles; full-batch over ~500 profiles × ~2500 samples = 1.25M samples/step exceeds GPU memory and slows iteration |
| `val_fraction` | 0.15 | 85/15 split unchanged; consistent with v2 to make val_ode values comparable across runs |
| `grad_clip_norm` | 1.0 | Global norm clip prevents L_ODE second-order gradient spikes at heel-strike from causing parameter explosions, particularly at physics_loss_warmup_epochs=0 where ODE is active from the first step |
| `seed` | 42 | Matches architecture.json fourier_seed=42; fixed for full reproducibility of diagnostic results |

---

## §1 — Lambda Derivation (Amendment 17 Compliance)

### Reference measurements (v2 best epoch, interpolated at epoch ~35)

Per the Justice's directive and the manifest.json (best_epoch=35, best_val_loss=0.422757). The JSONL logs every 10 epochs; epoch 35 falls between the logged epoch 30 and 40 entries of the v2 run (third run block in run_v1_metrics.jsonl):

| Source | val_data | val_ode | val_vel | val_phase |
|---|---|---|---|---|
| Epoch 30 (logged) | 0.3523 | 43.16 | 0.345 | 0.00383 |
| Epoch 40 (logged) | 0.3235 | 41.56 | 0.262 | 0.00386 |
| Epoch 35 (Justice directive — midpoint approx.) | **0.31** | **40.0** | **0.24** | **0.008** |

The Justice-specified values (0.31, 40, 0.24, 0.008) are used as the reference point. These are the RAW (unweighted) residuals, i.e., the loss before lambda multiplication.

### Step 1: Target allocation

Justice directive: weighted data loss fraction ≈ 0.05, weighted physics fraction ≈ 0.95.

By convention, data loss weight = 1.0, so:

```
L_data = 1.0 * R_data = 1.0 * 0.31 = 0.31

Target L_physics = (0.95 / 0.05) * L_data = 19.0 * 0.31 = 5.89
```

### Step 2: Physics budget allocation across three terms

Physical priority order, tracing to Article I primitives:

1. L_ODE enforces `d²z/dt² + ω²z = F_contact` — the primary vertical oscillation ODE. ω traces directly to cadence_spm (Article I primitive 2): `ω = cadence_spm × π / 30`. F_contact traces to vertical_oscillation_cm (Article I primitive 1). This is the hardest constraint and receives the majority budget.

2. L_vel enforces dz/dt continuity — the first integral of the ODE. It traces to vertical_oscillation_cm and cadence_spm jointly (velocity amplitude = vertical_oscillation × cadence). Secondary constraint.

3. L_phase enforces gait periodicity — regularity constraint anchored to step_length_m and cadence_spm (Article I primitives 2 and 3). Tertiary constraint.

Allocation:

```
ODE  receives 70% of L_physics = 0.70 * 5.89 = 4.123
Vel  receives 20% of L_physics = 0.20 * 5.89 = 1.178
Phase receives 10% of L_physics = 0.10 * 5.89 = 0.589
```

The 70/20/10 split is physically grounded: the ODE is a second-order PDE (hardest to satisfy), velocity is first-order, phase is an integral regularity measure. In PINN literature, second-order constraints receive proportionally higher weight because their gradient curvature is steeper and their convergence slower.

### Step 3: Back-compute lambda values

```
lambda_ode   = Target_ODE   / R_ode   = 4.123 / 40.0  = 0.103075  → 0.1031
lambda_vel   = Target_vel   / R_vel   = 1.178 / 0.24  = 4.9083    → 4.908
lambda_phase = Target_phase / R_phase = 0.589 / 0.008 = 73.625    → 73.625
```

### Step 4: Verification

```
Weighted_ODE   = 0.1031  * 40.0  = 4.124
Weighted_vel   = 4.908   * 0.24  = 1.178
Weighted_phase = 73.625  * 0.008 = 0.589

L_physics_total = 4.124 + 1.178 + 0.589 = 5.891
L_data          = 1.0   * 0.31           = 0.310
L_total         = 5.891 + 0.310          = 6.201

Data fraction    = 0.310 / 6.201 = 4.999% ≈ 5.0%   [TARGET: 5%  — MET]
Physics fraction = 5.891 / 6.201 = 95.0%            [TARGET: 95% — MET]

ODE share of physics   = 4.124 / 5.891 = 70.0%     [TARGET: 70% — MET]
Vel share of physics   = 1.178 / 5.891 = 20.0%     [TARGET: 20% — MET]
Phase share of physics = 0.589 / 5.891 = 10.0%     [TARGET: 10% — MET]
```

All targets satisfied to within floating-point rounding.

### Step 5: Comparison to v2 ODE weighting

```
v2: Weighted_ODE = 1.3425e-4 * 40.0 = 0.00537   → 1.2% of total loss
v3: Weighted_ODE = 0.1031    * 40.0 = 4.124      → 66.5% of total loss
```

The v3 ODE receives 768x more gradient weight per unit raw residual than v2. This is the corrective action. If the model CAN close the ODE residual (i.e., it has sufficient capacity and the ODE is learnable from this checkpoint), val_ode should begin declining within the first 20–30 epochs of this diagnostic run.

### Amendment 17 Compliance Declaration

Each lambda value above:
1. Is derived algebraically from the reference raw residual measurements (not chosen by trial-and-error)
2. Traces to Article I primitives: lambda_ode → cadence_spm (ω) + vertical_oscillation_cm (F_contact); lambda_vel → both; lambda_phase → cadence_spm + step_length_m
3. Is documented with derivation arithmetic before training begins
4. Is filed in a ratified Bill (this document) and may not be adjusted mid-run

---

## Amendment 13 Compliance Declaration

Each hyperparameter is justified by engineering reasoning traceable to model scale, training set size, diagnostic purpose, or numerical stability requirements. No value was chosen by empirical trial-and-error. The lambda values are derived algebraically from the observed raw residual measurements at the v2 best checkpoint. If training diverges under these values, a new Bill must be filed — the configuration may not be adjusted mid-run.

Internal consistency check (Escalation Trigger):
```
early_stop_min_epoch (100) >= physics_loss_warmup_epochs (0) + early_stop_patience (100)
100 >= 0 + 100 = 100   [satisfied — no internal inconsistency]
```

---

## Expected Training Behaviour

- **Epoch 1–10:** val_ode should begin declining from ~40 (it will start at ~40 since we are loading from the v2 checkpoint, not from random init). Any decline confirms the ODE is receiving sufficient gradient signal.
- **Epoch 10–50:** If val_ode declines to below 20, the model has capacity to close the ODE residual under physics-dominant training. Val_data may temporarily increase — this is expected and acceptable in a physics-dominant configuration.
- **Epoch 50–100:** Diagnostic assessment window. The val_ode trajectory determines whether a production v3 training run (2000 epochs) is warranted.
- **Early stopping:** Disabled — early_stop_min_epoch = epochs_max = 100. Full 100-epoch trajectory is required for diagnostic evidence.

## What Constitutes a Failed Diagnostic Requiring a New Bill

- val_ode does not decrease by more than 20% from its initial value (~40) over 100 epochs — this would indicate the model cannot close the ODE residual even under physics-dominant training, suggesting a structural issue (architecture capacity, ODE formulation, or data-to-physics incompatibility)
- NaN or Inf appears in any loss component — escalate to human immediately; do not file a new Bill without human review
- val_data increases to > 5.0 (ten-fold from initial value) while val_ode does not decrease — this indicates the physics gradient is erasing the data-fitting structure entirely, and the lambda values require rebalancing

---

## Checkpoint Provenance (Amendment 16 Compliance)

- Load checkpoint: `simulator/pinn/checkpoints/best_v1.pt`
- SHA-256 at time of Bill filing: `5983ba9bc2015bdd8ac597b0f5597a0570a879442e3b9e0a68100970ba815bdc` (v2 weights — confirmed per manifest.json note: "File is best_v1.pt — run_id bug in train_pinn.py (now fixed). Contains v2 weights.")
- This checkpoint contains the data-fitting minimum reached at epoch 35 of the v2 run, at val_data ≈ 0.31 and val_ode ≈ 40.

---

## Branch

`bill-train-config-v3`

Implementation consists of a single file overwrite (`simulator/pinn/train_config.json`). No firmware, simulation profiles, or algorithm logic are modified.
