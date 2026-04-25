# Bill: Training Hyperparameter Configuration — v4

**Proposed by:** pinn-compiler agent (Legislature branch)
**Date:** 2026-04-19
**Status:** RATIFIED — 2026-04-19

---

## Problem Statement

The v3 training run (100 epochs, `bill_train_config_v3`) was a diagnostic run. It confirmed physics
convergence is achievable under physics-dominant training, but it did not constitute a production
training campaign: the gy amplitude target of >= 80 dps (Article I compliance) was not met at
epoch 15 (best checkpoint). The v3 checkpoint (`best_v3.pt`) cannot be reused for v4 because:

1. The architecture changed: `bill_layer_setter_v2.md` (ratified 2026-04-19) replaces the v3
   Fourier Feature Network (330,246 params) with the Polynomial-Wavelet Outer-Product Network
   (166,582 params). Weight tensors are structurally incompatible.
2. The physics loss changed: `bill_physics_loss_v3.md` (ratified 2026-04-19) removes `lambda_ode`
   and replaces the spring oscillator ODE with two data-aligned waveform constraints
   (`lambda_gyy`, `lambda_az`) per Amendment 21 (Data-Physics Alignment Rule). Continuing from
   a checkpoint trained on the invalidated ODE loss would carry misspecified weight structure
   into the new loss regime.

A fresh training campaign is required — starting from random initialisation, with sufficient epochs
to allow physics convergence (Amendment 20 warmup) and subsequent data fitting to reach the gy
amplitude threshold.

---

## Proposed train_config.json

```json
{
  "run_id": "v4",
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
  "lambda_gyy": 2.994e-05,
  "lambda_az": 6.054e-03,
  "lambda_vel": 4.908,
  "lambda_phase": 73.625,
  "load_checkpoint": null,
  "_checkpoint_note": "Fresh start — v3 checkpoint is architecturally incompatible with v4 model (PolynomialWaveletOuterProduct vs FourierFeatureNetwork). No warm-start possible.",
  "_data_config_present_at_bill_time": true,
  "_batch_size_source": "mini_batch — 504 profiles in data_config.json, >20 profiles; 350 train profiles x ~2500 samples/profile = ~875000 samples/epoch; full-batch infeasible on RTX 2080 Ti (11 GB VRAM)",
  "_approx_parameters_from_architecture": 166582,
  "_architecture_bill": "bill_layer_setter_v2",
  "_physics_loss_bill": "bill_physics_loss_v3",
  "_created_by": "pinn-compiler",
  "_bill": "bill_train_config_v4",
  "_date": "2026-04-19"
}
```

---

## Justification Per Parameter

### Training Duration

| Parameter | Value | Justification |
|---|---|---|
| `epochs_max` | 500 | v3's 100-epoch diagnostic was insufficient to reach the gy >= 80 dps threshold (Article I); 500 epochs provides 5x coverage, accommodating a 100-epoch Amendment 20 warmup phase plus 400 epochs of data-dominant refinement on the 166k-parameter model |
| `lr_scheduler_T_max` | 500 | CosineAnnealingLR must equal `epochs_max` to span the full training run; a T_max shorter than epochs_max would hold LR near `lr_min` for the final epochs, eliminating any further learning |

### Optimiser and Learning Rate

| Parameter | Value | Justification |
|---|---|---|
| `optimizer` | Adam | Adaptive moment estimation handles the heterogeneous gradient scales between lambda_gyy (2.994e-05, small) and lambda_phase (73.625, large) without per-parameter LR tuning; appropriate when physics loss terms span four orders of magnitude |
| `lr_initial` | 0.001 | v4 is a cold start (no loaded checkpoint); 1e-3 is the canonical Adam starting point for a 166k-parameter MLP from random initialisation; the wavelet branch's trainable mu and log_scale parameters benefit from a moderate initial rate to explore the temporal domain before settling near the physical gy pulse timing |
| `lr_scheduler` | CosineAnnealingLR | Smooth cosine decay avoids the sharp LR drop of StepLR that can destabilise physics loss convergence mid-ramp; the smooth monotone decay is particularly appropriate during the Amendment 20 warmup phase where physics gradients must remain consistent |
| `lr_scheduler_eta_min` | 1e-05 | At lr < 1e-5 in float32, Adam's epsilon (1e-8) dominates the effective step size, making further decay numerically meaningless; 1e-5 is the practical floor for float32 PINN training |

### Amendment 20 Compliance (Physics-First Training Order)

| Parameter | Value | Justification |
|---|---|---|
| `physics_loss_warmup_epochs` | 100 | Amendment 20 requires physics weighted contribution >= 80% of total loss and each physics term to show a statistically downward trend over at least 10 consecutive logged epochs before data fitting may dominate; with log_every=5, 10 logged epochs = 50 real epochs; 100-epoch warmup provides the required minimum (50 epochs for Amendment 20 criterion) plus a 50-epoch safety margin to absorb the slower convergence of the new waveform-aligned loss terms (l_gyy_pulse, l_az_gravity) whose gradients depend on trainable wavelet positions settling first |
| `early_stop_min_epoch` | 200 | Early stopping must not fire before the warmup phase is complete and has had time to influence weight space; 200 = 100 (warmup) + 100 (post-warmup settling before stopping is valid); this ensures the Amendment 20 criterion is always fully evaluated before any stop decision |
| `early_stop_patience` | 100 | 100 consecutive epochs with no val_loss improvement triggers stop; with early_stop_min_epoch=200, the earliest possible stop is epoch 300, giving the scheduler sufficient time to decay lr and consolidate physics-constrained representations |

### Dataset and Batching

| Parameter | Value | Justification |
|---|---|---|
| `batch_size` | 256 | data_config.json is present with 504 profiles (350 train); 350 profiles x ~2500 samples/profile = ~875,000 samples/epoch; full-batch over 875k samples exceeds RTX 2080 Ti VRAM (11 GB, fp16 only) and produces prohibitively slow iteration; 256-sample mini-batch with shuffle is standard practice at this scale and keeps per-step GPU memory under 1 GB |
| `val_fraction` | 0.15 | 85/15 split consistent with v3, making val_loss comparable across runs; 15% of 350 train profiles = 52 held-out profiles, sufficient for a statistically reliable val_loss estimate |

### Numerical Stability

| Parameter | Value | Justification |
|---|---|---|
| `grad_clip_norm` | 1.0 | Global L2 gradient norm clip prevents the l_gyy_pulse term — which computes MSE against a composite waveform with sharp heel-strike peaks (~186 dps in the flat anchor profile) — from generating gradient spikes that would destabilise the wavelet mu/log_scale parameters during the warmup phase |

### Reproducibility and Logging

| Parameter | Value | Justification |
|---|---|---|
| `seed` | 42 | Fixed seed for all random operations (weight init, batch shuffle, data split); matches architecture.json and all prior runs for cross-run comparability per Amendment 16 |
| `checkpoint_every` | 25 | 25-epoch granularity over 500 epochs yields 20 checkpoint snapshots, sufficient to reconstruct the convergence trajectory for the Amendment 16 manifest and for identifying the best epoch for Amendment 19 fidelity evaluation |
| `log_every` | 5 | 5-epoch logging gives 100 log entries over 500 epochs; with Amendment 20 requiring downward trend over 10 consecutive logged epochs, log_every=5 means the criterion can be evaluated from epoch 50 onward — adequate resolution without excessive I/O |

### Physics Loss Lambda Values (Amendment 17 Compliance)

| Parameter | Value | Source |
|---|---|---|
| `lambda_gyy` | 2.994e-05 | Derived in bill_physics_loss_v3.md: 1 / mean(peak_angvel_dps^2) across 4 anchor profiles = 1/33435 = 2.994e-05; normalises the gy waveform MSE to unit scale relative to the primitive-derived angular velocity amplitude |
| `lambda_az` | 6.054e-03 | Derived in bill_physics_loss_v3.md: 1 / mean(az_dc^2) across 4 anchor profiles = 1/157.6 = 6.054e-03; normalises the az mean constraint to unit scale relative to the primitive-derived DC gravity baseline |
| `lambda_vel` | 4.908 | Carried from bill_loss_weights_v1.md (unchanged in bill_physics_loss_v3.md); derived from raw vel residual at v2 best epoch and the 70/20/10 physics budget allocation |
| `lambda_phase` | 73.625 | Carried from bill_loss_weights_v1.md (unchanged in bill_physics_loss_v3.md); derived from raw phase residual and the physics budget allocation |
| `lambda_ode` | REMOVED | bill_physics_loss_v3.md removes l_ode per Amendment 21 — the spring oscillator ODE is structurally misspecified for the half-sine composite gy waveform; including it here would contradict the ratified physics loss Bill |

### Checkpoint Loading

| Parameter | Value | Justification |
|---|---|---|
| `load_checkpoint` | null | The v3 checkpoint (best_v3.pt, FourierFeatureNetwork, 330,246 params) is architecturally incompatible with the v4 model (PolynomialWaveletOuterProduct, 166,582 params); no warm-start is possible; fresh initialisation from Xavier uniform is required per pinn_model.py `_init_weights()` |

---

## Amendment 13 Compliance Declaration

Each hyperparameter is justified by engineering reasoning traceable to model scale (166,582
parameters, architecture.json v2), training set size (504 profiles, 350 train, data_config.json),
GPU hardware constraints (RTX 2080 Ti, 11 GB VRAM, fp16), numerical stability requirements, or
the Amendment 20 warmup schedule. No value was chosen by empirical trial-and-error. The lambda
values are carried verbatim from their derivations in bill_physics_loss_v3.md (ratified 2026-04-19)
and bill_loss_weights_v1.md — they were derived algebraically from the three Article I primitives.
If training diverges under these values, a new Bill must be filed — the configuration may not be
adjusted mid-run.

Internal consistency check (Escalation Trigger — early_stop_min_epoch must not be less than
physics_loss_warmup_epochs):

```
early_stop_min_epoch (200) >= physics_loss_warmup_epochs (100)   [SATISFIED]
early_stop_min_epoch (200) >= physics_loss_warmup_epochs (100) + early_stop_patience guard
200 - 100 = 100 post-warmup epochs before early stop can fire   [ADEQUATE]
```

No internal inconsistency. Bill may proceed to ratification.

---

## Amendment 20 Compliance Declaration (PINN Physics-First Training Order)

This Bill satisfies all four operational criteria of Amendment 20:

1. **Warmup phase:** `physics_loss_warmup_epochs = 100` is specified. The training executor
   (`pinn-executor`) must enforce physics weighted contribution >= 80% of total loss from epoch 1
   through epoch 100.

2. **Downward trend criterion:** With `log_every = 5`, the Amendment 20 criterion (downward
   trend over 10 consecutive logged epochs) is assessable from epoch 50 onward. The 100-epoch
   warmup provides a 50-epoch margin beyond the minimum assessment window.

3. **Data-dominant phase:** Data fitting begins after epoch 100. The transition is controlled by
   `pinn-executor`'s physics weight scheduler. `pinn-executor` must log the epoch at which all
   physics terms first show a net downward trend over 10 consecutive logged epochs, and record
   that this criterion was satisfied before data-dominant training began.

4. **Warmup schedule in this ratified Bill:** `physics_loss_warmup_epochs = 100` is the
   normative value. Any deviation requires a new Bill under Amendment 17.

---

## Expected Training Behaviour

- Epochs 1–100 (warmup): physics terms (l_gyy_pulse, l_az_gravity, l_vel, l_phase) should all
  show net downward trend; data loss may increase temporarily — this is expected and acceptable
- Epochs 100–200 (post-warmup settling): data loss should begin declining as data-dominant
  training activates; physics terms should remain stable, not diverge
- Epochs 200–300: early stopping is now permitted; val_loss should be declining consistently;
  if it has plateaued, patience clock has started
- Epochs 300–500: if not stopped early, lr will have decayed to ~lr_min; final convergence
  expected in this window; gy amplitude should approach >= 80 dps threshold

Early stopping will not fire before epoch 200. If the run completes all 500 epochs, this indicates
the model found continuous improvement throughout — a healthy sign for a fresh-start cold-training
campaign of this scale.

## What Constitutes a Failed Run Requiring a New Bill

- Total val_loss does not decrease in 100 consecutive epochs after epoch 200 (early_stop_patience
  fires) AND the gy amplitude at the best checkpoint is still below 80 dps — the run completed
  without meeting Article I exit criterion; a new Bill with revised epochs_max or lambda values
  is required
- Any physics loss term shows a net upward trend during the warmup phase (epochs 1–100) — the
  l_gyy_pulse or l_az_gravity waveform constraints are not converging; escalate to human before
  filing a new Bill (the cause may be in physics_loss.py, not the training config)
- NaN or Inf appears in any loss component — hardware or numerical issue; escalate to human
  immediately; do not file a new Bill without human review
- val_loss at epoch 200 is higher than at epoch 1 — warmup phase produced no improvement in
  any objective; structural problem with the new architecture or physics loss terms; escalate

---

## Amendment 16 Compliance (PINN Model Provenance)

- `load_checkpoint: null` — v4 starts from random initialisation; no prior checkpoint is loaded
- The resulting best checkpoint must be registered in `simulator/pinn/checkpoints/manifest.json`
  by `pinn-archivist` before it is used for any grid search or boundary-finding campaign
- SHA-256 hash and (parameter set, loss weight vector, seed, best epoch) tuple must be recorded
  at checkpoint time

---

## Branch

`hybrid-model`

Implementation consists of two file writes:
1. `simulator/pinn/train_config.json` — updated for v4 run (this document's JSON block)
2. No firmware, simulation profiles, walker model, or algorithm logic are modified
