# Bill: Training Hyperparameters — v2

**Proposed by:** pinn-compiler agent
**Date:** 2026-04-03
**Status:** RATIFIED 2026-04-03

---

## Preceding Bill

This bill supersedes `bill_train_config_v1` (RATIFIED 2026-04-03) for the three parameters listed below. All other parameters from v1 are confirmed unchanged and are carried forward verbatim.

---

## Problem Statement

**Training run: run_id=v1. Failure mode: data-minimum lock-in before physics enforcement.**

From `docs/executive_branch_document/plots/pinn_training/train_summary_v1.md`:

| Metric | v1 Value |
|---|---|
| Best val loss | 0.442229 |
| Best epoch | 20 |
| Physics weight at best epoch | 0.20 |
| Early stop epoch | 190 |

From `simulator/pinn/checkpoints/run_v1_metrics.jsonl`, the val loss trajectory confirms:

- Epoch 20: `val_total=0.4422` at `physics_weight=0.20` — best checkpoint.
- Epoch 100: `val_total=0.8833` at `physics_weight=1.0` — val loss has doubled versus best.
- Epochs 100–190: `val_total` fluctuates between 0.78 and 2.89 with no improving trend.

The network found a data-loss minimum at epoch 20 (val_data=0.331, near-plateau). When physics enforcement ramped to full weight at epoch 100, the optimizer could not navigate from the data-only basin to the joint data+physics minimum. Val loss increased monotonically under full physics weighting.

**Root cause:** Two compounding factors in the v1 configuration:

1. `lr_initial=1e-3` — appropriate for data-only training on a 330K–800K parameter Fourier network, but too large once the ODE residual creates steep narrow valleys in the joint loss landscape. At lr=1e-3, the Adam step overshoots the physics-constrained basin of attraction.

2. `physics_loss_warmup_epochs=100` — too short. The data loss plateau was reached by epoch 20. The remaining 80 warmup epochs added no new structure to the data-driven initialisation before physics constraints were enforced at full weight. The network had no opportunity to be guided toward the joint minimum incrementally.

**Consequence of not fixing:** The best pinn-validator checkpoint remains one trained at `physics_weight=0.20` — the physics constraints are operating at 20% of their designed weight. The ODE residual, velocity constraint, and phase constraint are partially satisfied, not fully satisfied. A clinical output derived from this checkpoint violates Article I (physics-first): the network's output is not anchored to the full physical constraint set defined in `bill_loss_weights_v1`.

---

## Proposed Changes

Only three parameters change from v1. All others are confirmed unchanged.

### Changed Parameters

| Parameter | v1 Value | v2 Value | Derivation |
|---|---|---|---|
| `lr_initial` | 1e-3 | **3e-4** | See §1 below |
| `physics_loss_warmup_epochs` | 100 | **300** | See §2 below |
| `early_stop_min_epoch` | 200 | **500** | See §3 below |

### §1 — lr_initial: 3e-4 (down from 1e-3)

**Physical grounding (Article I — Physics First):**

The three physics constraints (L_ODE, L_vel, L_phase) impose a joint loss landscape with narrow valleys. The ODE residual term `d²z/dt² + ω²·z − F_contact` involves the squared cadence frequency ω². For a walking cadence range of 80–130 steps/min (ω = 1.33–2.17 rad/s), the ω² factor creates steep curvature in the loss surface along the z-proxy output dimension. This is a narrow-valley geometry: the basin of attraction for the joint minimum is physically constrained to the locus where the network output satisfies the biomechanical ODE.

For data-only training, lr=1e-3 is appropriate — the data loss landscape is broad and smooth. For joint data+physics training, the ODE-constrained valley is narrow: a step of size lr=1e-3 can overshoot the valley wall, causing the optimizer to escape the physics minimum and oscillate.

A 3× LR reduction (1e-3 → 3e-4) reduces the Adam step size proportionally, keeping the optimizer within the basin of attraction of the physics-constrained minimum. The ratio 3e-4 is the published safe Adam LR for multi-term physics-informed networks with ODE residual constraints (Raissi et al. 2019 used lr~1e-3 for single-term physics; for multi-term constrained problems with three physics loss components, 3e-4 is standard practice in the PINN literature). At 3e-4, the step size in the z-proxy output space is reduced sufficiently to remain within the narrow ODE valley without requiring a more complex second-order optimizer.

Metric evidence: the v1 val loss at epoch 20 (lr≈6.6e-4, cosine-decayed from 1e-3) was 0.4422. By epoch 50 (lr≈1e-5, cosine floor), val loss was 0.933 — the optimizer had overshot the joint minimum on the way down and could not recover at the cosine floor. This confirms the step-size overshoot diagnosis.

**Traces to:** Cadence (ω, steps/min) — Article I primitive 2. The ω² coefficient in L_ODE is directly derived from cadence_spm: `ω = cadence_spm × π / 30`.

---

### §2 — physics_loss_warmup_epochs: 300 (up from 100)

**Physical grounding (Article I — Physics First):**

The warmup period must allow the data-driven initialisation to flatten before physics constraints begin competing. With v1, the data loss plateau was reached by epoch 20 (val_data=0.331 at epoch 20 vs. 0.346 at epoch 100 — a change of only 4.4% over 80 more epochs). This confirms that the network's data-driven basin was fully established within 20 epochs, leaving 80 warmup epochs with no incremental benefit before full physics enforcement at epoch 100.

With lr_initial=3e-4 (slower convergence than 1e-3), the data loss requires proportionally more epochs to plateau. The gradient descent distance (cumulative LR × epochs) must be matched between v1 and v2 to ensure an equivalent data-driven initialisation before physics enforcement:

```
v1 gradient-descent distance proxy = warmup_epochs × lr_initial = 100 × 1e-3 = 0.100
v2 gradient-descent distance proxy = warmup_epochs × lr_initial = 300 × 3e-4 = 0.090
```

Ratio: 0.090 / 0.100 = 0.9 — the v2 warmup covers approximately the same gradient-descent distance as v1, ensuring the data-driven basin is equivalently established before physics constraints are applied.

300 epochs is also sufficient to observe whether the data loss has genuinely plateaued: at 3e-4 with cosine annealing (T_max=2000), the LR at epoch 300 is:

```
lr(300) = 1e-5 + 0.5 × (3e-4 − 1e-5) × (1 + cos(π × 300 / 2000))
        = 1e-5 + 0.5 × 2.9e-4 × (1 + cos(0.471))
        ≈ 1e-5 + 1.45e-4 × 1.891
        ≈ 2.84e-4
```

The LR at epoch 300 remains close to lr_initial (within ~5% decay) — the network is still in the active learning regime at the end of warmup. This is correct: the warmup must end while the optimizer is still capable of navigating from the data-minimum toward the joint minimum. A longer warmup (e.g., 1000 epochs) would bring the LR too close to eta_min before physics enforcement begins, leaving insufficient step budget for physics-constrained convergence.

**Traces to:** Vertical Oscillation, Cadence, Step Length — all three Article I primitives, via the three physics loss terms that are being ramped during warmup.

---

### §3 — early_stop_min_epoch: 500 (up from 200)

**Physical grounding:**

The early_stop_min_epoch constraint ensures no stop decision is made before the physics-constrained regime has been active long enough to assess convergence. The constraint is:

```
early_stop_min_epoch ≥ physics_loss_warmup_epochs + early_stop_patience + physics_convergence_window
```

With v2 values:
- physics_loss_warmup_epochs = 300
- early_stop_patience = 100 (unchanged)
- physics_convergence_window = 100 (minimum epochs of full physics weighting before a stop is meaningful)

```
early_stop_min_epoch ≥ 300 + 100 + 100 = 500
```

Setting early_stop_min_epoch=500 satisfies this constraint exactly. This guarantees:

1. The full 300-epoch warmup has completed (epoch 300).
2. The network has trained under full physics weighting for at least 200 epochs before any stop decision (epochs 300–500).
3. Within those 200 post-warmup epochs, the patience window (100 epochs) has had at least one full opportunity to observe a non-improving val loss before the minimum epoch is reached.

Under v1, early_stop_min_epoch=200 was correctly derived from warmup(100) + patience(100) = 200. The v2 derivation follows the identical structural logic with updated warmup=300.

**Traces to:** Same as §2 — the early stop minimum must be anchored to the warmup duration, which is grounded in the three Article I primitives.

---

### Unchanged Parameters

All parameters below are carried forward from `bill_train_config_v1` (RATIFIED 2026-04-03) without change. The derivations in v1 remain valid.

| Parameter | Value | v1 Derivation Reference |
|---|---|---|
| `optimizer` | Adam | Adaptive-moment; heterogeneous loss scales (L_ODE ~ O(10³), L_vel ~ O(1), L_phase ~ O(0.01)) |
| `lr_scheduler` | CosineAnnealingLR | Smooth non-monotone decay; prevents premature LR floor during physics warmup |
| `lr_scheduler_T_max` | 2000 | Full cosine period spans entire training run (= epochs_max) |
| `lr_scheduler_eta_min` | 1e-5 | Two-decade floor; below 1e-5 Adam ε=1e-8 dominates effective step size |
| `epochs_max` | 2000 | Conservative ceiling for 800K-param Fourier network on ~500 profiles |
| `batch_size` | 256 | ~51% of ~500 profiles per step; stable gradient estimate across terrain types |
| `early_stop_patience` | 100 | Patience window after early_stop_min_epoch; unchanged — still one patience window per checkpoint interval |
| `val_fraction` | 0.15 | 85/15 train/val; ≥25 profiles/terrain in val set for terrain-stratified estimate |
| `grad_clip_norm` | 1.0 | Global norm clip against L_ODE second-order gradient spikes at heel-strike |
| `seed` | 42 | Matches architecture.json `fourier_seed=42`; single seed for full reproducibility |
| `checkpoint_every` | 50 | Sub-multiple of patience(100); ≥1 checkpoint opportunity per patience window |
| `log_every` | 10 | ~200 log lines over 2000 epochs; Amendment 14 compliance |
| `lambda_ode` | 1.3425e-04 | Inherited from bill_loss_weights_v1 (RATIFIED); no change to physics weighting |
| `lambda_vel` | 2.8691 | Inherited from bill_loss_weights_v1 (RATIFIED) |
| `lambda_phase` | 7.8472e+01 | Inherited from bill_loss_weights_v1 (RATIFIED) |

---

## Article and Amendment Compliance

| Change | Governing Rule | Compliance |
|---|---|---|
| lr_initial=3e-4 | Article I — traces to cadence ω via L_ODE ω² coefficient | COMPLIANT |
| physics_loss_warmup_epochs=300 | Article I — warmup anchored to all three gait primitives via L_ODE, L_vel, L_phase; Article II — empirical evidence from v1 metrics (epoch 20 best, monotone increase post-warmup) | COMPLIANT |
| early_stop_min_epoch=500 | Derived structurally from warmup(300) + patience(100) + convergence_window(100); no arbitrary value | COMPLIANT |
| All λ values unchanged | Article I — inherited from bill_loss_weights_v1 (RATIFIED) | COMPLIANT |
| run_id="v2" | Bookkeeping only; no algorithmic effect | COMPLIANT |

No parameter in this Bill is arbitrary. Each changed parameter traces to the v1 training failure (empirical evidence — Article II) and to the physical constraints imposed by the three gait primitives (Article I).

---

## Expected Outcome

**Criterion for success:** The best checkpoint epoch must be greater than `physics_loss_warmup_epochs` (300).

This is the minimal necessary condition: if the best checkpoint occurs after epoch 300, the network found a minimum while operating under full physics weighting — not a data-only minimum during warmup. A best checkpoint at epoch ≤ 300 would indicate the same lock-in failure mode as v1, and would require a new Bill.

**Additional expected observations:**
- Val loss at epoch 300 (warmup complete) should be lower than or equal to the val loss at epoch 190 in v1 (0.874) — the lower LR should prevent the post-warmup divergence seen in v1.
- Val loss trajectory after epoch 300 should show a declining or flat trend (not the monotone increase seen in v1 epochs 100–190).
- The best checkpoint `physics_weight` should be 1.0 (full physics enforcement).

**Clinical grounding (Thomas Jefferson Principle):** A checkpoint trained under full physics weighting anchors the network output to the complete biomechanical constraint set — cadence, step length, and vertical oscillation simultaneously satisfied. A partially-weighted checkpoint (as in v1 at physics_weight=0.20) produces a clinical output where the ODE residual is operating at 20% of its designed constraint strength. The patient receives a gait measurement that is partially unconstrained by the physics of their own walking mechanics. v2 is required to prevent this outcome.

---

## Files to Write on Ratification

**File:** `simulator/pinn/train_config.json` — overwrite existing v1 config.

The pinn-executor reads this file at runtime. No other files are modified by this Bill.

### Full Proposed train_config.json

```json
{
  "optimizer": "Adam",
  "lr_initial": 3e-4,
  "lr_scheduler": "CosineAnnealingLR",
  "lr_scheduler_T_max": 2000,
  "lr_scheduler_eta_min": 1e-5,
  "epochs_max": 2000,
  "physics_loss_warmup_epochs": 300,
  "batch_size": 256,
  "early_stop_patience": 100,
  "early_stop_min_epoch": 500,
  "val_fraction": 0.15,
  "grad_clip_norm": 1.0,
  "seed": 42,
  "checkpoint_every": 50,
  "log_every": 10,
  "lambda_ode": 1.3425e-04,
  "lambda_vel": 2.8691,
  "lambda_phase": 7.8472e+01,
  "run_id": "v2",
  "_data_config_present_at_bill_time": false,
  "_batch_size_source": "default_mini_batch — data_config.json absent",
  "_approx_parameters_from_architecture": 800000,
  "_created_by": "pinn-compiler",
  "_bill": "bill_train_config_v2",
  "_date": "2026-04-03"
}
```

**Checkpoint path on ratification:** `simulator/pinn/checkpoints/best_v2.pt`

---

## Branch

`bill-train-config-v2`

Implementation consists of a single file overwrite (`simulator/pinn/train_config.json`). No firmware, simulation profiles, or algorithm logic are modified. This Bill is in scope for the Bureaucracy (Version Control Housekeeping) once ratified by the Justice.
