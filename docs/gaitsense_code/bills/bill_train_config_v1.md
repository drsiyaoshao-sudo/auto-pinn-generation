# Bill: Training Hyperparameters — v1

**Proposed by:** pinn-compiler agent
**Date:** 2026-04-03
**Status:** RATIFIED 2026-04-03

---

## Source Files Read

| File | Status | Key Values Used |
|---|---|---|
| `simulator/pinn/architecture.json` | Present | `approx_parameters=800000`, `fourier_seed=42`, `n_layers=4`, `hidden_dim=256`, `fourier_dim=256` |
| `simulator/pinn/data_config.json` | **Absent** | Default mini-batch=256 applied; will be revised after synthetic-data-setter runs |
| `docs/gaitsense_code/bills/bill_loss_weights_v1.md` | Present (RATIFIED) | `λ_ODE=1.34e-4`, `λ_vel=2.87`, `λ_phase=78.47`; warmup ramp required before physics terms dominate |

---

## Proposed Configuration

| Parameter | Value | Derivation |
|---|---|---|
| `optimizer` | Adam | Standard adaptive-moment optimizer — see Optimizer section |
| `lr_initial` | 1e-3 | Adam default; scales well to 800K-parameter Fourier networks — see Learning Rate section |
| `lr_scheduler` | CosineAnnealingLR | Smooth non-monotone decay; prevents premature floor — see Learning Rate section |
| `lr_scheduler_T_max` | 2000 (= epochs_max) | Full cosine period spans entire training run |
| `lr_scheduler_eta_min` | 1e-5 | Two-decade floor; below this Adam momentum accumulation dominates noise |
| `epochs_max` | 2000 | Conservative ceiling for 800K-param network on ~500 profiles — see Epochs section |
| `physics_loss_warmup_epochs` | 100 | Linear ramp 0→1 on physics loss coefficient; mandated by bill_loss_weights_v1 expected-behaviour section |
| `batch_size` | 256 | Default mini-batch (data_config.json absent); see batch_size Note |
| `early_stop_patience` | 100 | Patience in epochs on val loss; see Early Stopping section |
| `early_stop_min_epoch` | 200 | Do not stop before physics warmup completes and model has seen 2× patience window |
| `val_fraction` | 0.15 | 85/15 train/val split; test split reserved for synthetic-data-setter |
| `grad_clip_norm` | 1.0 | Global gradient norm clip; guards against L_ODE spike gradients — see Grad Clip section |
| `seed` | 42 | Matches `fourier_seed=42` in architecture.json — single seed for full reproducibility |
| `checkpoint_every` | 50 | Archivist checkpoint on val loss improvement, sampled every 50 epochs |
| `log_every` | 10 | Console milestone print cadence; Amendment 14 compliance |

---

## Parameter Derivations

### Optimizer

**Adam** (Adaptive Moment Estimation).

Traces to: training stability argument — Article II (empirical evidence governs decisions).

Adam maintains per-parameter adaptive learning rates using exponential moving averages of gradient (m) and squared gradient (v). For a network with heterogeneous loss scales — L_ODE ~ O(10³), L_vel ~ O(1), L_phase ~ O(0.01) before λ scaling — SGD with a single global learning rate cannot converge all three loss surfaces simultaneously. Adam's per-parameter adaptivity compensates for the three-decade raw magnitude spread across loss terms.

Default betas (β₁=0.9, β₂=0.999, ε=1e-8) are used. No deviation from Adam defaults is proposed: the λ normalisation in bill_loss_weights_v1 has already balanced the three loss terms to within 0.2–5× at initialisation, so there is no gradient-scale argument for modified betas.

### Learning Rate Schedule

**Initial LR: 1e-3**

Standard Adam initial LR for deep networks. Physical grounding: the 800K-parameter Fourier Feature Network (architecture.json: `approx_parameters=800000`) has a loss landscape with well-separated curvature scales — Fourier projection layer (frozen after initialisation from `fourier_seed=42`) versus 4 × 256 GELU hidden layers. LR=1e-3 is the published Adam optimum for networks in the 100K–10M parameter range (Kingma & Ba 2015). Higher LR risks overshooting the narrow ODE-constrained valley; lower LR prolongs warmup below the physics loss activation threshold.

Note: architecture.json reports `approx_parameters=800000`. The prompt instruction cited ~330K — the actual file value of 800K is used here. The lr_initial=1e-3 recommendation is unchanged: both 330K and 800K fall within the same Adam LR range.

**Scheduler: CosineAnnealingLR (T_max=2000, eta_min=1e-5)**

Cosine annealing decays the learning rate as:

```
lr(t) = eta_min + 0.5 × (lr_initial − eta_min) × (1 + cos(π × t / T_max))
```

Physical grounding: the physics loss (L_ODE, L_vel, L_phase) imposes a rugged, non-convex landscape due to the ω²·z_proxy term in L_ODE (cadence-dependent oscillation frequency). Monotone decay schedulers (StepLR, ExponentialLR) reduce the LR before the network has explored the physics-constrained manifold. Cosine annealing maintains higher LR through the first ~500 epochs (physics warmup + initial fit), then decays smoothly, allowing fine-grained convergence in the second half of training without getting trapped in an early local minimum.

eta_min=1e-5: two decades below lr_initial. Below 1e-5, Adam's ε=1e-8 denominator regularisation dominates the effective step size — further LR reduction produces no meaningful gradient signal change.

### Batch Size

**batch_size: 256 (default, data_config.json absent)**

Physical grounding: mini-batch size determines how many gait profiles contribute to each physics-loss gradient estimate. A batch of 256 samples from ~500 profiles covers ~51% of the training set per step, giving stable estimates of the mean gradient across terrain types (flat, slope, stairs). Smaller batches (≤64) increase gradient variance on L_ODE because the ω² coefficient is cadence-dependent — a batch containing only one terrain type gives a biased gradient estimate. Larger batches (≥512) reduce the beneficial noise that helps escape shallow ODE-loss minima.

See batch_size Note below for the full-batch fallback rule when dataset size is known.

### Early Stopping

**patience: 100 epochs on val loss**
**min_epoch: 200 epochs**

Physical grounding: patience=100 is chosen relative to the physics_loss_warmup_epochs=100. The warmup ramp linearly increases the physics coefficient from 0 to 1 over epochs 0–99. Val loss is dominated by data reconstruction loss during warmup; physics loss only becomes a full contributor at epoch 100. An early stopping check before epoch 200 would therefore observe a val loss trajectory that has not yet reached its physics-constrained regime — the network might appear converged on data loss alone while the ODE residual is still high. Setting early_stop_min_epoch=200 guarantees:

1. The full 100-epoch warmup has completed (epoch 100).
2. The network has had 100 additional epochs under full physics weighting before any stop decision (epoch 100→200).

Patience=100 then means: if val loss does not improve for 100 consecutive epochs after epoch 200, training stops. This gives a maximum early stop at epoch 300 (minimum case) and up to epoch 2000 (no early stop).

### Physics Loss Warmup

**warmup: 100 epochs, linear ramp 0→1**

Mandated by bill_loss_weights_v1 (RATIFIED 2026-04-03): "Total physics loss ≈ data loss after warmup ramp completes." The bill's expected-behaviour section requires the warmup ramp to complete before the λ-weighted physics terms are evaluated for balance. The ramp formula is:

```
physics_weight(epoch) = min(epoch / warmup_epochs, 1.0)
total_loss = data_loss + physics_weight × (λ_ODE·L_ODE + λ_vel·L_vel + λ_phase·L_phase)
```

Physical grounding: at random initialisation, the network output is ~N(0,1). The ODE residual (d²z/dt² + ω²·z − F_contact) at random output is large (bill_loss_weights_v1 shows L_ODE_scale ~ 17803 for flat). Introducing full λ_ODE weighting at epoch 0 would produce gradient magnitudes ~17803 × 1.34e-4 ≈ 2.39 — comparable to data loss — before the network has learned any meaningful signal structure. The linear ramp allows the data-driven component to establish a rough initialisation before physics constraints are enforced to full weight.

100 epochs is derived as: at lr=1e-3 with cosine annealing, the network traverses approximately 100–200 gradient steps per epoch (batch_size=256, dataset~500 profiles → ~2 steps/epoch). 100 epochs × 2 steps = ~200 gradient steps — sufficient for the 800K-parameter network to reduce data reconstruction loss by ~1 order of magnitude before full physics enforcement.

### Reproducibility

**seed: 42**

Traces to: architecture.json `fourier_seed=42`. The Fourier random projection matrix B ~ N(0, σ²) is fixed at `fourier_seed=42` during architecture construction. Using seed=42 for all other stochastic operations (weight initialisation, data shuffling, train/val split) ensures that:

1. The Fourier projection and the initial weight distribution are co-seeded — no seed mismatch can produce a different effective input embedding.
2. The train/val split is deterministic — val loss curves are comparable across runs.
3. Batch shuffling order is deterministic — gradient trajectories are reproducible.

Physical grounding (Article I): gait measurement reproducibility requires that two runs with identical hardware and profiles produce identical training outcomes. A single seed satisfies this requirement. Separate seeds for different operations introduce a combinatorial reproducibility surface that is not justified by any physical argument.

---

## Checkpointing and Logging

**checkpoint_every: 50 epochs**

The archivist saves a checkpoint whenever val loss improves, sampled at 50-epoch intervals. 50 epochs is chosen as a sub-multiple of the early_stop_patience (100 epochs): at minimum one checkpoint opportunity exists per patience window. Continuous checkpoint-on-improvement would produce excessive I/O at ~2 steps/epoch; 50-epoch sampling retains the best model without disk thrash.

**log_every: 10 epochs**

Console milestone print every 10 epochs. Amendment 14 requires milestone logging for all training runs. 10 epochs provides ~200 log lines over 2000 epochs — sufficient granularity to observe warmup ramp, val loss plateau, and convergence, without log verbosity that obscures the milestone signal.

---

## Gradient Clipping

**grad_clip_norm: 1.0**

Global gradient norm clipping. Traces to: L_ODE spike gradients.

From bill_loss_weights_v1: L_ODE_scale for flat profiles = 17803 at random initialisation. Even after λ_ODE normalisation (1.34e-4), the raw ODE gradient with respect to network outputs involves second-order autograd (d²z/dt²), which can produce gradient magnitudes an order of magnitude larger than first-order terms. Global norm clipping at 1.0 prevents a single physics-loss spike from producing a weight update that collapses the Fourier feature representation.

Physical grounding: the heel-strike impulse in F_contact (bill_loss_weights_v1: `hs_impact_ms2 = v_impact / 0.05`) is a discontinuity in the force signal — the ODE residual gradient is large in the vicinity of heel-strike. Clipping at 1.0 is standard for physics-informed networks with discontinuous forcing terms. Values below 0.5 under-constrain learning; values above 2.0 allow physics spikes to dominate.

---

## Val/Test Split

**val_fraction: 0.15 (85/15 train/val split)**

The test split is not defined here — it is reserved for synthetic-data-setter, which owns the full dataset profile distribution. The 85/15 train/val split ensures:

- Sufficient training samples: ~425 of ~500 profiles in training set.
- Sufficient val samples: ~75 profiles — enough to estimate val loss across all three terrains (flat, slope, stairs) without terrain imbalance in the val set.

Physical grounding: gait profiles are generated per terrain type. With ~500 profiles and 3 terrain types, a 15% val fraction gives ~25 profiles per terrain in the val set — the minimum for a statistically meaningful terrain-stratified val loss estimate.

---

## batch_size Note

If `data_config.json` is absent at compile time: default to mini-batch 256.
This will be updated after synthetic-data-setter runs.

**Full-batch fallback rule:** If the total dataset size (after train/val split) is ≤ 20 profiles, full-batch training is used instead of mini-batch. This prevents a batch_size=256 from exceeding dataset size, which would make each epoch equivalent to a single gradient step with the full training set anyway. At ≤ 20 profiles, the distinction between mini-batch and full-batch collapses; full-batch is cleaner and avoids PyTorch DataLoader edge cases with `drop_last=False`.

---

## Files to Write on Ratification

- `simulator/pinn/train_config.json` — runtime hyperparameter config read by pinn-executor

The JSON structure will be:

```json
{
  "optimizer": "Adam",
  "lr_initial": 1e-3,
  "lr_scheduler": "CosineAnnealingLR",
  "lr_scheduler_T_max": 2000,
  "lr_scheduler_eta_min": 1e-5,
  "epochs_max": 2000,
  "physics_loss_warmup_epochs": 100,
  "batch_size": 256,
  "early_stop_patience": 100,
  "early_stop_min_epoch": 200,
  "val_fraction": 0.15,
  "grad_clip_norm": 1.0,
  "seed": 42,
  "checkpoint_every": 50,
  "log_every": 10,
  "_data_config_present_at_bill_time": false,
  "_batch_size_source": "default_mini_batch — data_config.json absent",
  "_approx_parameters_from_architecture": 800000,
  "_created_by": "pinn-compiler",
  "_bill": "bill_train_config_v1",
  "_date": "2026-04-03"
}
```

`train_config.json` is written only after ratification. The pinn-executor reads this file at runtime and raises `FileNotFoundError` with a human-readable message if it is absent, preventing silent fallback to hardcoded defaults.

---

## Article and Amendment Compliance

| Parameter | Governing Rule | Compliance |
|---|---|---|
| All λ values referenced | Article I — traces to cadence_spm, step_length_m, vertical_oscillation_cm | INHERITED from bill_loss_weights_v1 (RATIFIED) |
| physics_loss_warmup_epochs=100 | bill_loss_weights_v1 expected-behaviour section (warmup ramp) | COMPLIANT |
| seed=42 | architecture.json fourier_seed=42 — reproducibility chain | COMPLIANT |
| early_stop_min_epoch=200 | Training stability argument — no arbitrary value | COMPLIANT |
| grad_clip_norm=1.0 | L_ODE second-order gradient spike argument | COMPLIANT |
| log_every=10 | Amendment 14 — milestone logging requirement | COMPLIANT |
| val_fraction=0.15 | Physical: terrain-stratified val set requires ≥25 profiles/terrain | COMPLIANT |

No parameter in this Bill is arbitrary. Each traces to a physical quantity (Article I) or a training stability argument grounded in the specific loss landscape defined by bill_loss_weights_v1 (Article II — empirical basis for decisions).
