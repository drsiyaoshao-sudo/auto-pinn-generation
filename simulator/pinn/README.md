# PINN Generation Workflow

**Constitutional grounding:** Article I (Physics First) + Article II (Learner-in-the-Loop)  
**Layer role:** Layer 1 surrogate — replaces `walker_model.py::generate_imu_sequence()` with continuous-parameter PINN inference.  
**Interface contract (Amendment 3):** Output is `(N, 6) float32` at 208 Hz, columns `[ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]`. Layer 2+ unchanged.

---

## Agent Flow

```
                        PINN GENERATION AGENT FLOW
                   (GaitSense Constitutional Governance)
  ═══════════════════════════════════════════════════════════════════

   LEGISLATURE (Bills)           BUREAUCRACY (Standing Orders)
   ────────────────────          ──────────────────────────────

   ┌─────────────┐
   │ loss-setter │ ── derives L_ODE, L_vel, L_phase
   │ (Stage 1)   │    λ weights from E[L²] at init
   └──────┬──────┘
          │ bill_loss_weights_v1
          ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Review derivation trace
   │  Ratify λ    │      Approve λ values + balance
   └──────┬───────┘
          │ RATIFIED
          ▼
   ┌──────────────┐
   │ pinn-compiler│ ── lr, epochs, warmup,
   │ (Stage 2)    │    batch_size, early stop
   └──────┬───────┘
          │ bill_train_config_v{N}
          ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Review hyperparameters
   │  Ratify cfg  │
   └──────┬───────┘
          │ RATIFIED
          ▼
   ┌─────────────────┐
   │synthetic-data-  │ ── 500 random profiles
   │setter (Stage 3) │    + 4 anchors, distributions
   └────────┬────────┘
            │ bill_data_config_v1
            ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Review param distributions
   │  Ratify data │      terrain splits, anchor roles
   └──────┬───────┘
          │ RATIFIED
          │
          │                ┌──────────────────┐
          │                │   layer-setter   │ ── Fourier vs MLP
          │                │   (parallel)     │    architecture.json
          │                └────────┬─────────┘
          │                         │ pinn_model.py
          ▼                         ▼
   ┌───────────────────────────────────────────────┐
   │           synthetic-data-generator            │
   │  walker_model.py ──► 504 profiles × 208 Hz    │
   │  training_data/{X,Y,t}_{train,val,test}.npy   │
   └────────────────────┬──────────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────────┐
   │             physics-reviewer                  │
   │  derivation trace table + 4-panel balance     │
   │  plot ──► physics_balance_v{N}.png            │
   └────────────────────┬──────────────────────────┘
                        │
                        ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Review balance plot
   │  Confirm     │      Confirm λ·L within 0.1–10×
   └──────┬───────┘
          │ CONFIRMED
          ▼
   ┌───────────────────────────────────────────────┐
   │               pinn-executor                   │
   │  train_pinn.py ──► MPS/CPU training loop      │
   │  Amendment 14 milestone logs every 10 epochs  │
   │  NaN/Inf guard ── Amendment 7 three-strike    │
   │  best_v{N}.pt saved on val loss improvement   │
   └──────┬──────────────────────┬─────────────────┘
          │                      │
          ▼                      ▼
   ┌─────────────┐      ┌────────────────┐
   │  train-sum  │      │ pinn-archivist │
   │  loss curve │      │ SHA-256 hash   │
   │  4-panel    │      │ manifest.json  │ ◄── Amendment 16
   │  plots      │      │ pinn_registry  │
   └──────┬──────┘      └───────┬────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Review loss curves
   │  Review      │      Check best epoch > warmup
   └──────┬───────┘      (success criterion)
          │
          │  if best_epoch ≤ warmup ──► pinn-compiler (new Bill)
          │  if PASS ────────────────────────────┐
          ▼                                      │
   ┌───────────────────────────────────────────────┐
   │               pinn-validator                  │
   │  Check 1: fidelity ≤15% per-axis (Amend. 19) │
   │  Check 2: signal plots 4 profiles (Amend. 11) │
   │  Check 3: VABS.F32 si_true=25% → SI>10%      │
   │  ── Python screening only (not Renode)        │
   └────────────────────┬──────────────────────────┘
                        │
                        ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Review signal plots
   │  Approve     │      Confirm fidelity + VABS
   │  checkpoint  │
   └──────┬───────┘
          │ APPROVED
          ▼
   ┌─────────────────┐
   │ pinn-grid-      │ ── axes (primitive-traced)
   │ controller      │    min/max, clinical hypothesis
   │ (Legislature)   │    Renode assertion
   └────────┬────────┘
            │ bill_grid_search_v{N}
            ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Ratify search domain
   │  Ratify grid │
   └──────┬───────┘
          │ RATIFIED
          ▼
   ┌───────────────────────────────────────────────┐
   │           batch PINN inference                │
   │  grid_search/run_grid.py ──► Python screening │
   │  boundary candidates ──► pinn-validator       │
   │  plausible candidates ──► Renode (Amend. 18)  │
   └────────────────────┬──────────────────────────┘
                        │
                        ▼
   ┌──────────────┐
   │  [HUMAN]     │ ◄─── Confirm each boundary
   │  Confirm     │      verbatim (Amend. 18)
   │  boundary    │
   └──────┬───────┘
          │ CONFIRMED
          ▼
   ┌───────────────────────────────────────────────┐
   │           case_law.md entry                   │
   │  parameter values + Renode evidence           │
   │  + firmware Bill if clinically reachable      │
   └───────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════════
  LAYER BOUNDARY (Amendment 3)
  PINN output ≡ generate_imu_sequence(): (N,6) float32 @ 208 Hz
  Layer 2 (imu_model.py) and above: UNCHANGED
  ═══════════════════════════════════════════════════════════════════

  AGENT ROLES
  ───────────
  [HUMAN]               Decision gate — Article II unconditional
  loss-setter           Legislature  — derives + proposes λ weights
  pinn-compiler         Legislature  — proposes training hyperparams
  synthetic-data-setter Legislature  — proposes dataset config
  layer-setter          Bureaucracy  — writes architecture from spec
  synthetic-data-gen    Bureaucracy  — generates data from ratified config
  physics-reviewer      Bureaucracy  — produces evidence, makes no ruling
  pinn-executor         Bureaucracy  — runs training loop
  train-sum             Bureaucracy  — plots loss curves
  pinn-archivist        Bureaucracy  — hashes + registers checkpoints
  pinn-validator        Bureaucracy  — blocks or passes (cannot approve)
  pinn-grid-controller  Legislature  — proposes grid search domains
```

---

## What This Is

`walker_model.py` generates IMU signals for 4 discrete walker profiles. The PINN generalises this to a **continuous parameter space** — enabling the `pinn-grid-controller` to search for gait algorithm failure boundaries that the 4 fixed profiles cannot reach.

The PINN takes 10 walking parameters (the three Article I primitives + terrain geometry + sensor fit) and a normalised step time `t ∈ [0,1]`, and outputs a 6-channel IMU signal for one complete step at 208 Hz.

---

## Architecture

**Model:** Fourier Feature Network (`pinn_model.py`)

| Parameter | Value |
|---|---|
| Input dim | 10 (conditioning) + 1 (time) = 11 |
| Fourier projection | dim=256, σ=1.0, seed=42 → 512-dim embedding |
| Hidden layers | 4 × 256, GELU activation |
| Output dim | 6 (ax, ay, az, gx, gy, gz) |
| Output activation | None (unbounded physical quantities) |
| Trainable parameters | 330,246 |

**Why Fourier Features?** Plain MLPs exhibit spectral bias — they learn low-frequency components first and struggle with the sharp heel-strike impulse (Gaussian with σ≈25 ms) and the non-sinusoidal sigmoid toe-roll loading on stairs. Random Fourier features project the input into a frequency-rich space, allowing the network to represent these sharp transitions without spectral bias.

**Input fields (10-dim conditioning vector):**

| Index | Field | Unit | Article I primitive |
|---|---|---|---|
| 0 | `cadence_spm` | steps/min | Cadence (direct) |
| 1 | `step_length_m` | m | Step Length (direct) |
| 2 | `vertical_oscillation_cm` | cm | Vertical Oscillation (direct) |
| 3 | `slope_deg` | deg | Vertical Oscillation (via CoM path geometry) |
| 4 | `stance_frac` | — | Cadence (via step_period_s) |
| 5 | `si_stance_true_pct` | % | Cadence (stance duration asymmetry) |
| 6 | `mounting_offset_deg` | deg | Vertical Oscillation (sensor frame rotation) |
| 7 | `loose_fit_attenuation` | — | Vertical Oscillation (HS impulse attenuation) |
| 8 | `step_variability_ms` | ms | Cadence (temporal noise on step_period_s) |
| 9 | `terrain_int` | 0/1/2 | All three (conditions all distributions) |

Terrain encoding: `flat=0`, `slope=1`, `stairs=2`

---

## Physics Loss (Amendment 17)

Three physics loss terms, each algebraically traced to Article I primitives. Weights derived from `E[L²]` at initialisation — not empirically tuned. Ratified in `bill_loss_weights_v1`.

**L_ODE — CoM Vertical Oscillation ODE** (`λ = 1.3425e-04`)
```
d²z/dt² + ω²·z_proxy = F_contact(t)
  ω = 2π × (cadence_spm / 60)           [traces to: cadence_spm]
  F_contact = hs_impact × Gaussian(t)   [traces to: vertical_oscillation_cm]
```

**L_vel — Horizontal Velocity Constraint** (`λ = 2.8691`)
```
v_x_expected = (cadence_spm / 60) × step_length_m   [traces to: cadence_spm, step_length_m]
v_x_pred     = mean(ax_pred) × step_period_s
Loss = (v_x_pred − v_x_expected)²
```
> **Critical:** `ax_pred` must remain an attached tensor. An `assert ax_pred.requires_grad` guard in `physics_loss.py` catches silent zero-gradient failures at runtime.

**L_phase — Stance/Swing Timing Constraint** (`λ = 78.472`)
```
stance_frac: flat=0.60, slope=0.62, stairs=0.65   [physiological 60/40 split, Amendment 15]
Loss = hinge(mean_gy_stance)² + 0.5 × boundary²
```

**Total loss:**
```
L_total = L_data + physics_ramp(epoch) × (λ_ODE·L_ODE + λ_vel·L_vel + λ_phase·L_phase)
```
Physics ramp: linear 0→1 over `physics_loss_warmup_epochs` (prevents ODE gradients from dominating random initialisation).

**Amendment 20 constraint:** The warmup phase must run until each of `l_ode`, `l_vel`, and `l_phase` shows a net downward trend over at least 10 consecutive logged epochs, and physics weighted contribution must be ≥ 80% of total loss during this phase. The data-dominant phase may not begin until this criterion is documented in the training log. `pinn-executor` enforces and logs the criterion; `pinn-compiler` documents the warmup schedule in the ratified Bill.

---

## Training Dataset

Generated by `generate_training_data.py` using `walker_model.py` as physics engine. Ratified in `bill_data_config_v1`.

| Split | Profiles | Samples |
|---|---|---|
| Train | 350 random | 72,800 |
| Val | 75 random + 4 anchors | 16,432 |
| Test | 75 random (withheld) | 15,600 |
| **Total** | **504** | **104,832** |

**Terrain distribution (train):** flat 59.4% / slope 21.1% / stairs 19.4%  
**Terrain-conditional distributions:** `vertical_oscillation_cm` uses N(4.5, 1.5²) for flat/slope and N(15.0, 3.0²) for stairs — the primary Article I terrain discriminator.

**4 anchor profiles** (always in validation, never training):

| Anchor | Key params | Role |
|---|---|---|
| `flat` | cadence=105, vert_osc=4.0cm | Baseline fidelity reference |
| `bad_wear` | offset=20°, attenuation=0.55 | Out-of-distribution stress test |
| `stairs` | cadence=70, vert_osc=18.0cm | Extreme oscillation regime |
| `slope` | cadence=95, slope=10° | Terrain projection test |

`bad_wear` mounting offset (20°) and attenuation (0.55) are **intentionally outside the training distribution** to test physics generalisation.

---

## Training Configuration

Current config: `train_config.json` (v2, ratified `bill_train_config_v2`)

| Parameter | v1 | v2 | Reason for change |
|---|---|---|---|
| `lr_initial` | 1e-3 | **3e-4** | ODE loss creates narrow valleys; lower LR stays in basin |
| `physics_loss_warmup_epochs` | 100 | **300** | Data plateau reached at epoch 20 in v1; extended warmup allows data-driven initialisation |
| `early_stop_min_epoch` | 200 | **500** | warmup(300) + patience(100) + convergence window(100) |
| `epochs_max` | 2000 | 2000 | unchanged |
| `batch_size` | 256 | 256 | unchanged |
| `grad_clip_norm` | 1.0 | 1.0 | unchanged |
| `seed` | 42 | 42 | unchanged |

---

## Training Run History

| Run | Best epoch | Best val loss | Physics weight at best | Status |
|---|---|---|---|---|
| v1 | 20 / 200 | 0.4422 | 0.20 | Early stop — best during warmup |
| v2 | 35 / 500 | 0.4228 | 0.12 | Early stop — best during warmup |

**Current diagnostic:** Both runs found a data-only minimum before full physics weighting. L_vel is the dominant physics term at full weight (λ·L_vel ≈ 0.37 vs data_loss ≈ 0.43). Pending investigation of whether `walker_model.py` generated data satisfies the L_vel constraint — if not, a structural data/physics tension exists and `bill_loss_weights_v2` may be required.

**Success criterion for next run:** Best checkpoint epoch must be > `physics_loss_warmup_epochs`. A best epoch during warmup indicates the network is fitting data-only and the physics constraints are not converging.

---

## File Reference

```
simulator/pinn/
├── pinn_model.py              — FourierFeatureEmbedding + PINNModel; forward(x,t) → (N,6)
├── physics_loss.py            — PhysicsLoss: l_ode, l_vel (assert guard), l_phase, total_loss
├── train_pinn.py              — Training loop; reads train_config.json; --max-epochs override for dry runs
├── generate_training_data.py  — Synthetic dataset generator; reads data_config.json
├── architecture.json          — Model topology metadata (layer-setter output)
├── train_config.json          — Runtime hyperparameters (pinn-compiler output, Bill-locked)
├── data_config.json           — Dataset parameter distributions (synthetic-data-setter output, Bill-locked)
├── training_data/
│   ├── X_{train,val,test}.npy      — Conditioning vectors (350/79/75, 10-dim)
│   ├── Y_{train,val,test}.npy      — IMU sequences (N, 208, 6)
│   ├── t_{train,val,test}.npy      — Normalised time (208,) linspace[0,1]
│   ├── anchor_{flat,bad_wear,stairs,slope}.npy  — 4 fixed reference sequences
│   ├── anchor_profiles.json        — WalkerProfile fields for 4 anchors
│   └── dataset_manifest.json       — Counts, shapes, splits, terrain distribution
└── checkpoints/
    ├── best_v1.pt             — Best checkpoint (currently v2 run, pending rename fix)
    ├── run_v1_metrics.jsonl   — Per-epoch val metrics (epoch, losses, ramp, lr)
    └── manifest.json          — SHA-256 hashes + parameter provenance (Amendment 16)

docs/gaitsense_code/
├── bills/
│   ├── bill_loss_weights_v1.md    — λ derivation, RATIFIED 2026-04-03
│   ├── bill_train_config_v1.md    — Hyperparameters v1, RATIFIED 2026-04-03
│   ├── bill_train_config_v2.md    — Hyperparameters v2 (lower LR, extended warmup), RATIFIED 2026-04-03
│   └── bill_data_config_v1.md     — Dataset config, RATIFIED 2026-04-03
└── pinn_registry.md               — Immutable checkpoint registry (Amendment 16)

docs/executive_branch_document/plots/pinn_training/
├── physics_balance_v1.png         — 4-panel: loss balance, warmup ramp, λ ranges, terrain dist
├── physics_review_v1.md           — Derivation trace table (physics-reviewer output)
├── train_summary_v1.png           — 4-panel: loss curves, components, ramp/LR, summary table
└── train_summary_v1.md            — Training outcome summary (train-sum output)
```

---

## How to Reproduce

**1. Install dependencies**
```bash
pip install torch==2.2.2 scikit-learn==0.24.2
```

**2. Generate training data** (requires `simulator/walker_model.py`)
```bash
python simulator/pinn/generate_training_data.py
```
Outputs 13 files to `simulator/pinn/training_data/`. Deterministic at seed=42.

**3. Run training**
```bash
# Dry run (50 epochs, ~15s on MPS)
python simulator/pinn/train_pinn.py --max-epochs 50

# Full run (up to 2000 epochs with early stopping)
python simulator/pinn/train_pinn.py
```
Checkpoint saved to `simulator/pinn/checkpoints/best_v1.pt` on val loss improvement.  
Metrics logged to `simulator/pinn/checkpoints/run_v1_metrics.jsonl`.

**4. Validate checkpoint** (after pinn-validator Step 10)
```bash
# Amendment 19 fidelity check (≤15% per-axis error vs walker_model.py)
# Amendment 11 signal plots
# VABS.F32 pathological check (si_true=25% → SI>10%)
# Invoked by pinn-validator agent — see .claude/agents/pinn-validator.md
```

---

## Constitutional Governance

All design decisions in this workflow are governed by CLAUDE.md. Key constraints:

- **Article I:** Every parameter traces to `vertical_oscillation_cm`, `cadence_spm`, or `step_length_m`
- **Amendment 15:** All statistical constants derived from population data, not empirical tuning
- **Amendment 16:** Every checkpoint SHA-256 hashed in `manifest.json` before use
- **Amendment 17:** Loss weights derived from physics, locked by Bill before training
- **Amendment 18:** Grid search boundary findings require Renode confirmation before Case Law
- **Amendment 19:** PINN must reproduce all 4 anchor profiles within 15% peak error per axis before grid search
- **Amendment 20:** Physics-dominant warmup (≥ 80% of total loss) must precede any data-dominant phase; each physics loss term (`l_ode`, `l_vel`, `l_phase`) must show a net downward trend over at least 10 consecutive logged epochs before the data phase begins; `pinn-executor` enforces and logs the transition criterion; warmup schedule locked in the ratified Bill (`pinn-compiler`). *Grounds: PINN Data Loss Dominance Hearing + z_proxy Collapse Case (2026-04-03).*

Any change to loss weights, training config, or dataset parameters requires a new ratified Bill. No hardcoded values in any source file — all constants read from JSON config at runtime.
