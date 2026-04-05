Run the full PINN model training pipeline from data generation through validation.

Usage: /model-train [stage]

Stages (run in order — each requires human confirmation before the next begins):
  all       — full pipeline from data generation to validation (default)
  data      — data generation only (synthetic-data-setter → synthetic-data-generator)
  design    — architecture + loss design (layer-setter → loss-setter → physics-reviewer)
  compile   — hyperparameter lock (pinn-compiler)
  train     — training run (pinn-executor + pinn-monitor → train-sum)
  archive   — checkpoint archival (pinn-archivist)
  validate  — validation checks (pinn-validator)

If no stage given, runs the full pipeline with human gates between each phase.

---

## Amendment 21 Pre-Flight Checklist

**This checklist runs BEFORE any agent is invoked. All three items must pass.**

Print and verify each item. If any item fails, stop and escalate — do not proceed to data generation.

```
AMENDMENT 21 PRE-FLIGHT (Data–Physics–Model Triad Alignment)
─────────────────────────────────────────────────────────────
□ Item 1 — Physics model matches data generator
    Action: read simulator/walker_model.py __post_init__
    Check:  loss terms derive from empirical equations
            (Gaussian heel-strike, peak_angvel = (100+65v)×slope_factor×stairs_factor)
            NOT from linearised spring-mass ODE
    Verify: ODE residual diagnostic — residual(true data) < residual(az=0 baseline)
            If residual(true) > residual(zero): physics is wrong — do NOT proceed

□ Item 2 — Embedding basis spans cadence frequency content
    Check:  embedding covers frequencies [1, H_max] cycles/step
            where H_max derives from cadence_spm range in data_config.json
            Chebyshev T1–T5 on normalised t covers 5 harmonics — acceptable for cadence 70–120 spm
            Random Fourier B at σ=1.0 on (x,t) concatenated — NOT acceptable

□ Item 3 — Model capacity ≤ 0.1 × N_independent_profiles
    N_independent_profiles = number of distinct conditioning vectors (NOT × T_steps)
    Read data_config.json for N_random_profiles
    N_train = N_random_profiles × train_fraction + 4 anchor profiles
    Check:  param_count / N_train ≤ 0.1 (target: ≤ 3.4 params/profile from v21 baseline)
    If capacity too high: reduce hidden_dim or increase N_random_profiles
─────────────────────────────────────────────────────────────
Print: AMENDMENT 21 PRE-FLIGHT: [PASS / FAIL — item N]
```

Reference: `docs/gaitsense_code/lesson_nine_run_diagnostic.md` — nine-run failure arc

---

## Phase 1 — Data Generation

**Agents: synthetic-data-setter → [JUSTICE GATE] → synthetic-data-generator**

### 1a. synthetic-data-setter
Invoke `synthetic-data-setter` to:
- Define N_random_profiles, parameter space bounds per axis, train/val/test split ratios
- Write `simulator/pinn/training_data/data_config.json`
- Write `docs/gaitsense_code/bills/bill_dataset_<version>.md`
- Requires a Bill — dataset configuration is a calibration decision (Amendment 13)

**[JUSTICE GATE 1]** Present the Bill to the Justice. Wait for explicit ratification before continuing.
Print: `GATE 1: Awaiting Justice ratification of dataset Bill.`

### 1b. synthetic-data-generator
After Bill is ratified, invoke `synthetic-data-generator` to:
- Sample WalkerProfiles within bounds from data_config.json
- Call walker_model.py::generate_imu_sequence() for each profile
- Save numpy arrays with provenance metadata
- Print generation summary (N profiles, per-terrain counts, any errors)

**Stairs coverage check:** Print count of stairs profiles in training set.
If stairs < 15% of total: warn — stairs underrepresented (v21 lesson: stairs failure was data coverage, not architecture).

---

## Phase 2 — Architecture and Loss Design

**Agents: layer-setter → loss-setter → physics-reviewer → [JUSTICE GATE]**

### 2a. layer-setter
Invoke `layer-setter` to:
- Define and write neural network architecture to `simulator/pinn/pinn_model.py`
- Select embedding type: Chebyshev T1–T5 on t (preferred, bill_architecture_v21a baseline)
  or alternative if a new Bill justifies it
- Verify input dimension = 10 (from walker_model.py WalkerProfile fields)
- Freeze architecture before any training begins

### 2b. loss-setter
Invoke `loss-setter` to:
- Derive physics loss terms from walker_model.py actual equations (not linearisations)
- Write `simulator/pinn/physics_loss.py`
- Write `docs/gaitsense_code/bills/bill_physics_loss_<version>.md`
- Requires a Bill (Amendment 13 + Amendment 17)

Mandatory diagnostic output from loss-setter:
```
Physics residual diagnostic (Amendment 21 Item 1):
  Profile   | residual(true data) | residual(az=0) | Verdict
  ──────────┼────────────────────┼────────────────┼────────
  flat      |                    |                |
  stairs    |                    |                |
  slope     |                    |                |
  bad_wear  |                    |                |
Rule: residual(true) < residual(zero) required for each term.
Any row where true > zero: remove that loss term.
```

### 2c. physics-reviewer
Invoke `physics-reviewer` to:
- Compute numerical loss term values across all 4 anchor profiles
- Plot loss magnitudes vs primitive-derived expectations
- Print full derivation trace table
- Does not rule — output is evidence for Justice review

**[JUSTICE GATE 2]** Present physics-reviewer output to Justice. Wait for confirmation before compiling.
Print: `GATE 2: Awaiting Justice review of physics derivation evidence.`

---

## Phase 3 — Hyperparameter Lock

**Agent: pinn-compiler → [JUSTICE GATE]**

Invoke `pinn-compiler` to:
- Define and lock all training hyperparameters
- Write `simulator/pinn/train_config.json`
- Write `docs/gaitsense_code/bills/bill_hyperparams_<version>.md`
- Requires a Bill (Amendment 13)

Mandatory capacity check output:
```
Amendment 21 Item 3 — Capacity check:
  N_independent_profiles (train): [N]
  Model parameters:               [N]
  Params / profile:               [N]  (must be ≤ 0.1 for safe, ≤ 3.4 from v21 baseline)
  Verdict: [PASS / FAIL]
```

**[JUSTICE GATE 3]** Present the hyperparameter Bill to the Justice. Wait for ratification.
Print: `GATE 3: Awaiting Justice ratification of hyperparameter Bill.`

---

## Phase 4 — Training Run

**Agents: pinn-monitor (setup) → pinn-executor → train-sum → [JUSTICE GATE]**

### 4a. pinn-monitor
Invoke `pinn-monitor` to write training callback files:
- Per-epoch metric log file (fresh — stateless between runs)
- Checkpoint-on-improvement handler
- Early stopping trigger

### 4b. pinn-executor
Invoke `pinn-executor` to run the training loop.
Prerequisites confirmed: layer-setter ✓, loss-setter ✓, pinn-compiler ✓, physics-reviewer ✓, Justice confirmed ✓.

pinn-executor must NOT invoke layer-setter or loss-setter during the run unless Justice explicitly requests it.

### 4c. train-sum
After pinn-executor completes, invoke `train-sum` to:
- Generate loss curve plots (total loss + each physics component separately)
- Print final metrics table (best epoch, best val loss, early stop status)

**[JUSTICE GATE 4]** Present train-sum output to Justice for review.
Print: `GATE 4: Awaiting Justice review of training results.`

---

## Phase 5 — Archival

**Agent: pinn-archivist → [JUSTICE GATE]**

After Justice reviews training results, invoke `pinn-archivist` to:
- SHA-256 hash the best checkpoint
- Write to manifest with full provenance (run_id, bill references, architecture, loss config)
- Update PINN model registry
- Implements Amendment 16

**[JUSTICE GATE 5]** Confirm archival is complete before validation begins.
Print: `GATE 5: Checkpoint archived. Awaiting Justice confirmation to begin validation.`

---

## Phase 6 — Validation

**Agent: pinn-validator → [JUSTICE GATE]**

Invoke `pinn-validator` to run all three mandatory checks:

```
Validation checks (pinn-validator):
  □ Amendment 11 — Signal plots: generate IMU plots for all 4 anchor profiles
  □ Amendment 19 — Fidelity: per-axis error < 15% vs walker_model.py on all 4 profiles
  □ VABS.F32     — Pathological check: si_true=25% must produce SI > 10%
```

pinn-validator can only BLOCK or PASS — it cannot approve.
pinn-validator output is evidence. The Justice approves.

**[JUSTICE GATE 6 — FINAL]** Present pinn-validator output to Justice.
If all checks pass and Justice confirms: Stage 2 gate closes, stage-compactor may be invoked.
If any check fails: declare a hearing or propose a new Bill before re-running.
Print: `GATE 6 (FINAL): Awaiting Justice approval of validation results.`

---

## Pipeline Summary

```
  /model-train
      │
      ├── [AMENDMENT 21 PRE-FLIGHT]
      │
      ├── Phase 1: Data
      │     synthetic-data-setter → [GATE 1] → synthetic-data-generator
      │
      ├── Phase 2: Design
      │     layer-setter → loss-setter → physics-reviewer → [GATE 2]
      │
      ├── Phase 3: Compile
      │     pinn-compiler → [GATE 3]
      │
      ├── Phase 4: Train
      │     pinn-monitor → pinn-executor → train-sum → [GATE 4]
      │
      ├── Phase 5: Archive
      │     pinn-archivist → [GATE 5]
      │
      └── Phase 6: Validate
            pinn-validator → [GATE 6 FINAL]
```

**No phase may begin before its preceding Justice gate is cleared.**
**No agent in this pipeline may edit source files outside its designated scope.**
**If Amendment 21 pre-flight fails, the pipeline stops — no data is generated.**

---

## Constitutional References

- Article I: all loss terms must trace to a first-order gait primitive
- Article II: no phase advances without human Justice confirmation
- Amendment 13: dataset configuration and hyperparameters require Bills
- Amendment 16: checkpoint archival required before validation (pinn-archivist)
- Amendment 17: loss weight changes require a Bill
- Amendment 19: fidelity check < 15% per-axis error on all 4 profiles
- Amendment 21: Data–Physics–Model Triad Alignment pre-flight mandatory
- Lesson: `docs/gaitsense_code/lesson_nine_run_diagnostic.md`

Now parse "$ARGUMENTS":
  If a stage name is given (data / design / compile / train / archive / validate), run only that phase.
  If "all" or no argument given, run the full pipeline starting with the Amendment 21 pre-flight.
  Print the pipeline summary at the start so the Justice knows what is coming.
