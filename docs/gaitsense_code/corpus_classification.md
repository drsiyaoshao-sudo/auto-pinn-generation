# GaitSense — RAG Corpus Classification

**Branch:** hybrid-model  
**Date:** 2026-04-04  
**Purpose:** Defines the sensitivity tier for every document and source file in the corpus.  
The tier governs which LLM can retrieve it, and what can be forwarded upstream.

---

## Tier Definitions

```
  PRIVATE  ── must not leave local infrastructure
             contains: derivation formulas, implementation equations,
             patient-derived distributions, patentable IP, raw signal data

  DERIVED-OK ── abstracted from private data; safe to send to cloud
               contains: scalar outputs of private computations,
               summary statistics, loss values, hashes
               rule: opaque keys only — no variable names that reveal formula structure

  PUBLIC   ── safe for cloud retrieval
             contains: governance rules, constitutional text, procedural
             records, architecture decisions, agent definitions
             no raw data, no derivation formulas, no patient statistics
```

---

## Classification Table

### Governance & Constitutional

| File | Tier | Reason |
|---|---|---|
| `CLAUDE.md` | PUBLIC | Constitutional rules only — no data |
| `docs/gaitsense_code/amendments.md` | PUBLIC | Governance rules — no data |
| `docs/gaitsense_code/case_law.md` | PUBLIC | Ruling records — no raw data or formulas |
| `docs/gaitsense_code/hearing_procedure_abstract.md` | PUBLIC | Procedural template only |
| `docs/gaitsense_code/judicial-process.md` | PUBLIC | Flowchart only |
| `docs/gaitsense_code/evidence_session.md` | PUBLIC | Architecture diagram only |
| `docs/gaitsense_code/demo_cicd_flowchart.md` | PUBLIC | Demo record — no data |

### Agent Definitions

| File | Tier | Reason |
|---|---|---|
| `.claude/agents/attorney-A.md` | PUBLIC | Role definition only |
| `.claude/agents/attorney-B.md` | PUBLIC | Role definition only |
| `.claude/agents/judicial-clerk.md` | PUBLIC | Role definition only |
| `.claude/agents/simulator-operator.md` | PUBLIC | Role definition only |
| `.claude/agents/plot-orchestrator.md` | PUBLIC | Role definition only |
| `.claude/agents/plotter.md` | PUBLIC | Role definition only |
| `.claude/agents/uart-reader.md` | PUBLIC | Role definition only |
| `.claude/agents/train-sum.md` | PUBLIC | Role definition only |
| `.claude/agents/pinn-executor.md` | PUBLIC | Role definition only |
| `.claude/agents/pinn-monitor.md` | PUBLIC | Role definition only |
| `.claude/agents/pinn-archivist.md` | PUBLIC | Role definition only |
| `.claude/agents/pinn-validator.md` | PUBLIC | Role definition only |
| `.claude/agents/pinn-grid-controller.md` | PUBLIC | Role definition only |
| `.claude/agents/pinn-compiler.md` | PUBLIC | Role definition only |
| `.claude/agents/layer-setter.md` | PUBLIC | Role definition only |
| `.claude/agents/loss-setter.md` | PUBLIC | Role definition — the *rules* for derivation, not the derivation itself |
| `.claude/agents/physics-reviewer.md` | PUBLIC | Role definition only |
| `.claude/agents/synthetic-data-setter.md` | PUBLIC | Role definition only |
| `.claude/agents/synthetic-data-generator.md` | PUBLIC | Role definition only |
| `.claude/agents/stage-compactor.md` | PUBLIC | Role definition only |
| `.claude/agents/package-manager.md` | PUBLIC | Role definition only |

### Skill Commands

| File | Tier | Reason |
|---|---|---|
| `.claude/commands/plot-evidence.md` | PUBLIC | Interface contract only |
| `.claude/commands/plot-profile.md` | PUBLIC | Interface contract only |
| `.claude/commands/plot-training.md` | PUBLIC | Interface contract only |

### Bills

| File | Tier | Reason |
|---|---|---|
| `docs/gaitsense_code/bills/bill_train_config_v1.md` | PUBLIC | Hyperparameter values only — no derivation formulas |
| `docs/gaitsense_code/bills/bill_train_config_v2.md` | PUBLIC | Hyperparameter values only |
| `docs/gaitsense_code/bills/bill_train_config_v3.md` | PUBLIC | Hyperparameter values only |
| `docs/gaitsense_code/bills/bill_loss_weights_v1.md` | **PRIVATE** | Contains full ODE derivation formulas and λ computation — trade secret |
| `docs/gaitsense_code/bills/bill_data_config_v1.md` | **PRIVATE** | Contains parameter distributions derived from population assumptions — patentable |
| `docs/gaitsense_code/bills/bill_physics_loss_v2.md` | **PRIVATE** | Contains true double-integration derivation and algebraic collapse proof — trade secret |

### Source Code

| File | Tier | Reason |
|---|---|---|
| `simulator/pinn/pinn_model.py` | **PRIVATE** | Architecture implementation — Fourier feature encoding detail is IP |
| `simulator/pinn/physics_loss.py` | **PRIVATE** | ODE equations, velocity constraint, phase constraint implementations — core trade secret |
| `simulator/pinn/train_pinn.py` | **PRIVATE** | Training loop with physics ramp schedule — implementation detail |
| `simulator/pinn/generate_training_data.py` | **PRIVATE** | Data generation logic reveals population parameter assumptions |
| `simulator/walker_model.py` | **PRIVATE** | Biomechanical derivation chain — hs_impact, peak_angvel formulas — core IP |
| `simulator/pinn/__init__.py` | PUBLIC | Empty init |

### Configuration Files

| File | Tier | Reason |
|---|---|---|
| `simulator/pinn/architecture.json` | PUBLIC | Topology metadata only (dims, activation) — no weights or formulas |
| `simulator/pinn/train_config.json` | DERIVED-OK | λ values present but as scalars; opaque key rule applies when forwarding |
| `simulator/pinn/data_config.json` | **PRIVATE** | Full parameter distributions with μ, σ per field — reveals population model |

### Training Evidence & Outputs

| File | Tier | Reason |
|---|---|---|
| `simulator/pinn/training_logs/*.jsonl` | DERIVED-OK | Per-epoch loss scalars — no patient data, no formulas; safe as summary |
| `simulator/pinn/checkpoints/*.pt` | **PRIVATE** | Model weights encode the private physics derivations implicitly |
| `docs/gaitsense_code/pinn_registry.md` | DERIVED-OK | SHA-256 hashes + scalar metrics only |
| `docs/executive_branch_document/plots/pinn_training/train_summary_v1.md` | DERIVED-OK | Scalar loss values + epoch counts — no formulas, no patient data |
| `docs/executive_branch_document/plots/pinn_training/physics_review_v1.md` | **PRIVATE** | Contains λ·L per profile with formula context — reveals derivation structure |
| `docs/executive_branch_document/plots/pinn_training/*.png` | DERIVED-OK | Visual summaries — no extractable raw data |
| `docs/executive_branch_document/plots/*.png` | DERIVED-OK | Signal diagnostic plots — visual only |
| `docs/executive_branch_document/plots/STEP2_ZEROCROSS_EVIDENCE.md` | DERIVED-OK | Timing measurements only — no patient identity |

### Executive Branch Documents

| File | Tier | Reason |
|---|---|---|
| `docs/executive_branch_document/handoff.md` | PUBLIC | Algorithm description + bring-up procedure — no patient data |
| `docs/executive_branch_document/algorithm_hunting_stair_walker.md` | PUBLIC | Algorithm analysis — no patient data |
| `docs/executive_branch_document/bug_receipt.md` | PUBLIC | Bug records — no patient data |
| `docs/executive_branch_document/hw_bom.md` | PUBLIC | Hardware BOM |
| `docs/executive_branch_document/sw_bom.md` | PUBLIC | Software BOM |

### Training Data (not yet in repo — future classification)

| Resource | Tier | Reason |
|---|---|---|
| `simulator/pinn/training_data/X_train.npy` | **PRIVATE** | Raw conditioning vectors — patient gait parameter space |
| `simulator/pinn/training_data/Y_train.npy` | **PRIVATE** | Raw IMU signal sequences — derived from patient movement |
| `simulator/pinn/training_data/anchor_*.npy` | **PRIVATE** | Fixed reference profiles — parameter values are IP |
| `simulator/pinn/training_data/dataset_manifest.json` | DERIVED-OK | Counts, shapes, terrain distribution — no raw values |
| `simulator/pinn/training_data/anchor_profiles.json` | **PRIVATE** | Exact WalkerProfile parameter values — core IP |

---

## Forwarding Rules

```
  PRIVATE  →  local LLM only
              never forwarded upstream
              exception: scalar outputs may be forwarded as DERIVED-OK
              with opaque keys (see masking rule below)

  DERIVED-OK → may be forwarded to cloud LLM
               opaque key rule: variable names that reveal formula
               structure must be replaced before forwarding
               e.g. {"lambda_ode": 1.34e-4} → {"w0": 1.34e-4}

  PUBLIC   → may be retrieved by any LLM (local or cloud)
             no transformation required before forwarding
```

### Masking Rule — loss-setter two-step

The λ derivation lives in `bill_loss_weights_v1.md` (PRIVATE).  
Only the scalar output is forwarded to Sonnet, with opaque keys:

```json
// PRIVATE — stays local
{
  "lambda_ode_from_cadence_and_vertical_oscillation": 1.3425e-04,
  "lambda_vel_from_cadence_and_step_length": 2.8691,
  "lambda_phase_from_cadence_stance_frac": 78.472
}

// DERIVED-OK — forwarded to cloud with opaque keys
{
  "w0": 1.3425e-04,
  "w1": 2.8691,
  "w2": 78.472
}
```

Key mapping `w0→λ_ode, w1→λ_vel, w2→λ_phase` stored only in private corpus.  
Sonnet receives magnitudes and relative balance — not what they physically represent.

### Justice as Abstraction Boundary

Every human decision gate in the judicial process is also a data abstraction boundary.  
The Justice receives:
- DERIVED-OK evidence (loss scalars, UART tables, plot summaries) from cloud/local
- Connects them to clinical meaning from first principles — in their head, not in any LLM context

No LLM ever holds both the private derivation and the clinical interpretation simultaneously.  
Article II is a privacy control by design.

---

## Agent-to-Tier Mapping

```
  LOCAL LLM (Qwen/DeepSeek)          CLOUD (Sonnet)
  ──────────────────────────          ──────────────────────────
  Retrieves: PRIVATE + DERIVED-OK     Retrieves: PUBLIC + DERIVED-OK
  + PUBLIC                            Never sees: PRIVATE

  plotter                             loss-setter (step 2 — scalars only)
  uart-reader                         physics-reviewer (step 2 — scalars only)
  train-sum                           pinn-executor
  plot-orchestrator                   pinn-compiler
  simulator-operator                  layer-setter
  synthetic-data-generator            attorneys (A + B)
  pinn-monitor                        judicial-clerk
  loss-setter (step 1 — local math)   pinn-grid-controller
  physics-reviewer (step 1 — local)   pinn-validator (check logic only)
  pinn-archivist (hashing only)       pinn-archivist (registry write)
```

### Borderline agents — split execution

**loss-setter:**  
Step 1 (local): retrieve `bill_loss_weights_v1.md` (PRIVATE), execute derivation, produce scalar dict  
Step 2 (cloud): receive opaque scalar dict + `amendments.md` (PUBLIC), reason about balance, propose Bill  

**physics-reviewer:**  
Step 1 (local): retrieve `physics_loss.py` (PRIVATE), compute λ·L per profile, produce balance table  
Step 2 (cloud): receive balance table (DERIVED-OK) + `amendments.md` (PUBLIC), assess constitutional compliance  

**pinn-archivist:**  
Step 1 (local): SHA-256 hash the `.pt` checkpoint (PRIVATE → hash is DERIVED-OK)  
Step 2 (cloud): write hash + scalar metrics to `pinn_registry.md` (DERIVED-OK)

---

## Open Questions

1. **`physics_review_v1.md` reclassification:** Currently PRIVATE because it contains formula context alongside λ values. A stripped version (scalars only, no formula text) would be DERIVED-OK. The physics-reviewer agent should produce two outputs: a private full report and a derived-ok summary for cloud consumption.

2. **Checkpoint weights (`.pt`):** Classified PRIVATE because weights encode private physics implicitly. If model inversion is not a realistic threat in this context, reclassify to DERIVED-OK. Needs a decision before grid search begins (pinn-grid-controller will need to load weights).

3. **`data_config.json` parameter distributions:** Currently PRIVATE. The terrain distribution (flat 60% / slope 20% / stairs 20%) is probably not sensitive — it's in `pinn_registry.md` already. Consider splitting into a public terrain-distribution summary and a private per-field distribution file.
