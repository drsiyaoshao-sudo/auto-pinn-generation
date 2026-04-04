# GaitSense — Agent Hierarchy, Skill Contracts, and RAG Structure

**Branch:** constitution-style-management
**Date:** 2026-04-04
**Scope:** Complete snapshot of the agent workflow as it stands on this branch.

---

## 1. Agent Hierarchy

### Constitutional Branch Assignments

```
JUDICIARY
├── judicial-clerk   [haiku]   — warms courtroom, launches hearing agents
├── Attorney-A       [sonnet]  — argues assigned position with physical evidence
└── Attorney-B       [sonnet]  — argues opposing position with physical evidence

LEGISLATURE
├── layer-setter          [sonnet]  — defines + locks pinn_model.py architecture
├── loss-setter           [sonnet]  — derives physics loss terms + λ weights → Bill
├── pinn-compiler         [sonnet]  — locks train_config.json hyperparameters → Bill
├── synthetic-data-setter [sonnet]  — defines dataset scope + bounds → Bill
└── pinn-grid-controller  [sonnet]  — proposes grid search domains → Bill

BUREAUCRACY (Execution — no Bill required)
├── Evidence Layer
│   ├── plot-orchestrator    [sonnet]  — coordinates plotter/uart-reader/train-sum
│   ├── plotter              [haiku]   — IMU signal diagnostic plots
│   ├── uart-reader          [haiku]   — UART log capture and print
│   └── train-sum            [haiku]   — loss curve plots + training summary table
│
├── PINN Pipeline
│   ├── simulator-operator   [sonnet]  — orchestrates full simulation run
│   ├── synthetic-data-generator [sonnet] — generates synthetic training dataset
│   ├── pinn-executor        [sonnet]  — runs training loop
│   ├── pinn-monitor         [haiku]   — per-epoch logging + early stop callbacks
│   ├── pinn-validator       [sonnet]  — Amendment 11 + VABS.F32 + Amendment 19 checks
│   ├── pinn-archivist       [haiku]   — SHA-256 hashes checkpoint + writes registry
│   └── physics-reviewer     [sonnet]  — derivation trace + λ balance table (split)
│
└── Housekeeping
    ├── stage-compactor      [sonnet]  — freezes + compacts stage case law on gate confirm
    └── package-manager      [haiku]   — installs/pins dependencies
```

### Call Chains

```
Human /plot-profile  →  plotter
Human /plot-training →  train-sum (via plotter agent per skill file)
Human /plot-evidence →  plot-orchestrator
                              ├── signal     → plotter
                              ├── uart       → uart-reader
                              ├── training   → train-sum
                              ├── all-sim    → uart-reader → plotter
                              └── all-pinn   → train-sum

simulator-operator   →  plot-orchestrator (all-simulation)
                              └── uart-reader → plotter

pinn-executor        →  pinn-monitor (training start)
pinn-executor        →  plot-orchestrator (all-pinn, post-training)
                              └── train-sum

Human confirms       →  pinn-archivist  →  registry write
Human confirms       →  pinn-validator  →  Amendment 11 + fidelity check

Judicial Hearing:
Justice declares     →  judicial-clerk
                              ├── Attorney-A
                              ├── Attorney-B
                              ├── simulator-operator
                              ├── plotter
                              └── uart-reader
```

---

## 2. Skill Contracts

### Format (frontmatter contract: block)

```yaml
contract:
  execution: local | cloud | split
  retrieves:
    - tier: PUBLIC | DERIVED-OK | PRIVATE
      sources: [file patterns]
  receives:
    - name: <input>
      tier: <tier>
      format: scalar | path | table | json-opaque | free-text | raw-signal
  produces:
    - name: <output>
      tier: <tier>
      format: <format>
      destination: <path | stdout | upstream-agent>
  may_forward:
    - tier: PUBLIC | DERIVED-OK
      to: <agent | cloud | any>
  must_not_forward:
    - tier: PRIVATE
      reason: <reason>
  opaque_keys: true | false
```

### Contracts — Evidence Session Agents (PoC — contracts written in agent files)

| Agent | Execution | Retrieves | Input tier | Output tier | Opaque keys |
|---|---|---|---|---|---|
| **plotter** | local | PRIVATE + PUBLIC | PUBLIC (profile name) | DERIVED-OK (plot PNG, data table) | false |
| **uart-reader** | local | PRIVATE (UART logs) | DERIVED-OK or PUBLIC | DERIVED-OK (STEP table, summary JSON) | false |
| **train-sum** | local | DERIVED-OK (epoch logs) + PUBLIC (train_config) | PUBLIC (run_id) | DERIVED-OK (loss PNG, summary JSON, table) | false |
| **plot-orchestrator** | local | PUBLIC (agent defs, amendments) | PUBLIC + DERIVED-OK | DERIVED-OK (consolidated evidence block) | false |

### Contracts — Split Agents (defined in corpus_classification.md, not yet in frontmatter)

| Agent | Step 1 (local) | Step 2 (cloud) | Opaque keys |
|---|---|---|---|
| **loss-setter** | Retrieve `bill_loss_weights_v1.md` (PRIVATE), derive λ scalars | Receive opaque `{w0, w1, w2}` + `amendments.md`, reason about balance, write Bill | **true** — `λ_ode→w0`, `λ_vel→w1`, `λ_phase→w2` |
| **physics-reviewer** | Retrieve `physics_loss.py` (PRIVATE), compute λ·L per profile, full trace → Output A (PRIVATE md) | Receive opaque scalar summary (DERIVED-OK JSON) + amendments, assess compliance | **true** |
| **pinn-archivist** | SHA-256 hash `.pt` checkpoint (PRIVATE → hash is DERIVED-OK) | Write hash + scalar metrics to `pinn_registry.md` | false |

### Forwarding Chain

```
PRIVATE sources                       (never leave local)
(walker_model.py, physics_loss.py,
 UART logs, .pt checkpoints,
 training_data/*.npy)
        │
        ▼  local LLM retrieves only
 ┌─────────────────────────────────────────────────────┐
 │  plotter       uart-reader    train-sum              │
 │  PRIVATE→DERIVED   PRIVATE→DERIVED   DERIVED→DERIVED│
 │  (plot PNG,        (STEP table,       (loss PNG,     │
 │   data table)       summary JSON)     summary JSON)  │
 └──────────────────┬──────────────────────────────────┘
                    ▼
        ┌───────────────────────┐
        │   plot-orchestrator   │   (local)
        │   DERIVED-OK in/out   │
        │   consolidated block  │
        └───────────┬───────────┘
                    │
                    ▼
            JUSTICE (human)              ← Article II boundary
            reads DERIVED-OK evidence
            connects to clinical meaning in-head
                    │
                    ▼
        Cloud agents (sonnet) if needed
        receive DERIVED-OK only
        never see PRIVATE
```

---

## 3. RAG Structure

### Three Tiers

```
  PRIVATE  ── stays local, never forwarded
             contains: derivation formulas, ODE equations, implementation code,
             population parameter distributions, model weights, raw signals

  DERIVED-OK ── safe for cloud after opaque-key masking
               contains: loss scalars, epoch logs, plot paths, SI%, cadence,
               SHA-256 hashes, summary statistics
               rule: variable names revealing formula structure must be replaced
               before forwarding (w0/w1/w2 instead of λ_ode/λ_vel/λ_phase)

  PUBLIC   ── retrievable by any LLM, no transformation
             contains: CLAUDE.md, amendments, case law, agent definitions,
             architectural metadata, bills (hyperparameter values only),
             BOM docs, procedure records
```

### File-to-Tier Map (key entries)

```
PRIVATE
  simulator/walker_model.py               biomechanical derivation chain — core IP
  simulator/pinn/physics_loss.py          ODE + velocity + phase constraint impls
  simulator/pinn/pinn_model.py            Fourier feature encoding — architecture IP
  simulator/pinn/train_pinn.py            physics ramp schedule impl
  simulator/pinn/generate_training_data.py population parameter assumptions
  simulator/pinn/checkpoints/*.pt         model weights (encode private physics)
  simulator/pinn/data_config_private.json per-field μ, σ distributions
  simulator/pinn/data_config.json         legacy combined — treat as PRIVATE
  simulator/pinn/training_data/*.npy      raw signal sequences
  simulator/pinn/training_data/anchor_profiles.json exact WalkerProfile values
  docs/gaitsense_code/bills/bill_loss_weights_v1.md    full ODE derivation
  docs/gaitsense_code/bills/bill_physics_loss_v2.md    double-integration proof
  docs/gaitsense_code/bills/bill_data_config_v1.md     population distributions
  docs/executive_branch_document/plots/.../physics_review_vN.md  full trace

DERIVED-OK
  simulator/pinn/training_logs/*.jsonl    per-epoch loss scalars
  simulator/pinn/physics_review_summary.json  opaque {w0,w1,w2} scalars
  simulator/pinn/physics_review_log.json  human_decision + bill_reviewed only
  simulator/pinn/data_config_public.json  counts, splits, terrain dist, seed
  simulator/pinn/train_config.json        λ values as scalars (opaque key rule)
  simulator/pinn/checkpoints/manifest.json  SHA-256 hashes + scalar metrics
  docs/gaitsense_code/pinn_registry.md   hashes + metrics
  docs/executive_branch_document/plots/pinn_training/*.png  visual summaries
  docs/executive_branch_document/plots/*.png                visual only

PUBLIC
  CLAUDE.md
  docs/gaitsense_code/amendments.md
  docs/gaitsense_code/case_law.md
  docs/gaitsense_code/corpus_classification.md
  docs/gaitsense_code/skill_contract_spec.md
  docs/gaitsense_code/evidence_session.md
  docs/gaitsense_code/hearing_procedure_abstract.md
  docs/gaitsense_code/judicial-process.md
  docs/gaitsense_code/bills/bill_train_config_v*.md   hyperparameter values only
  simulator/pinn/architecture.json                    topology metadata only
  simulator/pinn/data_config_public.json
  .claude/agents/*.md                                 role definitions
  .claude/commands/*.md                               skill interface contracts
  docs/executive_branch_document/handoff.md
  docs/executive_branch_document/hw_bom.md
  docs/executive_branch_document/sw_bom.md
```

### Agent-to-Corpus Access

```
  LOCAL LLM only                          CLOUD (Sonnet) — DERIVED-OK + PUBLIC
  ──────────────────────────────────      ──────────────────────────────────────
  plotter              (PRIVATE in)       pinn-executor
  uart-reader          (PRIVATE in)       pinn-compiler
  train-sum            (DERIVED-OK in)    layer-setter
  plot-orchestrator    (PUBLIC + DER-OK)  synthetic-data-setter
  simulator-operator   (PRIVATE access)  pinn-grid-controller
  synthetic-data-generator               pinn-validator (check logic only)
  pinn-monitor                           attorneys A + B
  loss-setter step 1   (PRIVATE derive)  judicial-clerk
  physics-reviewer step1 (PRIVATE trace) pinn-archivist step 2 (registry write)
  pinn-archivist step1 (SHA-256 local)   loss-setter step 2 (opaque scalars in)
                                         physics-reviewer step 2 (opaque in)
```

### Opaque Key Masking (loss-setter, physics-reviewer)

```
  PRIVATE side (local, never forwarded):
    { "lambda_ode_from_cadence_and_vertical_oscillation": 1.3425e-04,
      "lambda_vel_from_cadence_and_step_length":          2.8691,
      "lambda_phase_from_cadence_stance_frac":            78.472 }

  DERIVED-OK side (forwarded to cloud, opaque keys):
    { "w0": 1.3425e-04,
      "w1": 2.8691,
      "w2": 78.472 }

  Key mapping (w0→λ_ode, w1→λ_vel, w2→λ_phase) stored only in private corpus.
  Cloud receives magnitudes and relative balance — not what they physically mean.
```

---

## Summary: The Three Structures Together

```
  USER / JUSTICE
       │
       │  /plot-profile  /plot-training  /plot-evidence  (Skills)
       ▼
  SKILL COMMANDS (.claude/commands/)
  — skill contract: input tier, output tier, which agent to dispatch
       │
       ▼
  AGENTS (.claude/agents/)
  — agent contract: execution tier, retrieves tier, produces tier
  — hierarchy: Judiciary / Legislature / Bureaucracy
       │
       ├── LOCAL agents → PRIVATE corpus  (walker_model, physics_loss, UART, .pt)
       │                  DERIVED-OK out  (scalars, plots, tables)
       │
       └── CLOUD agents → PUBLIC corpus   (CLAUDE.md, amendments, agent defs)
                          DERIVED-OK in   (scalars only, opaque keys)
                          DERIVED-OK out  (Bills, rulings, summaries)

  RAG CORPUS
  PRIVATE ──────────────────────────────► LOCAL agents only
  DERIVED-OK ────────────────────────────► LOCAL + CLOUD (opaque keys if formula-derived)
  PUBLIC ─────────────────────────────────► any agent
```
