# Hybrid-Model Rebase Design Note

**Purpose:** Documents the design decisions that hybrid-model must resolve when it rebases
onto constitution-style-management. The session orchestrator (/session) and skill
architecture on this branch are the target state. Hybrid-model adds the context mask
and local/cloud routing layer on top.

---

## What This Branch Provides (constitution-style-management)

```
  Skills (all run Sonnet/Haiku — no local/cloud differentiation yet):
    /session [stage]       — full 5-stage pipeline orchestrator
    /model-train [phase]   — Stage 2: data → design → compile → train → archive → validate
    /hear "<name>" A vs B  — judicial hearing (7-step procedure)
    /plot-evidence <type>  — evidence collection

  Agents (22 total):
    Legislature:  synthetic-data-setter, loss-setter, pinn-compiler,
                  pinn-grid-controller, fw-generator [new]
    Judiciary:    judicial-clerk, attorney-A, attorney-B
    Executive:    synthetic-data-generator, layer-setter, physics-reviewer,
                  pinn-executor, pinn-monitor, train-sum, pinn-archivist,
                  pinn-validator, simulator-operator, plotter, uart-reader,
                  plot-orchestrator, stage-compactor
    Unclassified: package-manager
```

---

## What Hybrid-Model Must Add

### 1. Skill frontmatter: corpus_tier + model_target + forward_policy

Each agent definition needs three new frontmatter fields:

```yaml
corpus_tier: public | private | derived-ok
model_target: cloud | local
forward_policy: what this agent may pass upstream (scalar_only | summary | full)
```

Example (loss-setter — the hard case):
```yaml
corpus_tier: private        # reads walker_model params and calibration constants
model_target: local         # local LLM executes the derivation equation
forward_policy: scalar_only # passes λ values to pinn-compiler, never equation text
```

Example (attorney-A):
```yaml
corpus_tier: public         # reads amendments, case_law, CLAUDE.md only
model_target: cloud         # complex constitutional reasoning → Sonnet
forward_policy: full        # arguments are public constitutional text
```

### 2. Corpus tier classification

Every document in the repo that an agent may retrieve must be tagged.
Classification schema (from memory/project_local_llm_rag.md):

| Tier | What | Cloud retrieval |
|------|------|-----------------|
| `public` | amendments.md, case_law.md, CLAUDE.md, lesson docs, bills, physics derivations | OK |
| `private` | training data, patient profiles, walker_model params, SI measurements, calibration constants | Local only |
| `derived-ok` | per-epoch loss scalars, summary statistics, evidence tables derived from private data | Summaries only |

### 3. Context mask per agent

Each agent sees only its corpus tier. No agent sees above its clearance.

```
  Local agents (private tier):
    synthetic-data-generator → sees training data, walker_model params
    pinn-monitor, train-sum, pinn-archivist → sees loss scalars (derived-ok)
    simulator-operator, plotter, uart-reader → sees UART logs, signal arrays (derived-ok)

  Cloud agents (public tier only):
    attorneys, judicial-clerk → see amendments, case_law, bills, CLAUDE.md
    loss-setter → local derives λ scalars, passes scalars only to cloud pinn-compiler
    pinn-validator → sees summary statistics, not raw training arrays

  Human Justice (all tiers):
    The Justice is the only entity that holds both the private signal data
    and the constitutional interpretation simultaneously. Article II is
    accidentally a privacy control — every Justice gate abstracts private
    data to derived-ok before it reaches cloud reasoning.
```

### 4. Model assignments

```
  [CLOUD — Sonnet]:
    Judiciary: judicial-clerk, attorney-A, attorney-B
    Legislature: synthetic-data-setter, pinn-compiler, pinn-grid-controller, fw-generator
    Executive: layer-setter, physics-reviewer, pinn-executor, pinn-validator

  [LOCAL — small model ~7B]:
    Legislature: loss-setter (derivation step — private data)
    Executive: synthetic-data-generator, pinn-monitor, train-sum, pinn-archivist,
               simulator-operator, plotter, uart-reader, plot-orchestrator

  Note: loss-setter is split. The derivation (reads private params, computes λ)
  runs local. The balance/convergence reasoning (why this λ, is it correct?) runs
  cloud on scalar outputs only.
```

---

## The Open Questions (from project_local_llm_rag.md)

These must be resolved before hybrid-model can implement the split:

1. **Skill frontmatter format** — `corpus_tier`, `model_target`, `forward_policy` fields
   need to be standardised. What exactly can be in `forward_policy: scalar_only`?
   Define the format precisely so agents cannot accidentally inline equation text.

2. **RAG chunk classification** — every document needs explicit sensitivity tags
   before the retrieval layer can enforce the boundary. The bills from this session
   (bill_physics_loss_v4/v5/v6, bill_architecture_v21a) are `public`.
   data_config.json and training arrays are `private`. Classify before building RAG.

3. **loss-setter masking format** — what exactly gets forwarded from local to cloud?
   Scalar dict only: `{"lambda_ode": 0.0, "lambda_vel": 0.0, "lambda_phase": 0.1}`.
   No equation text. No variable names that reveal formula structure.
   Define the JSON schema and enforce it in the forward_policy.

---

## Rebase Sequence

When hybrid-model rebases onto constitution-style-management:

1. Resolve any conflicts in `.claude/agents/` and `.claude/commands/`
   — this branch is the canonical agent definitions
2. Add `corpus_tier`, `model_target`, `forward_policy` to each agent's frontmatter
3. Classify all documents in `docs/` with sensitivity tags
4. Implement RAG retrieval layer with tier enforcement
5. Route each agent to its target model (local vs cloud) based on frontmatter
6. Test loss-setter masking: verify λ scalars pass, equation text does not
7. Verify human Justice gates still act as data abstraction boundaries
