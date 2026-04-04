# GaitSense — Skill Contract Format Specification

**Branch:** hybrid-model  
**Date:** 2026-04-04  
**Scope:** Proof-of-concept on evidence session agents: plotter, uart-reader, train-sum, plot-orchestrator  
**Purpose:** Defines the data contract embedded in every agent and skill file that governs corpus access, model assignment, and forwarding rules in the hybrid local/cloud LLM architecture.

---

## Why Skill Contracts Exist

RAG retrieval without access control is a privacy gap. Agent model assignment without data contracts is a capability gap. The skill contract is the document that closes both simultaneously — it declares, per agent:

- Which corpus tier it may retrieve from
- Which model executes it
- What it receives as input (and from which tier)
- What it produces as output (and at which tier)
- What it may and may not forward upstream

Every agent `.md` file carries its contract in a `contract:` block appended to the frontmatter. Every skill command `.md` file carries a matching contract. Together they form the enforceable data boundary.

---

## Contract Block Format

```yaml
---
name: <agent-name>
description: "..."
tools: ...
model: <local-model | sonnet>
color: ...

contract:
  execution: local | cloud | split
  retrieves:
    - tier: PUBLIC | DERIVED-OK | PRIVATE
      sources: [list of file patterns or corpus labels]
  receives:
    - name: <input name>
      tier: PUBLIC | DERIVED-OK | PRIVATE
      format: <scalar | path | table | json-opaque | free-text>
  produces:
    - name: <output name>
      tier: PUBLIC | DERIVED-OK | PRIVATE
      format: <scalar | path | table | json-opaque | free-text>
      destination: <file path pattern | stdout | upstream-agent>
  may_forward:
    - tier: PUBLIC | DERIVED-OK
      to: <agent-name | cloud | any>
  must_not_forward:
    - tier: PRIVATE
      reason: <one-line reason>
  opaque_keys: true | false
---
```

### Field Definitions

| Field | Description |
|---|---|
| `execution` | `local` = runs on local LLM; `cloud` = runs on Sonnet; `split` = step 1 local, step 2 cloud |
| `retrieves` | Corpus tiers this agent is permitted to retrieve from. Local agents may retrieve PRIVATE. Cloud agents may not. |
| `receives` | Named inputs, their tier, and format. An agent must not accept PRIVATE input if `execution: cloud`. |
| `produces` | Named outputs, their tier, and where they go. |
| `may_forward` | What the agent is permitted to pass upstream. PRIVATE is never in this list. |
| `must_not_forward` | Explicit prohibition — redundant with `may_forward` absence, but written explicitly for auditability. |
| `opaque_keys` | If `true`, this agent applies opaque key substitution before forwarding any DERIVED-OK scalar dict. |

---

## Format Types

| Format | Description |
|---|---|
| `scalar` | A single numeric value |
| `scalar-dict` | `{"w0": 1.34e-4, "w1": 2.87, ...}` — opaque keys if derived from PRIVATE |
| `path` | File path to a saved artifact (plot, log) |
| `table` | Formatted text table printed to stdout |
| `json-summary` | Flat JSON with scalar values and status fields — no formula text |
| `free-text` | Human-readable prose — never forwarded upstream |
| `raw-signal` | NumPy array or binary — always PRIVATE |

---

## Proof-of-Concept Contracts — Evidence Session Agents

### plotter

```
execution:    local
retrieves:    PRIVATE (walker_model.py, signal arrays)
              PUBLIC  (amendments.md for threshold annotations)
receives:     profile name (PUBLIC), mode (PUBLIC)
produces:     plot PNG (DERIVED-OK → file)
              data table (DERIVED-OK → stdout)
may_forward:  plot path (PUBLIC), peak scalar values (DERIVED-OK)
must_not_forward: raw signal arrays (PRIVATE)
opaque_keys:  false  ← peak values are not formula-derived
```

### uart-reader

```
execution:    local
retrieves:    PRIVATE (UART log files, raw serial output)
receives:     log file path (DERIVED-OK) or serial port (PUBLIC)
produces:     structured STEP/SNAPSHOT/SESSION_END table (DERIVED-OK → stdout)
              session summary JSON (DERIVED-OK → file)
may_forward:  summary scalars: steps, SI%, cadence (DERIVED-OK)
must_not_forward: raw UART binary (PRIVATE)
opaque_keys:  false  ← SI% and cadence are not formula-derived
```

### train-sum

```
execution:    local
retrieves:    DERIVED-OK (training_logs/*.jsonl — per-epoch loss scalars)
              PUBLIC     (train_config.json for warmup reference lines)
receives:     run_id (PUBLIC)
produces:     loss curve PNG (DERIVED-OK → file)
              summary JSON (DERIVED-OK → file)
              metrics table (DERIVED-OK → stdout)
may_forward:  summary JSON (DERIVED-OK), plot path (PUBLIC)
must_not_forward: nothing PRIVATE (this agent never touches PRIVATE)
opaque_keys:  false  ← loss scalars are already DERIVED-OK
```

### plot-orchestrator

```
execution:    local
retrieves:    PUBLIC (agent definitions, amendments for Amendment 20 assessment)
              DERIVED-OK (receives outputs from sub-agents)
receives:     evidence type (PUBLIC), profile/run_id (PUBLIC/DERIVED-OK)
produces:     consolidated evidence block (DERIVED-OK → stdout)
              Amendment 20 assessment (DERIVED-OK → stdout)
may_forward:  consolidated block scalars (DERIVED-OK) to Justice or calling agent
must_not_forward: sub-agent raw outputs if they contain PRIVATE content
opaque_keys:  false  ← orchestrator passes DERIVED-OK scalars as-is
```

---

## The Forwarding Chain in the Evidence Session

```
  PRIVATE source files
  (walker_model.py, UART logs, training data)
         │
         ▼  local LLM retrieves
  ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
  │    plotter      │    │ uart-reader  │    │  train-sum   │
  │  PRIVATE in     │    │  PRIVATE in  │    │ DERIVED-OK in│
  │  DERIVED-OK out │    │ DERIVED-OK   │    │ DERIVED-OK   │
  │  (plot PNG,     │    │  out (table, │    │  out (loss   │
  │   data table)   │    │   summary)   │    │  curve, JSON)│
  └────────┬────────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                    │
           └──────────┬────────┘                    │
                      ▼                             │
           ┌──────────────────────┐                 │
           │   plot-orchestrator  │◄────────────────┘
           │   DERIVED-OK in      │
           │   DERIVED-OK out     │
           │   (consolidated      │
           │    evidence block)   │
           └──────────┬───────────┘
                      │
                      ▼
              JUSTICE (human)
              ← receives DERIVED-OK
              ← connects to clinical meaning
                from first principles
                (Article II privacy boundary)
                      │
                      ▼
           Cloud agents (if needed)
           receive DERIVED-OK only
           never see PRIVATE
```

---

## What Comes Next (beyond this PoC)

Once the four evidence session contracts are proven, the same format extends to:

1. **loss-setter** — `split` execution: step 1 local (PRIVATE retrieval, scalar derivation), step 2 cloud (DERIVED-OK opaque scalar dict)
2. **physics-reviewer** — `split` execution: step 1 local (PRIVATE source, full report), step 2 cloud (DERIVED-OK summary JSON)
3. **pinn-executor** — `cloud` execution: receives DERIVED-OK only from plot-orchestrator and pinn-monitor
4. **RAG retrieval layer** — reads `contract.retrieves` tier list to filter corpus before returning chunks to any agent

The contract block is the machine-readable spec that the retrieval layer enforces. The current implementation (manual discipline) becomes automated enforcement when the RAG layer reads and acts on the `retrieves` field.
