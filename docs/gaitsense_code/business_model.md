# GaitSense — Business Model and Strategic Thesis

**Date:** 2026-04-04
**Status:** Internal — investor thesis + roadmap
**Audience:** YC / a16z / Antler + ongoing customer discovery

---

## The One-Line Thesis

**We are building governed co-design infrastructure — the layer above EDA and PLM tools where physics, process, IP, and compliance are the same artifact, enforced by hybrid intelligence.**

---

## The Problem

Hardware design has three systems with no shared epistemological model:

```
Physics simulation     →  tool A (MATLAB, COMSOL, custom scripts)
Design process         →  tool B (Jira, Confluence, email)
IP + compliance record →  tool C (PDF, Word, SharePoint)
```

Manual handoffs between all three. A design decision made in simulation never reliably reaches the compliance record. An IP claim is only as strong as the documentation discipline of the engineer who made it. A regulatory submission is a retroactive reconstruction — not a live trace.

**The result:** A hardware team of 10 produces what an AI-augmented team of 1 should be able to produce — but the 1-person team has no paper trail, no cross-team synchronization, and no auditable evidence chain.

Incumbent tools (Cadence, Synopsys, Altium, Windchill) track versions. They do not track *why* decisions were made or *what physical evidence* justified them. They have no epistemological model.

---

## The Solution: Constitutional Governance as Infrastructure

The GaitSense repo already demonstrates the full system in production on a medical wearable:

### 1. Constitutional Governance Layer
Every design decision traces to a physically measurable quantity (Article I) and has a human sign-off backed by empirical evidence (Article II). Not documentation discipline — enforced by the system.

```
Bill proposed  →  physical evidence required to debate
Bill enacted   →  implemented on dedicated branch
Stage gate     →  exit criteria validated, case law frozen
Amendment      →  supermajority ratification, immutable once ratified
```

### 2. Agents as Evidence Probes
21 agents across the constitutional branches — none of them decide anything. They produce measurements. Humans decide.

```
plotter       →  generates signal plots, prints data table, stops
uart-reader   →  captures UART, prints structured table, stops
physics-reviewer → computes λ balance, prints derivation trace, stops
```

**The guarantee:** every human decision in this system is backed by empirical evidence. No AI self-selects algorithmic direction. This is not a limitation — it is the product promise to regulated industries.

### 3. Hybrid Intelligence as Trust Boundary
The local/cloud split is architecturally necessary — not a cost optimization:

```
LOCAL LLM (on bench)               CLOUD LLM (Sonnet)
──────────────────────             ──────────────────────
The physics                        The governance
The derivations                    Constitutional compliance
The signal data                    Balance checks
The IP (private corpus)            Process validation
```

The human Justice is the only node holding both sides simultaneously — in their head, not in any model context. This satisfies FDA 21 CFR Part 11, patent defensibility, ITAR/EAR export control, and cross-org collaboration simultaneously.

### 4. Auto PINN Generation — The Physics Primitives Engine
Physics-informed neural networks derived entirely from three first-order gait primitives:

```
1. Vertical Oscillation (cm)
2. Cadence (steps/min)
3. Step_Length (m)
```

Every loss term traces to these. No parameter is a guess. The PINN learns physics — it does not fit noise. This is the technical foundation that makes distillation viable: you know exactly what the model learned and why.

### 5. The Repo IS the Product
```
.claude/agents/           →  agent deployment units (skill marketplace)
.claude/commands/         →  user-facing skill API
docs/gaitsense_code/      →  governance layer (IP + process record)
simulator/pinn/           →  model training pipeline
simulator/pinn/checkpoints/manifest.json  →  model provenance chain
```

A customer forks the repo. The constitutional governance, the agent hierarchy, the RAG tier structure, and the CI/CD pipeline come pre-wired. The repo is the onboarding.

---

## What We Are Building (Analogies)

| Analogy | What they did | What we do differently |
|---|---|---|
| **Hugging Face** | Model hub + deployment | We add governance provenance — every model is traceable to evidence-backed design decisions, not just a weight file |
| **GitHub for hardware** | Version control | We add epistemological layer — *why* decisions were made, not just *what* changed |
| **Figma for cross-team** | Shared design surface | We add exit criteria enforcement — teams synchronize through constitutional process, not just file sharing |
| **Notion + Linear** | Docs + issue tracking | We collapse docs, decisions, and compliance into one artifact |

---

## Revenue Model — Three Tiers

### Tier 1 — Platform License
**What:** Constitutional governance repo + agent hierarchy, customer brings their own LLM API keys
**Who:** Hardware engineering teams at startups and mid-size OEMs
**Revenue:** Per-seat license per engineering team
**Moat:** Network effects from shared case law structure; switching cost grows as case law accumulates

### Tier 2 — Managed Distillation
**What:** We run the distillation pipeline on the customer's private corpus. Deliver a domain model that runs fully local — on bench laptop, RPi5, or edge MCU.
**Who:** Teams that cannot send derivation formulas to cloud (medical, defense, aerospace)
**Revenue:** Distillation run fee + on-prem model delivery
**Key:** IP never leaves customer infrastructure. We never see the private corpus — we run the pipeline on their hardware.

**The distillation flywheel:**
```
Stage gate closes → stage-compactor freezes case law (IP provenance)
Private corpus grows (validated physics + evidence-backed decisions)
Distillation: Sonnet teacher → small local student
Student knows domain physics, runs fully offline
Next domain: orthopedics, power electronics, aerospace structures
Same pipeline, different corpus
```

### Tier 3 — Certified Model + Compliance Package
**What:** Distilled model + full provenance chain (Bills, case law, manifest.json, amendment record) formatted for regulatory submission
**Who:** Medical device manufacturers (FDA), defense contractors (ITAR), automotive (ISO 26262)
**Revenue:** Per-submission certification fee
**Moat:** No one else has the evidence chain. Competitors can produce a model — they cannot produce a model whose every training decision is traceable to a dated Bill, a judicial hearing, and a human sign-off.

---

## The Flywheel

```
More customers
      ↓
More domain corpora (each customer's private corpus grows with their design history)
      ↓
Better distillation pipeline (more domains trained, pipeline matures)
      ↓
Stronger governance guarantees (more case law = stronger precedent)
      ↓
Regulated industries pay for Tier 3 certification
      ↓
New customers in harder domains (defense, nuclear, implantables)
      ↓
Back to top
```

---

## Moat — What Cannot Be Replicated

### Technical moat
1. **Physics primitives enforcement** — Article I mandates every parameter traces to vertical oscillation, cadence, or step length. This is not a design choice — it is a constitutional constraint. It makes the PINN auditable in a way no black-box model can be.
2. **Auto PINN generation pipeline** — Steps 1–7 fully agentic, from layer architecture to checkpoint registration, with Amendment 20 physics warmup constraint. A competitor building this from scratch needs 6–12 months minimum.
3. **RAG tier structure + opaque key masking** — The PRIVATE/DERIVED-OK/PUBLIC split with w0/w1/w2 substitution means cloud LLMs reason about balance without seeing formulas. This is a novel architecture — not published, not in any existing AI framework.

### Process moat
4. **Case law as IP provenance** — Every enacted Bill, every judicial ruling, is immutable case law. This is a dated, evidence-backed IP chain that satisfies both patent priority and regulatory audit simultaneously. Incumbents have version control. We have epistemology.
5. **Constitutional governance structure** — The four-branch system (Legislature, Judiciary, Bureaucracy, Amendment ratification) maps directly to org structure. As a customer's engineering team grows, the governance structure scales with them without process redesign.

### Distribution moat
6. **The repo IS the product** — No sales motion required for technical founders. Fork, configure, run. The governance layer is self-explanatory because it is constitutional — every rule cites its physical justification.
7. **Hybrid intelligence as entry barrier** — The local/cloud split requires a bench-side infrastructure that pure cloud AI companies cannot replicate. We are the only vendor who can sit next to the hardware.

---

## Delta vs. Incumbents and Competing Startups

### vs. EDA tools (Cadence, Synopsys, Altium)
- They track versions. We track decisions.
- They have no AI layer. We have a governed AI layer where every agent output is evidence, not direction.
- They require expensive licenses and specialist training. We require a terminal and a compiler.

### vs. PLM tools (Windchill, Teamcenter, Arena)
- They manage BOMs and change orders. We generate them as constitutional artifacts.
- They separate documentation from design. We make them the same thing.
- They have no physics layer. We enforce physics primitives at the constitutional level.

### vs. AI coding assistants (Copilot, Cursor, Devin)
- They write code. They do not govern decisions.
- They have no epistemological model — no way to know if a suggestion is physically grounded.
- They cannot produce a regulatory submission. We can.

### vs. AI for hardware startups (Flux, Celus, etc.)
- They focus on schematic generation and PCB layout. We focus on the decision layer above hardware.
- They are cloud-only. We are hybrid — local physics, cloud governance.
- They have no physics primitives enforcement. Every parameter they generate is a guess unless the engineer validates it manually.

### vs. MLOps platforms (Weights & Biases, MLflow, Hugging Face)
- They track model experiments. We track the physical justification for every training decision.
- They have no governance layer. We have a constitutional system with judicial review.
- They cannot produce a regulatory-grade provenance chain. We can.

---

## What Needs to Be Built

### Near-term (YC sprint — closes ~2026-04-24)
- [ ] Close 2 remaining PINN ratification criteria (Amendment 19 fidelity + VABS.F32 pathological check)
- [ ] First end-to-end distillation run: Sonnet teacher → small student on validated PINN corpus
- [ ] BOM diff in CI/CD: enacted Bill triggers hw_bom.md / sw_bom.md validation
- [ ] Customer discovery: 3 interviews with medical device firmware teams on governance pain points

### Platform layer (6 months)
- [ ] RAG retrieval layer that reads `contract.retrieves` tier field and enforces access automatically (currently manual discipline)
- [ ] Virtual bench abstraction: oscilloscope SCPI, PPK2, J-Link RTT as first-class tool calls in the agent hierarchy
- [ ] Cross-team corpus sync: PUBLIC tier corpus shareable across orgs, PRIVATE tier stays behind customer firewall
- [ ] Stage-compactor as IP filing assistant: output formatted for provisional patent filing

### Scale (12 months)
- [ ] Domain expansion: orthopedic wearables, power electronics, aerospace structures
- [ ] Distillation-as-a-service pipeline: automated, runs on customer infrastructure, we never see private corpus
- [ ] Tier 3 certification package: FDA 21 CFR Part 11 submission template driven from case law + manifest.json
- [ ] Agent marketplace: community-contributed agents with constitutional compliance checks before merge

---

## The Pitch in Three Sentences

Hardware design is broken because physics, process, and IP record are three separate systems with no shared model of truth. We built the governance layer that collapses all three — enforced by hybrid intelligence where local models protect the IP and cloud models validate the process. The result is a co-design platform where agents act as evidence probes, humans remain the decision authority, and every design choice is automatically an IP record and a compliance artifact.
