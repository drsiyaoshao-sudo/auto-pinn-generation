Run a full GaitSense development session from data generation through hardware bring-up.

Usage: /session [stage]

Stages:
  status    — read constitutional record, print current stage and open gates
  1         — Stage 1: Firmware simulation (Renode)
  2         — Stage 2: PINN training pipeline  [calls /model-train]
  3         — Stage 3: Grid search (PINN boundary discovery)
  4         — Stage 4: Edge cases (fw-generator → firmware Bills)
  5         — Stage 5: Hardware bring-up (physical flash + clinical validation)

If no stage given, run /session status first, then ask Justice which stage to begin.

---

## Session Initialisation (always runs first)

**Step 0 — Package check (runs before anything else):**
Invoke `package-manager` to verify all required Python packages are installed and at the correct versions.
package-manager checks: `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`, and any packages listed in `requirements.txt`.
If any package is missing or at the wrong version, package-manager installs/pins it and reports the result.
Session init does not proceed to the constitutional record print until package-manager reports clean.

Read and print a session header from the constitutional record:

```
══════════════════════════════════════════════════════════════════
  GAITSENSE SESSION — $(date)
══════════════════════════════════════════════════════════════════
  Constitutional record:
    Amendments: [N] ratified  — most recent: [title]
    Case law:   [N] precedents recorded
    Bills:      [list of bills/ filenames]

  Stage status (from docs/executive_branch_document/ stage criteria):
    Stage 1 — Firmware Simulation:   [CLOSED / OPEN / NOT STARTED]
    Stage 2 — PINN Training:         [CLOSED / OPEN / NOT STARTED]
    Stage 3 — Grid Search:           [CLOSED / OPEN / NOT STARTED]
    Stage 4 — Edge Cases:            [CLOSED / OPEN / NOT STARTED]
    Stage 5 — Hardware Bring-up:     [CLOSED / OPEN / NOT STARTED]

  Agent roster:
    Legislature:  synthetic-data-setter  loss-setter  pinn-compiler
                  pinn-grid-controller   fw-generator
    Judiciary:    judicial-clerk  attorney-A  attorney-B
    Executive:    synthetic-data-generator  layer-setter  physics-reviewer
                  pinn-executor  pinn-monitor  train-sum  pinn-archivist
                  pinn-validator  simulator-operator  plotter  uart-reader
                  plot-orchestrator  stage-compactor
    Bureaucracy:  package-manager

  Skills:
    /session [stage]       — this orchestrator
    /model-train [phase]   — Stage 2 full pipeline
    /hear "<name>" A vs B  — judicial hearing
    /plot-evidence <type>  — evidence collection
══════════════════════════════════════════════════════════════════
```

---

## Stage 1 — Firmware Simulation

**Agents:** simulator-operator → uart-reader → plotter

### Entry condition
Stage 1 is the starting point. ELF must exist at the path expected by simulator-operator.

### Pipeline
```
  For each profile in [flat, bad_wear, stairs, slope]:
    simulator-operator → uart-reader (UART log) → plotter (signal plot)
    → print: step count, SI_stance, cadence per profile

  Pathological check:
    simulator-operator (si_true=25%) → SI > 10% required
```

### Exit criteria
- All 4 healthy profiles: ≥ 98/100 steps, SI < 10%
- Pathological: SI > 10%
- All signal plots reviewed by Justice (Amendment 11)

**[JUSTICE GATE S1]** Criteria met → invoke `stage-compactor` to close Stage 1.

---

## Stage 2 — PINN Training Pipeline

**Skill:** `/model-train`
**Agents:** 11 agents across 6 phases — see `/model-train` for full detail.

### Entry condition
Stage 1 closed. ELF validated in Stage 1.

### Pipeline
Invoke `/model-train`. It handles Amendment 21 pre-flight, all 6 phases, and 6 Justice gates.

### Exit criteria
- pinn-validator: Amendment 11 plots + Amendment 19 fidelity ≤ 15% + VABS.F32 all pass
- Checkpoint archived by pinn-archivist (Amendment 16)
- Justice confirms pinn-validator output

**[JUSTICE GATE S2]** Criteria met → invoke `stage-compactor` to close Stage 2.

---

## Stage 3 — Grid Search (PINN Boundary Discovery)

**Agents:** pinn-grid-controller → [/hear per domain] → pinn-executor (batch) → simulator-operator

### Entry condition
Stage 2 closed. Validated PINN checkpoint in pinn-archivist manifest.

### Pipeline
```
  1. pinn-grid-controller proposes search domain Bills
     (each domain: axis name, physical justification, clinical hypothesis, Renode assertion)

  2. [/hear per domain] — Justice ratifies each domain separately

  3. pinn-executor: batch PINN inference across domain grid
     → boundary candidates where algorithm fails

  4. simulator-operator (Renode): confirms each candidate
     → UART evidence of algorithm failure at boundary condition

  5. Confirmed boundaries → case_law.md → handed to fw-generator (Stage 4)
```

### Exit criteria
- All proposed domains ratified by Justice
- Each confirmed boundary has Renode UART evidence
- Boundary candidates written to case_law.md

**[JUSTICE GATE S3]** Criteria met → invoke `stage-compactor` to close Stage 3.

---

## Stage 4 — Edge Cases (Firmware Bills from Boundary Findings)

**Agents:** fw-generator → [/hear per boundary] → pinn-executor (regression) → simulator-operator

### Entry condition
Stage 3 closed. Confirmed boundary candidates in case_law.md.

### Pipeline
```
  For each confirmed boundary candidate:

    1. fw-generator drafts firmware Bill (one Bill per boundary — no batching)

    2. [/hear] — Justice ratifies Bill (or declares hearing if amendment conflict)

    3. Firmware change implemented on branch

    4. pinn-executor: regression — all 4 anchor profiles must still pass

    5. simulator-operator: Renode confirmation of fixed boundary condition

    6. Bill enacted → case_law.md updated
```

### Exit criteria
- All boundary Bills enacted and Renode-confirmed
- Regression: all 4 healthy profiles ≥ 98/100 steps
- Pathological walker still passes VABS.F32 (SI > 10%)

**[JUSTICE GATE S4]** Criteria met → invoke `stage-compactor` to close Stage 4.

---

## Stage 5 — Hardware Bring-up

**Agents:** simulator-operator (final Renode validation), stage-compactor

### Entry condition
Stage 4 closed. Final ELF validated on all 4 profiles + all boundary conditions.

### Pipeline
```
  1. Final Renode validation run — all 4 profiles + all enacted boundary conditions
     → Justice reviews evidence (Amendment 11)

  2. [JUSTICE APPROVES FLASH — Article II irreversibility gate]
     The Justice approves based on simulation evidence.
     No agent self-selects this action. This is the constitutional stress test.

  3. Physical flash → ELF to hardware

  4. SI measurement on real hardware:
     Engineer walks 4 profiles → compare SI vs Renode prediction
     Tolerance: within 6.3% of Renode SI (Amendment 11 derivation)

  5. BLE export capture → binary snapshot decode → CSV export

  6. Clinical validation:
     Healthy: SI < 10%
     Pathological (si_true=25%): SI > 10%
     Hardware SI within tolerance of Renode SI

  7. Deviations → cross-reference with bug_receipt.md
     Unexpected deviation → [/hear] before any fix
```

### Exit criteria
- Hardware SI within 6.3% of Renode prediction on all profiles
- BLE export captured and decoded
- Clinical validation passed
- All deviations explained and recorded in case_law.md

**[JUSTICE GATE S5 — FINAL]** Criteria met → invoke `stage-compactor` to close Stage 5.
Constitutional method validated end-to-end.

---

## Full Pipeline Map

```
  /session
      │
      ├── [SESSION INIT] constitutional record · stage status · agent roster
      │
      ├── Stage 1 ── Firmware Simulation
      │               simulator-operator × 4 profiles + pathological
      │               uart-reader · plotter
      │               [GATE S1] → stage-compactor
      │
      ├── Stage 2 ── PINN Training          ← /model-train
      │               [Amendment 21 pre-flight]
      │               data · design · compile · train · archive · validate
      │               [GATE S2] → stage-compactor
      │
      ├── Stage 3 ── Grid Search
      │               pinn-grid-controller → [/hear × domains]
      │               pinn-executor (batch) · simulator-operator (Renode)
      │               [GATE S3] → stage-compactor
      │
      ├── Stage 4 ── Edge Cases
      │               fw-generator → [/hear × boundaries]
      │               pinn-executor (regression) · simulator-operator
      │               [GATE S4] → stage-compactor
      │
      └── Stage 5 ── Hardware Bring-up
                      [JUSTICE APPROVES FLASH — Article II]
                      physical flash · SI measurement · BLE export
                      [GATE S5 FINAL] → stage-compactor
```

---

## Missing Agents (gaps before full pipeline run)

| Agent | Stage | Status |
|-------|-------|--------|
| `fw-generator` | 4 | Added this session — stub complete |
| `hw-operator` | 5 | Not yet written — hardware flash, BLE, SI measurement interface |

---

## Constitutional References

- Article I: all signal values and algorithm parameters trace to physical primitives
- Article II: no irreversible action (flash) without Justice approval
- Amendment 1: five-stage gate structure
- Amendment 7: three-strike escalation to human
- Amendment 11: signal plots mandatory at each stage gate
- Amendment 13: dataset config and hyperparameters require Bills
- Amendment 16: checkpoint archival before validation
- Amendment 17: loss weight changes require Bills
- Amendment 19: fidelity ≤ 15% per axis before grid search
- Amendment 21: Data–Physics–Model Triad Alignment (Stage 2 pre-flight)

Now parse "$ARGUMENTS":
  If a stage number (1–5) or keyword (status) is given, run that stage only.
  If no argument: run session initialisation, print stage status, ask Justice which stage to begin.
