# GaitSense Judicial Hearing — Process Flowchart

**Constitutional grounding:** CLAUDE.md — The Judicial Process (Sections 1–5)  
**Last updated:** 2026-04-04  
**Change:** Added `pinn-executor` to hearing roster (trial training + callback analysis as physical evidence)

---

```
┌─────────────────────────────────────────────────────────────────┐
│                        JUSTICE (human)                          │
│                   declares judicial hearing                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      JUDICIAL-CLERK                             │
│  1. Print hearing header + timestamp                            │
│  2. Verify agent roster (6 agents):                             │
│       attorney-A ✓   attorney-B ✓                              │
│       simulator-operator ✓   plotter ✓                         │
│       uart-reader ✓   pinn-executor ✓                          │
│  3. Print COURTROOM READY                                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        JUSTICE                                  │
│  Declares hearing: case name + Position A + Position B          │
│  Assigns Attorney-A → Position A                                │
│  Assigns Attorney-B → Position B                                │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│       ATTORNEY-A         │    │         ATTORNEY-B           │
│  Reads:                  │    │  Reads:                      │
│    amendments.md         │    │    amendments.md             │
│    case_law.md           │    │    case_law.md               │
│    relevant src files    │    │    relevant src files        │
│  Argues Position A:      │    │  Argues Position B:          │
│    1. Amendment invoked  │    │    1. Amendment invoked      │
│    2. Precedent cited    │    │    2. Precedent cited        │
│    3. Physical outcome   │    │    3. Physical outcome       │
│    4. Opposing risk      │    │    4. Opposing risk          │
└──────────────┬───────────┘    └──────────────┬───────────────┘
               │                              │
               └──────────────┬───────────────┘
                              │  both arguments complete
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        JUSTICE                                  │
│              requests physical evidence                         │
│    (may request simulation evidence, PINN evidence, or both)    │
└────────────┬──────────────────────────────┬─────────────────────┘
             │                              │
             │ gait/firmware question        │ PINN/training question
             ▼                              ▼
┌────────────────────────────┐  ┌──────────────────────────────────┐
│     SIMULATOR-OPERATOR     │  │         PINN-EXECUTOR            │
│  1. Validate firmware ELF  │  │  1. Check existing config files  │
│     (BUG-005 guard)        │  │     (train_config.json,          │
│  2. Generate IMU signals   │  │      physics_loss.py,            │
│     from walker profiles   │  │      pinn_model.py)              │
│  3. Launch Renode,         │  │  2. Run trial training           │
│     feed IMU stub          │  │     (default 200 epochs          │
│  4. Run declared profile   │  │      unless Justice specifies)   │
│     × declared mode        │  │  3. Monitor per-epoch callbacks: │
│  5. Dispatch uart-reader   │  │     loss components, val_loss,   │
│  6. Dispatch plotter       │  │     convergence trend            │
│  7. Print results table    │  │  4. Print structured evidence    │
└────────────┬───────────────┘  │     table: epoch / total_loss /  │
             │                  │     l_ode / l_vel / l_phase /    │
             │                  │     val_loss + observation line  │
             │                  │  5. Dispatch train-sum for       │
             │                  │     loss curve plot              │
             │                  │                                  │
             │                  │  Does NOT invoke:                │
             │                  │    layer-setter, loss-setter     │
             │                  │    (require separate Bills)      │
             │                  │  Does NOT archive checkpoints    │
             │                  │    (trial run = hearing evidence  │
             │                  │     only, not production)        │
             │                  └──────────────┬───────────────────┘
             │                                 │
     ┌───────┴──────┐                  ┌───────┴──────┐
     │              │                  │              │
     ▼              ▼                  ▼              ▼
┌─────────┐  ┌──────────┐       ┌──────────────────────┐
│UART-    │  │ PLOTTER  │       │      TRAIN-SUM       │
│READER   │  │          │       │  loss curve plots    │
│         │  │ Signal   │       │  4-panel: total,     │
│ STEP →  │  │ diag-    │       │  components, ramp,   │
│ ts,acc, │  │ nostic   │       │  LR schedule         │
│ gyr,cad │  │ plots    │       │  saved to            │
│         │  │ saved to │       │  docs/.../plots/     │
│SNAPSHOT │  │ docs/... │       │  pinn_training/      │
│ SI %    │  │ /plots/  │       └──────────┬───────────┘
│         │  │          │                  │
│SESSION  │  │ Annotate:│                  │
│ _END    │  │ thresholds                  │
│         │  │ markers  │                  │
│ Print   │  │ crossings│                  │
│ summary │  │          │                  │
└────┬────┘  └────┬─────┘                  │
     │            │                        │
     └─────┬──────┘                        │
           │                               │
           └──────────────┬────────────────┘
                          │  evidence complete
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        JUSTICE                                  │
│  Reviews evidence (UART table / signal plots / loss curves)     │
│  Applies Benjamin Franklin Principle (empirical basis only)     │
│  Applies Thomas Jefferson Principle (best patient outcome)      │
│  Issues ruling: Position A or Position B prevails               │
│  States physical basis + patient outcome consequence            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ATTORNEY-B                                │
│  (clerk duty falls to Attorney-B regardless of which prevails)  │
│  Writes ruling to case_law.md using standard template           │
│  Commit required before any implementation begins               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Roster Summary

| Agent | Role in hearing | Model | Invoked when |
|---|---|---|---|
| judicial-clerk | Roster verification + readiness gate | haiku | Always — first step |
| attorney-A | Argues assigned position (parallel) | sonnet | Always — assigned by Justice |
| attorney-B | Argues assigned position (parallel) + writes case law | sonnet | Always — assigned by Justice |
| simulator-operator | Gait/firmware physical evidence | sonnet | Hearing involves firmware, FSM, or signal behaviour |
| uart-reader | UART output capture and formatting | haiku | Dispatched by simulator-operator |
| plotter | Signal diagnostic plots | haiku | Dispatched by simulator-operator |
| pinn-executor | PINN trial training + callback evidence | sonnet | Hearing involves PINN architecture, loss weights, or training behaviour |
| train-sum | Loss curve plots from trial run | sonnet | Dispatched by pinn-executor |

---

## Evidence Path Selection

The Justice selects the evidence path(s) after both attorneys complete arguments:

**Simulation path** — use when the hearing concerns:
- Firmware algorithm behaviour (step detection, phase segmentation, SI calculation)
- FSM state transitions
- Signal threshold values or window timing
- Amendment 15 / terrain gate / VABS.F32 classification

**PINN path** — use when the hearing concerns:
- Physics loss convergence order (Amendment 20)
- Loss weight balance (Amendment 17)
- Training configuration (warmup schedule, LR, early stopping)
- PINN architecture or Fourier feature encoding
- Whether a training run satisfied the physics-first criterion

Both paths may be invoked in the same hearing. Each runs independently and reports to the Justice before ruling.

---

## Constraints on pinn-executor in a Hearing

`pinn-executor` in judicial session mode does **not** invoke `layer-setter` or `loss-setter` — those require independent Bills ratified before a hearing, not during one. If the hearing reveals that architecture or loss weights must change, the ruling directs a new Bill; it does not execute the change.

Trial run checkpoints produced during a hearing are **hearing evidence only** and are not archived by `pinn-archivist`. A production training run with archival requires the full precondition chain outside the hearing context.
