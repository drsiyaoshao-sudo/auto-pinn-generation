# GaitSense — Evidence Session Architecture

**Constitutional grounding:** Bureaucracy Standing Orders — Signal Plotting, Instrument API Calls  
**Purpose:** How evidence is collected and presented during Judicial Hearings and validation runs  
**Last updated:** 2026-04-04

---

```
  EVIDENCE SESSION ARCHITECTURE
  ════════════════════════════════════════════════════════════════════════

  ENTRY POINTS
  ────────────
  Two callers can open an evidence session:

  ┌─────────────────────┐          ┌─────────────────────┐
  │   JUSTICE (human)   │          │  PINN-EXECUTOR      │
  │                     │          │  (training run      │
  │  /plot-evidence     │          │   complete)         │
  │  skill              │          │                     │
  │                     │          │  dispatches after   │
  │  types:             │          │  each run:          │
  │  signal  <profile>  │          │  evidence type      │
  │  uart    <log>      │          │  = all-pinn         │
  │  training <run_id>  │          └──────────┬──────────┘
  │  sim     <profile>  │                     │
  │  pinn    <run_id>   │          ┌──────────┴──────────┐
  └──────────┬──────────┘          │  SIMULATOR-OPERATOR │
             │                     │  (Renode run        │
             │                     │   complete)         │
             │                     │                     │
             │                     │  dispatches after   │
             │                     │  each profile:      │
             │                     │  evidence type      │
             │                     │  = all-simulation   │
             │                     └──────────┬──────────┘
             │                                │
             └──────────────┬─────────────────┘
                            │
                            ▼
  ════════════════════════════════════════════════════════════════════════
  PLOT-ORCHESTRATOR  ◄── single evidence coordination layer
  ════════════════════════════════════════════════════════════════════════
  │
  │  Routes by evidence type. Dispatches sub-agents sequentially.
  │  Collects outputs. Prints consolidated evidence block.
  │  Assesses Amendment 20 compliance for training evidence.
  │
  │  Does NOT generate plots, parse UART, or read logs directly.
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  │  evidence type = signal
  │  ─────────────────────
  │  input:  profile name + mode
  │                    │
  │                    ▼
  │         ┌──────────────────┐
  │         │    PLOTTER       │
  │         │                  │
  │         │ walker_model.py  │
  │         │ → IMU signal     │
  │         │ → firmware       │
  │         │   filters        │
  │         │ → annotate:      │
  │         │   thresholds     │
  │         │   step markers   │
  │         │   zero-crossings │
  │         │   timing gaps    │
  │         │                  │
  │         │ prints data      │
  │         │ table to stdout  │
  │         │                  │
  │         │ saves →          │
  │         │ docs/.../plots/  │
  │         │ <profile>_       │
  │         │ signal_check.png │
  │         └────────┬─────────┘
  │                  │
  │                  ▼
  │         evidence block:
  │         PLOT: <path>
  │         peak values, zero-crossing ts, gap_ms
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  │  evidence type = uart
  │  ────────────────────
  │  input:  UART log file path or serial port
  │                    │
  │                    ▼
  │         ┌──────────────────┐
  │         │   UART-READER    │
  │         │                  │
  │         │ reads log file   │
  │         │ or serial port   │
  │         │                  │
  │         │ prints:          │
  │         │  STEP lines →    │
  │         │   ts, acc,       │
  │         │   gyr_y, cad     │
  │         │  SNAPSHOT →      │
  │         │   SI stance/     │
  │         │   swing, cad     │
  │         │  SESSION_END →   │
  │         │   total steps    │
  │         │                  │
  │         │ prints summary:  │
  │         │  steps / snaps / │
  │         │  final SI / cad  │
  │         └────────┬─────────┘
  │                  │
  │                  ▼
  │         evidence block:
  │         UART: steps / snapshots / SI% / cadence
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  │  evidence type = training
  │  ────────────────────────
  │  input:  run_id (or latest log)
  │                    │
  │                    ▼
  │         ┌──────────────────┐
  │         │   TRAIN-SUM      │
  │         │                  │
  │         │ reads            │
  │         │ training_logs/   │
  │         │ <run_id>.jsonl   │
  │         │                  │
  │         │ 4-panel plot:    │
  │         │  total loss      │
  │         │  components      │
  │         │   l_ode          │
  │         │   l_vel          │
  │         │   l_phase        │
  │         │  ramp + LR       │
  │         │  summary table   │
  │         │                  │
  │         │ saves →          │
  │         │ docs/.../plots/  │
  │         │ pinn_training/   │
  │         │ loss_curve_      │
  │         │ <run_id>.png     │
  │         └────────┬─────────┘
  │                  │
  │                  ▼
  │         evidence block:
  │         TRAINING: best_epoch / best_val_loss
  │         Amendment 20: PASS | FAIL | INCONCLUSIVE
  │         PLOT: <path>
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  │  evidence type = all-simulation  (standard hearing call)
  │  ───────────────────────────────
  │  input:  profile name + mode + UART log path
  │
  │         uart-reader first ──► then plotter
  │         (UART confirms run before plots generated)
  │                    │
  │                    ▼
  │         ┌──────────────────┐     ┌──────────────────┐
  │         │   UART-READER    │────►│    PLOTTER       │
  │         │  (see uart path) │wait │  (see signal     │
  │         │                  │     │   path)          │
  │         └──────────────────┘     └────────┬─────────┘
  │                                           │
  │                                           ▼
  │         consolidated evidence block:
  │         ─────────────────────────────────────────
  │         EVIDENCE PACKAGE — [profile] [mode] [ts]
  │         ─────────────────────────────────────────
  │         UART: steps / snapshots / SI% / cadence
  │         PLOT: <path>
  │         ─────────────────────────────────────────
  │
  ├─────────────────────────────────────────────────────────────────────
  │
  │  evidence type = all-pinn  (standard hearing call)
  │  ──────────────────────────
  │  input:  run_id
  │                    │
  │                    ▼
  │         ┌──────────────────┐
  │         │   TRAIN-SUM      │
  │         │  (see training   │
  │         │   path)          │
  │         └────────┬─────────┘
  │                  │
  │                  ▼
  │         consolidated evidence block:
  │         ─────────────────────────────────────────
  │         EVIDENCE PACKAGE — [run_id] [ts]
  │         ─────────────────────────────────────────
  │         TRAINING: best_epoch=N  best_val_loss=X
  │         Amendment 20: PASS — physics converged
  │                        before data phase
  │         PLOT: <path>
  │         ─────────────────────────────────────────
  │
  └─────────────────────────────────────────────────────────────────────
                            │
                            ▼
  ════════════════════════════════════════════════════════════════════════
  JUSTICE (human) receives consolidated evidence block
  ════════════════════════════════════════════════════════════════════════
  │
  │  Applies Benjamin Franklin Principle — empirical basis only
  │  Applies Thomas Jefferson Principle — best patient outcome
  │  Issues ruling
  │
  └──► Attorney-B writes ruling to case_law.md
       before any implementation begins


  ════════════════════════════════════════════════════════════════════════
  AGENT ROLES SUMMARY
  ════════════════════════════════════════════════════════════════════════

  plot-orchestrator   Routes evidence requests, coordinates sub-agents,
                      prints consolidated block, assesses Amendment 20.
                      Tools: Bash, Read, Glob, Agent
                      Model: sonnet

  plotter             Generates signal diagnostic plots from walker_model.py.
                      Applies firmware-matched filters. Annotates thresholds.
                      Tools: Bash, Read, Write, Glob, Grep
                      Model: haiku

  uart-reader         Reads UART log or serial port. Prints structured
                      STEP / SNAPSHOT / SESSION_END table.
                      Tools: Bash, Read, Glob, Grep
                      Model: haiku

  train-sum           Reads training .jsonl log. Generates 4-panel loss
                      curve plot. Prints summary table.
                      Tools: Read, Write, Bash, Glob
                      Model: sonnet

  ════════════════════════════════════════════════════════════════════════
  CONSTITUTIONAL CONSTRAINTS
  ════════════════════════════════════════════════════════════════════════

  Amendment 11  Signal plots are mandatory after any change to
                walker_model.py, filter coefficients, or algorithm
                parameters. plot-orchestrator enforces this.

  Amendment 20  Training evidence must show physics-dominant warmup
                (≥80% of total loss) before data phase begins.
                train-sum surfaces the data; plot-orchestrator
                assesses; Justice rules.

  Article II    No sub-agent interprets whether evidence is clinically
                correct. Evidence is presented. The Justice rules.
                plot-orchestrator prints and stops — it does not propose
                fixes, recommend positions, or suggest outcomes.
```
