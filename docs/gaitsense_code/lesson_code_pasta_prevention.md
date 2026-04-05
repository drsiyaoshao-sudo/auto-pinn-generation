# Lesson: Code Pasta Prevention

**Identified:** 2026-04-05 (v10–v21 session)  
**Problem:** Session generated ~8 ad-hoc scripts and inline diagnostics that are not connected to the agent infrastructure and cannot be reused by future agents.

---

## What Happened

The v10–v21 session produced the following throwaway code:

| File | What it does | What it should have been |
|------|-------------|--------------------------|
| `simulator/pinn/plot_v20_gy.py` | One-off gy/az step plot for v20/v21 | A standard plot utility callable by `pinn-validator` and `plot-orchestrator` |
| `simulator/pinn/run_v20_anchor_sim.py` | Runs PINN checkpoint on 4 anchor profiles | Part of `simulator-operator`'s standard interface, not a new script |
| Inline `python -c "..."` Bash calls | Checking gy ranges, loading checkpoints | Should not exist — use dedicated read/analysis tools |
| `simulator/pinn/physics_review_log.json` | Ad-hoc physics review output | Should be `physics-reviewer`'s standard output format |

Additionally, the entire session handled architecture, training, physics diagnosis, and simulation inline rather than routing through designated agents:

| Task | How it was done | How it should be done |
|------|-----------------|-----------------------|
| Neural architecture change | Main session edited pinn_model.py directly | Invoke `layer-setter` agent |
| Hyperparameter config | Main session edited train_config.json directly | Invoke `pinn-compiler` agent |
| Physics loss derivation | Main session reasoned inline, edited physics_loss.py | Invoke `loss-setter` → `physics-reviewer` |
| Training run | Main session ran `python -m simulator.pinn.train_pinn` | Invoke `pinn-executor` agent |
| Simulation run | Main session wrote and ran run_v20_anchor_sim.py | Invoke `simulator-operator` agent |

---

## The Code Pasta Pattern

Code pasta occurs when:
1. A task is handled inline because it's "quick to write"
2. The output is session-specific (hardcoded checkpoint path, hardcoded model params)
3. The script is not connected to the agent that owns that task domain
4. The next session has no idea what the script does or whether it's canonical

Signs you are generating code pasta:
- The script has a version number in the filename (`run_v20_...`)
- The script hardcodes a checkpoint path
- The script duplicates logic from an existing agent
- The script will only work for the current session's state

---

## Prevention Rules

### Rule 1: Match task to agent before writing code

Before writing any code, ask: which agent owns this task?

```
architecture change   →  layer-setter
hyperparameters       →  pinn-compiler
physics loss          →  loss-setter + physics-reviewer
training run          →  pinn-executor
checkpoint archival   →  pinn-archivist
validation plots      →  pinn-validator + plot-orchestrator
simulation run        →  simulator-operator
data generation       →  synthetic-data-generator
```

If the agent exists, invoke it. Do not write code for it.

### Rule 2: If a script must be written, write it as a standard module

Standard module criteria:
- No hardcoded version numbers or checkpoint paths — all paths from config or arguments
- Has a clear interface (function signature or CLI flags)
- Is documented: what it takes, what it returns
- Lives in the correct module directory, not in a session-specific location
- Is callable by the agent that owns its domain

### Rule 3: Inline Python via Bash is only allowed for 3-line sanity checks

Acceptable:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import numpy as np; print(np.load('file.npy').shape)"
```

Not acceptable:
```bash
python -c "
import sys, torch, numpy as np
# ... 40 lines of inference + plotting code ...
"
```
Anything longer than 3 lines → write a proper module.

### Rule 4: Diagnostic plots belong to pinn-validator / plot-orchestrator

Any plot of model output, loss curves, or signal comparison is diagnostic evidence.
Diagnostic evidence is the domain of `pinn-validator` (post-training check) and
`plot-orchestrator` (evidence collection for hearings). Write plots to those agents'
standard output locations, not to session-specific filenames.

---

## Standard Code Infrastructure Needed (Next Session)

The user will build a standard code layer for agents. Items needed:

| Module | Purpose | Owner agent |
|--------|---------|-------------|
| `simulator/pinn/inference.py` | Load checkpoint + run forward pass for N profiles | simulator-operator, pinn-validator |
| `simulator/pinn/evaluate_anchor.py` | Run 4 anchor profiles, return step counts + SI | simulator-operator |
| `simulator/pinn/plot_step_unit.py` | Plot one canonical step (gy + az) for any checkpoint | plot-orchestrator, pinn-validator |
| `simulator/pinn/threshold_adapter.py` | Compute PINN-adapted gait algorithm thresholds from signal amplitude | simulator-operator |

Until this infrastructure exists, the workaround scripts (`plot_v20_gy.py`, `run_v20_anchor_sim.py`) are
**temporary placeholders only** — not canonical code.
