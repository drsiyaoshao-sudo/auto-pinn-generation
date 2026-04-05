### BILL: Training Config v4 — Amendment 20 physics-first warmup + LR fix

Proposed by: Justice direction (2026-04-04)
Date drafted: 2026-04-04
Change type: software (PINN training script + hyperparameters)

---

**Problem statement:**
v3 training stalled at val_ode ~38 after 100 epochs. Two root causes identified:

1. **train_pinn.py warmup logic is inverted for Amendment 20.** The `physics_weight_ramp`
   starts at 0 and ramps to 1 over `warmup_epochs`, meaning DATA is dominant at epoch 1
   and physics ramps in gradually. Amendment 20 mandates the opposite: physics dominant
   from epoch 1, data suppressed during warmup. The current script structure is a
   data-first warmup — a constitutional violation of Amendment 20 when `warmup_epochs > 0`.
   v3 set `warmup_epochs = 0` (physics_weight = 1.0 from epoch 1), which was Amendment 20
   compliant, but still insufficient due to issue 2 below.

2. **Learning rate was 10× too low.** v3 used lr = 0.0001. v1 (which achieved val_loss 0.44)
   used lr = 0.001. With large λ_ode (0.1031) and small LR, the ODE gradients cannot
   overcome the loss plateau. The v3 diagnostic run showed ODE declining 48.66 → 43.72 over
   30 epochs — measurable convergence, but too slow to complete in 100 epochs at low LR.

---

**Proposed changes:**

### A. train_pinn.py — Flip warmup logic for Amendment 20 compliance

New behaviour:
- During `epoch <= warmup_epochs`: physics at full weight (physics_weight = 1.0),
  data loss suppressed (data_weight = 0.0). Model learns physics manifold first.
- After `epoch > warmup_epochs`: physics remains at full weight, data ramps in
  linearly over `data_ramp_epochs` from 0.0 to 1.0.
  
Replaces the existing `physics_weight_ramp` (which was a data-first ramp).

### B. train_config.json — v4 hyperparameters

| Parameter | v3 | v4 | Reason |
|---|---|---|---|
| `run_id` | v3 | v4 | new run |
| `lr_initial` | 0.0001 | 0.001 | 10× increase — match v1 which converged |
| `lr_scheduler_T_max` | 100 | 1000 | match epochs |
| `lr_scheduler_eta_min` | 1e-5 | 1e-6 | lower floor |
| `epochs_max` | 100 | 1000 | sufficient for convergence at correct LR |
| `physics_loss_warmup_epochs` | 0 | 100 | physics-only phase (Amendment 20) |
| `data_ramp_epochs` | — | 100 | epochs over which data ramps 0→1 after warmup |
| `early_stop_min_epoch` | 100 | 400 | don't stop before both phases complete |
| `early_stop_patience` | 100 | 100 | unchanged |
| `lambda_ode` | 0.1031 | 0.1031 | unchanged — physics dominant (Amendment 20) |
| `lambda_vel` | 4.908 | 4.908 | unchanged |
| `lambda_phase` | 73.625 | 73.625 | unchanged |
| `load_checkpoint` | best_v1.pt | best_v3.pt | warm start from latest weights |

---

**Article/Amendment grounding:**
- **Amendment 20**: Physics-first training order. warmup_epochs = 100 with
  data_weight = 0.0 during warmup guarantees physics ≥ 100% of total loss in
  the warmup phase. Satisfies Amendment 20 criterion 2 (physics dominant, verified
  downward trend required before data ramps in).
- **Amendment 13**: Hyperparameters are calibration constants requiring a Bill.
- **Article I**: λ values unchanged — physics traces to three walking primitives.
- **bill_physics_loss_v2**: double-integration ODE already implemented in
  physics_loss.py — no change required. This Bill uses it as ratified.

---

**Expected outcome:**
- Warmup phase (epochs 1–100): val_ode shows measurable downward trend.
  Target: val_ode < 30 by epoch 100 (from v3 start of 38.5).
- Data phase (epochs 101–200): data ramps in without destabilising physics.
- Convergence phase (epochs 200–1000): val_total converges toward < 1.0.
- Best checkpoint epoch > 200 (Amendment 20 PASS — best epoch after warmup).

---

**Branch:** constitution-style-management
