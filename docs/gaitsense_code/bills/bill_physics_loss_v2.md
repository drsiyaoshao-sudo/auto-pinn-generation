# Bill: Physics Loss — l_ode() True Double-Integration — v2

**Proposed by:** Legislature agent (loss-setter)
**Date:** 2026-04-03
**Status:** PENDING RATIFICATION
**Mandated by:** z_proxy Collapse Case ruling, 2026-04-03

---

## Change Type

software — `simulator/pinn/physics_loss.py`, method `l_ode()` only

## Problem Statement

The z_proxy Collapse Case (case_law.md, 2026-04-03) ruled that the existing
`l_ode()` implementation is constitutionally impermissible under Article I. The
substitution `z_proxy = d2z_dt2 / (omega2 + 1e-6)` algebraically collapses the
spring-mass ODE residual such that `cadence_spm` contributes zero gradient to
the training loss. Specifically:

```
Substituting z_proxy = d2z_dt2 / ω² into d2z_dt2 + ω²·z - F_contact:
  d2z_dt2 + ω²·(d2z_dt2 / ω²) - F_contact
= d2z_dt2 + d2z_dt2 - F_contact
= 2·d2z_dt2 - F_contact
```

The ω²·z restoring term cancels identically. cadence_spm enters only through
ω and is multiplied by zero — it contributes no gradient. Three training runs
(190, 500, 100 epochs) confirm val_ode never converges. The failure is
structural: a zero-gradient path cannot be resolved by scaling weights or
increasing epoch count.

## Proposed Change

Replace the proxy approximation with true double-integration of `(az_pred - G)`
to obtain `z_pred(t)`, then compute the correct ODE residual
`d2z_dt2 + ω²·z_pred - F_contact`.

File: `simulator/pinn/physics_loss.py`
Method: `l_ode()` only — `l_vel()` and `l_phase()` are unchanged.

### Article/Amendment Grounding

- Article I mandates all parameters trace to a first-order walking primitive.
  With true double-integration, `cadence_spm` enters through both `ω²` (the
  restoring-force coefficient) and `dt = step_period_s / (T_steps - 1)` (the
  integration time step). Both paths carry non-zero gradient. This satisfies
  Article I.
- Amendment 17 requires every physics loss weight to be derived from the
  walking primitives. The current `lambda_ode` in `train_config.json` was
  derived for the collapsed residual (scale ~17,800). The new residual has a
  different scale at initialisation (~336 for random weights, ~10.8 at
  converged physics). Lambda recalibration is a separate Bill; this Bill
  covers only the structural fix mandated by the ruling.

## Algebraic Derivation of the New l_ode()

### Governing equation

The vertical CoM motion approximates a damped harmonic oscillator:

```
d²z/dt² + ω²·z = F_contact(t)
```

Primitives involved:
- cadence_spm → ω = 2π × (cadence_spm / 60)  [rad/s]   [Article I: cadence]
- vertical_oscillation_cm → F_contact via hs_impact_ms2   [Article I: vert osc]

### Derivation of z_pred from az_pred

The IMU sensor measures vertical acceleration in the sensor frame where gravity
adds a positive baseline. Therefore:

```
az_pred ≈ d²z/dt² + G
→  d²z/dt² = az_pred - G        [d2z_dt2, shape (N,)]
```

True double-integration in normalised time, per-profile segment:

```
Physical time step:   dt = step_period_s / (T_steps - 1)   [s]
  step_period_s = 60 / cadence_spm  [traces to cadence_spm]

Velocity:  v_z(t_k) = Σ_{j=0}^{k} d2z_dt2[j] × dt        (cumsum × dt)
Position:  z_pred(t_k) = Σ_{j=0}^{k} v_z[j] × dt          (cumsum × dt again)
```

Both cumsums are applied per-profile segment (each of length T_steps=100),
not across the full flattened batch. Integrating across profile boundaries
would accumulate drift between physically independent walkers.

Drift correction: subtract the mean of each segment's z_pred so the integral
is zero-mean. This removes DC bias from random initialisation without
destroying the oscillatory structure that carries the ODE residual signal.

```
z_pred_seg = z_pred_seg - mean(z_pred_seg)
```

### ODE Residual

```
residual = d2z_dt2 + ω²·z_pred - F_contact
L_ODE = mean(residual²)
```

With true double-integration, cadence_spm appears in:
1. ω² = (2π × cadence_spm / 60)²            [restoring force coefficient]
2. dt = step_period_s / (T_steps - 1)        [integration time step magnitude]

Both contribute non-zero gradient to L_ODE with respect to network weights.
The z_proxy collapse is eliminated.

### Autograd requirement

`az_pred` must retain `requires_grad=True` through both `torch.cumsum` calls.
A `assert az_pred.requires_grad` guard is included at the start of `l_ode()`,
consistent with the guard in `l_vel()`.

### Per-profile integration: implementation

For a flattened batch of shape (B×T_steps,), the per-profile integration is:

```python
# Reshape to (B, T_steps), cumsum along dim=1, reshape back
d2z_mat = d2z_dt2.reshape(B, T_steps)             # (B, T_steps)
v_z_mat = torch.cumsum(d2z_mat, dim=1) * dt       # (B, T_steps)
z_mat   = torch.cumsum(v_z_mat,  dim=1) * dt      # (B, T_steps) — metres
z_mat   = z_mat - z_mat.mean(dim=1, keepdim=True) # drift correction
z_pred  = z_mat.reshape(-1)                        # (N,)
```

B is inferred as `N // T_steps`. T_steps defaults to 100.

## Expected New l_ode Magnitude

### At random initialisation (epoch 0, random network weights)

For random-init network: `az_pred ~ N(0, 1)`.
After removing gravity: `d2z_dt2 ~ N(-9.81, 1)`.

After drift-corrected double-integration (numerical, per profile):

```
dt (flat,  105 spm): step_period_s = 0.5714s → dt = 0.5714/99 = 0.00577s
dt (slope,  95 spm): step_period_s = 0.6316s → dt = 0.6316/99 = 0.00638s
dt (stairs, 70 spm): step_period_s = 0.8571s → dt = 0.8571/99 = 0.00866s

Random-walk z_pred magnitude (drift-corrected):
  σ_z ~ σ_d2z × dt² × T_steps^(3/2) / scaling_factor
  For flat: ~ 1 × (0.00577)² × 1000 ≈ 0.033 m
  For stairs: ~ 1 × (0.00866)² × 1000 ≈ 0.075 m

ω²·z_pred contribution at init:
  flat:   ω²=120.90 × 0.033 ≈  3.99 m/s²
  slope:  ω²= 98.97 × 0.041 ≈  4.06 m/s²
  stairs: ω²= 53.73 × 0.075 ≈  4.03 m/s²

d2z_dt2 mean at init (before drift correction of z, not d2z_dt2): ~ -9.81 m/s²
F_contact peak:
  flat:    12.53 m/s²
  slope:   14.01 m/s²
  stairs:   0.00 m/s² (toe-strike)

Residual ~ d2z_dt2 + ω²·z_pred - F_contact
  flat init:   ~ -9.81 + 3.99 - 12.53 × exp(-(t-0.025)²/2/0.05²)  → varies in t
  Mean residual squared (rough estimate, averaged over t): ~ (9.81)² + (4.0)² ≈ 112

Estimated L_ODE at init: ~100–150 (order of magnitude)
```

### At converged physics (ruling estimate)

Per the z_proxy Collapse Case ruling:

```
At cadence=100 spm: ω² ≈ 109.7
At vert_osc=3 cm: z_pred peak ≈ 0.03 m
ω²·z_pred ≈ 3.29 m/s²
Residual (converged): mean((3.29 - F_contact + residual_ODE)²) → ~10.8
```

### Lambda_ode recalibration note

The existing `lambda_ode = 1.3425e-04` was calibrated for the collapsed
residual (L_ODE ~ 17,804 at init). With the new true-integration residual
(L_ODE ~ 100–150 at init), the λ_ode needs recalibration:

```
New λ_ode estimate = 1 / mean_profile_L_ODE_at_init ≈ 1/120 ≈ 8.3e-03
```

This is ~62× larger than the current value. Recalibration is a separate Bill
(bill_loss_weights_v2 would be misnamed — this Bill is bill_physics_loss_v2
addressing the structural fix only). The Justice's ruling mandates:
"Do NOT change train_config.json — lambda_ode recalibration is a separate Bill
if needed."

The current `lambda_ode = 1.3425e-04` will under-weight the new l_ode term
relative to l_vel and l_phase at initialisation. This is acceptable for the
first post-fix training run: the gradient signal is now structurally present
(cadence_spm is active), even if the weighting is sub-optimal.

## λ Derivation (Magnitude Analysis for New Residual)

For documentation purposes only — λ recalibration proceeds under a separate Bill.

Per-profile L_ODE scale at init with true double-integration:

| Profile | cadence_spm | step_period_s | dt (s) | ω² (rad²/s²) | z_pred_scale (m) | ω²·z_pred (m/s²) | F_contact_peak (m/s²) | Estimated L_ODE_init |
|---|---|---|---|---|---|---|---|---|
| flat | 105 | 0.5714 | 0.00577 | 120.90 | 0.033 | 3.99 | 12.53 | ~110 |
| bad_wear | 105 | 0.5714 | 0.00577 | 120.90 | 0.033 | 3.99 | 12.53 | ~110 |
| stairs | 70 | 0.8571 | 0.00866 | 53.73 | 0.075 | 4.03 | 0.00 | ~130 |
| slope | 95 | 0.6316 | 0.00638 | 98.97 | 0.041 | 4.06 | 14.01 | ~120 |
| **mean** | — | — | — | — | — | — | — | **~118** |

Profile range ratio: max/min = 130/110 = 1.18× — well within the 10× threshold.
This confirms the new residual is physically consistent across all 4 profiles.

## Physical Evidence

1. Three training runs with z_proxy implementation all show val_ode plateau
   (190 epochs v1: ~38-40; 500 epochs v2: non-converging; 100 epochs v3: same)
   as recorded in the z_proxy Collapse Case ruling.

2. Algebraic proof of zero-gradient: z_proxy substitution shown above cancels
   ω²·z identically. cadence_spm × 0 = 0.

3. With true double-integration: ∂L_ODE/∂(cadence_spm) is non-zero because
   cadence_spm appears in dt (scaling the cumsum magnitude) AND in ω²
   (the restoring coefficient multiplying z_pred). These are independent
   gradient paths through the computation graph.

## Expected Outcome

After this change:
- val_ode should decrease toward ~10.8 as training proceeds (per ruling estimate)
- val_ode should no longer plateau at ~38–40 or higher
- cadence_spm will contribute gradient signal to PINN weight updates via l_ode
- The `assert az_pred.requires_grad` guard will catch any accidental detachment
  of az_pred before the double-integration

Measurable criterion: after 100 warmup epochs, val_ode < 20.0 (half the
current plateau). If val_ode remains above 30.0 after 200 epochs with full
physics ramp, escalate to human — may indicate lambda_ode recalibration is
needed urgently (separate Bill).

## Branch

constitution-style-management (current working branch)

## Amendment 17 Compliance Declaration

Each component of l_ode() — ω², z_pred, F_contact — traces to at least one
of the three walking primitives:
- ω² traces to cadence_spm (Article I primitive 2)
- dt traces to cadence_spm via step_period_s = 60 / cadence_spm
- F_contact traces to vertical_oscillation_cm (Article I primitive 1)
- z_pred is the double-integral of (az_pred - G), carried entirely on the
  live computation graph via torch.cumsum — no numpy, no detach.

No weight was empirically tuned in this change. The structural fix is
mandated by the z_proxy Collapse Case ruling (Article I + Amendment 14).

## Files to Be Written Upon Ratification

- `simulator/pinn/physics_loss.py` — rewrite of `l_ode()` method only;
  `l_vel()`, `l_phase()`, and `total_loss()` are unchanged.
- `train_config.json` is NOT changed by this Bill.
