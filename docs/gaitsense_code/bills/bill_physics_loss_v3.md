### BILL: Physics Loss v3 — Gravity DC correction before double-integration

Proposed by: Justice direction (Option A, 2026-04-04)
Date drafted: 2026-04-04
Change type: software (physics_loss.py — l_ode() double-integration)

Mandated by: physics-reviewer Option B investigation (2026-04-04)

---

**Problem statement:**
The double-integration in l_ode() accumulates a gravity DC offset that produces
a structurally unsolvable ODE residual. At initialisation, az_pred ≈ 0, so
d2z_dt2 = az_pred - G ≈ -9.81 m/s² (constant). Double-integrating a constant
produces a parabola with amplitude G × dt² × T²/8. At T=208 steps:

  z_pred amplitude ≈ 1.1 m  (100× larger than expected vert_osc ≈ 0.01 m)
  ω²·z_pred ≈ 130 m/s²      (6–7× larger than F_contact ≈ 18 m/s²)

The ODE residual (d2z + ω²·z - F_contact) is dominated by the parabolic z term.
The structural L_ODE minimum is ~3400 (at az=0), and even the partially-compensated
PINN minimum of ~44 cannot be trained to zero under this formulation.

---

**Proposed change — l_ode() in physics_loss.py:**

Before double-integration, subtract the per-profile temporal mean of d2z_dt2.
This removes the gravity DC component, leaving only the oscillatory signal for
integration. The physical interpretation: the PINN should enforce the spring-mass
ODE on the oscillatory deviation from the gravitational baseline, not on the full
acceleration including the DC gravitational offset.

```
# BEFORE (bill_physics_loss_v2 — accumulates gravity DC):
d2z_dt2 = az_pred - G
d2z_mat = d2z_dt2.reshape(-1, t_steps)
v_z_mat = torch.cumsum(d2z_mat, dim=1) * dt
z_mat   = torch.cumsum(v_z_mat, dim=1) * dt

# AFTER (bill_physics_loss_v3 — gravity-corrected before integration):
d2z_dt2 = az_pred - G
d2z_mat = d2z_dt2.reshape(-1, t_steps)          # (B, T)
d2z_dc  = d2z_mat.mean(dim=1, keepdim=True)     # per-profile DC offset (B, 1)
d2z_ac  = d2z_mat - d2z_dc                       # oscillatory component only (B, T)
v_z_mat = torch.cumsum(d2z_ac, dim=1) * dt       # integrate AC only
z_mat   = torch.cumsum(v_z_mat, dim=1) * dt
```

The drift correction (subtract mean of z_mat) is retained unchanged.

The ODE residual is also computed on the AC component only:
```
# residual uses d2z_ac (not d2z_dt2) to match the integration path
residual = d2z_ac.reshape(-1) + omega2 * z_pred - F_contact
```

**Physical justification:**
The spring-mass ODE governing vertical CoM motion describes oscillations about
the gravitational equilibrium position. The DC component of az_pred is the static
gravitational loading — it does not participate in the oscillatory dynamics. By
subtracting the per-profile DC before integration, we isolate the cadence-driven
oscillatory component that the ODE governs. cadence_spm remains active in ω² and
F_contact (unchanged from bill_physics_loss_v2). Article I is satisfied.

**Expected outcome:**
Expected z_pred amplitude after DC correction: ~vert_osc_m/4 (oscillatory component)
Expected ω²·z_pred: comparable to F_contact (~18 m/s²), eliminating the 6–7× dominance.
Expected structural L_ODE minimum: < 1.0 (vs current ~44).
Expected L_ODE convergence during physics-first warmup: measurable downward trend
toward < 5.0 within 100 epochs.

---

**Article/Amendment grounding:**
- Article I: cadence_spm remains in ω² and F_contact — DC correction does not
  remove any walking primitive from the gradient path.
- bill_physics_loss_v2 ruling (z_proxy Collapse Case): the double-integration path
  is retained — only the DC subtraction is added. The ω²·z restoring term with
  cadence_spm as an active signal is preserved.
- Amendment 14: physical evidence (physics-reviewer structural minimum table)
  supports this change.

**Branch:** constitution-style-management
