### BILL: Physics Loss v6 — Drop L_ODE and L_vel, data-dominant training

Proposed by: Justice direction (Option A, 2026-04-05)
Date drafted: 2026-04-05
Change type: software (train_config.json — lambda_ode=0, lambda_vel=0, warmup=0)

Supersedes: bill_physics_loss_v2 (L_ODE double-integration), bill_physics_loss_v3 (DC correction)

---

**Problem statement:**
Diagnostic run on 350 training profiles (2026-04-05) confirmed that both L_ODE and L_vel
are inconsistent with the walker_model forward model:

L_ODE (spring-mass ODE for az):
  az=0 random init → ODE residual = 38.85
  True walker_model az → ODE residual = 53–147 across profiles

  The training data gives HIGHER ODE residual than az=0. The spring-mass model does not
  describe walker_model's az generation (empirical heel-strike Gaussian + gait dynamics).
  L_ODE actively pulls az AWAY from the true signal — every gradient step that reduces
  L_ODE worsens the data match.

L_vel (horizontal velocity constraint):
  mean(ax_true) × step_period ≠ walking_speed for most profiles:
    Profile 3: v_expected=1.354 m/s, v_from_data=0.189 m/s (−86%)
    Profile 6: v_expected=0.770 m/s, v_from_data=1.889 m/s (+145%)

  The walker_model's ax signal does not integrate to walking speed via the mean-×-period
  approximation used in l_vel(). L_vel also pulls ax away from data.

Result: v10–v13 training runs all diverged. Data loss at epoch 500 (663) exceeded random
init baseline (521). The physics warmup (Amendment 20) made things worse — 100 epochs of
wrong physics set the weights in the wrong direction.

**Root cause: the physics constraints enforce a theoretical spring-mass model. The training
data comes from an empirical walker_model that does not satisfy these equations. The
physics constraints and the data objectives are incompatible.**

---

**Proposed change:**

1. Set lambda_ode = 0 and lambda_vel = 0 in train_config.json.
   L_ODE and L_vel are computed but multiplied by zero — no gradient, no influence.

2. Keep lambda_phase = 0.1 (sign hinge only, bill_physics_loss_v5).
   L_phase sign hinge (mean gy_stance < 0) is verified against training data:
   gy during stance IS negative in walker_model data. This constraint is consistent
   and provides a minimal physical sanity check on gyr_y direction.

3. Set physics_loss_warmup_epochs = 0 and data_ramp_epochs = 0.
   Physics-only warmup was harmful because the warmup physics (L_ODE, L_vel) conflict
   with data. With only L_phase (sign hinge, λ=0.1), warmup provides no useful signal.
   Data-dominant from epoch 1.

4. Increase lr_initial to 0.003, T_max=500.
   Higher LR for faster convergence on data-dominant regime.

---

**Article/Amendment grounding:**
- Article I: L_phase sign hinge traces to cadence_spm (stance_frac encodes the
  proportion of the step period in stance). Physical basis preserved.
- Article II: Human Justice ruling after diagnostic evidence showing L_ODE and L_vel
  are inconsistent with the walker_model forward model. Three failed training runs
  (v10–v13) as physical evidence of divergence.
- Amendment 20: Overridden for this configuration. Physics-first warmup is beneficial
  only when the physics constraints are consistent with the training data. They are not.

**Long-term path (Option B, deferred):**
Rederive L_ODE and L_vel from walker_model.py's actual forward equations — heel-strike
Gaussian + empirical peak_angvel fit. This would restore physics-grounded constraints
that are consistent with the data. Deferred pending training convergence with data-only.

**Expected outcome:**
- Data loss converges below random-init baseline (521) for the first time
- L_phase sign hinge satisfied throughout (gyr_y stance mean < 0)
- gyr_y amplitude learned from data → reaches ±30+ dps zero-crossings
- Step count within ±15% on all 4 anchor profiles
- SI_stance < 3% on all 4 anchor profiles (si_true=0)

**Branch:** hybrid-model
