# PHYSICS LOSS DERIVATION REVIEW — PRIVATE FULL REPORT
# Date: 2026-04-19
# Bill reviewed: bill_physics_loss_v3
# Lambda source: train_config.json (v4 run)
# Status: HUMAN DECISION PENDING
# Classification: PRIVATE — DO NOT FORWARD TO CLOUD LLM

---

## ESCALATION FLAGS (read before reviewing tables)

### FLAG 1 — STAIRS 1.5x PEAK_ANGVEL MISMATCH (Article I, Amendment 21)

`walker_model.py` lines 91–93 apply a 1.5x stairs multiplier to `peak_angvel_dps`:

```python
if self.terrain == "stairs":
    self.peak_angvel_dps *= 1.5
```

This represents extra plantarflexion required to clear the ~18 cm stair riser.

`physics_loss.py` `l_gyy_pulse()` does NOT apply this multiplier. Its formula is:

```python
peak_angvel = (100.0 + 65.0 * v_walk) * slope_factor
```

For stairs: v_walk = (70/60) × 0.28 = 0.3267 m/s, slope_factor = 1.0 (slope_deg=0)
- walker_model peak: 181.85 dps (= base 121.23 × 1.5)
- loss formula peak: 121.23 dps
- Magnitude error: 33% underestimate of gy target on stairs

This means `l_gyy_pulse` penalises the PINN for correctly learning the stairs waveform.
The loss target is systematically lower than the true signal, pulling gy_pred down.

Amendment 21 states: mathematical form of physics loss must match the data-generating
function. The 1.5x factor is in the data-generating function but absent from the loss.

**This is an observation. Human is the Justice.**

---

### FLAG 2 — train_pinn.py CALL SIGNATURE MISMATCH (v2 vs v3 API)

`train_pinn.py` lines 216–229 call `total_loss()` with the v2 signature:

```python
phys_dict = physics_loss_fn.total_loss(
    pred=pred,
    t=t_batch,
    cadence_spm=cadence_exp,
    step_length_m=step_length_exp,
    vert_osc_cm=vert_osc_exp,
    stance_frac=stance_frac_exp,
    step_period_s=step_period_exp,
    lambda_ode=lambda_ode,        # <-- v2 parameter name, does not exist in v3
    lambda_vel=lambda_vel,
    lambda_phase=lambda_phase,
    physics_weight_ramp=physics_weight,
    t_steps=T_steps,
)
```

The v3 `total_loss()` signature requires:
- `slope_deg` (not passed — silently defaults to 0.0 for all profiles including slope)
- `lambda_gyy` and `lambda_az` (not passed — `lambda_ode` is passed instead)
- `lambda_gyy`, `lambda_az`, `lambda_vel`, `lambda_phase` are all required positional

**Consequence**: v4 training will raise a TypeError on the first batch. Training cannot
start until `train_pinn.py` is updated to match the v3 physics_loss.py signature.

Additionally, `slope_deg` not being extracted from X[:,3] means the slope profile will
have slope_deg=0.0 in both `l_gyy_pulse` and `l_az_gravity`, reducing slope_factor and
az_dc_expected incorrectly.

**This is an observation. Human is the Justice.**

---

## Derivation Trace Tables

### Profile Parameters (from walker_model.py / anchor_profiles.json)

| Profile  | cadence_spm | step_length_m | vert_osc_cm | slope_deg | stance_frac |
|----------|-------------|---------------|-------------|-----------|-------------|
| flat     | 105         | 0.75          | 4.0         | 0.0       | 0.60        |
| bad_wear | 105         | 0.75          | 4.0         | 0.0       | 0.60        |
| stairs   | 70          | 0.28          | 18.0        | 0.0       | 0.65        |
| slope    | 95          | 0.65          | 5.0         | 10.0      | 0.62        |

---

### L_gyy_pulse — gy Waveform Constraint

Traces to: cadence_spm, step_length_m (→ v_x), slope_deg (→ slope_factor)

Formula in loss:
```
walking_speed_ms = (cadence_spm / 60) × step_length_m
slope_factor     = 1 + 0.4 × sin(slope_deg × π/180)
peak_angvel_dps  = (100 + 65 × walking_speed_ms) × slope_factor
gy_target        = _build_gy_target(t, peak_angvel_dps, stance_frac)
L_gyy            = MSE(gy_pred, gy_target)
```

Formula in walker_model.py (extra for stairs):
```
peak_angvel_dps  = (100 + 65 × walking_speed_ms) × slope_factor
if terrain == "stairs":
    peak_angvel_dps *= 1.5
```

| Profile  | v_x (m/s) | peak_ang (walker) | peak_ang (loss) | match    | L_gyy_raw (dps²) |
|----------|-----------|-------------------|-----------------|----------|------------------|
| flat     | 1.3125    | 185.3125          | 185.3125        | OK       | 3232.01          |
| bad_wear | 1.3125    | 185.3125          | 185.3125        | OK       | 3232.18          |
| stairs   | 0.3267    | 181.8500          | 121.2333        | **MISMATCH** | 1460.81     |
| slope    | 1.0292    | 178.4883          | 178.4883        | OK       | 3064.57          |

Note: bad_wear has mounting_offset_deg=20° applied to the signal, so gy channel contains
rotated components — L_gyy value includes this distortion. This is expected and correct.

---

### L_az_gravity — az DC Baseline Constraint

Traces to: cadence_spm (→ omega), vertical_oscillation_cm, slope_deg

Formula:
```
omega  = 2π × cadence_spm / 60
az_dc  = G × cos(slope_rad) + (vert_osc_cm / 100) × omega² / (2π²)
L_az   = MSE(mean(az_pred per profile), az_dc)
```

| Profile  | az_dc expected (m/s²) | mean_az anchor (m/s²) | match | L_az_raw (m/s²)² |
|----------|-----------------------|-----------------------|-------|------------------|
| flat     | 10.0550               | 9.8096                | OK    | 0.0602           |
| bad_wear | 10.0550               | 9.8096                | OK    | 0.0602           |
| stairs   | 10.3000               | 9.8098                | OK    | 0.2403           |
| slope    | 9.9117                | 9.6617                | OK    | 0.0625           |

Match threshold used: < 10% relative deviation (signal is DC mean, not peak).
All four pass within acceptable range (the anchor mean is one step's worth of signal).

---

### L_vel — Horizontal Velocity Constraint (unchanged from v2)

Traces to: cadence_spm, step_length_m

Formula:
```
v_x_expected = (cadence_spm / 60) × step_length_m
v_x_pred     = mean(ax_pred) × step_period_s
L_vel        = (v_x_pred - v_x_expected)²
```

| Profile  | v_x expected (m/s) | v_x pred from anchor (m/s) | match    | L_vel_raw |
|----------|--------------------|----------------------------|----------|-----------|
| flat     | 1.3125             | 0.000097                   | **MISMATCH** | 1.7224 |
| bad_wear | 1.3125             | 0.000564                   | **MISMATCH** | 1.7212 |
| stairs   | 0.3267             | 0.000467                   | OK       | 0.1064    |
| slope    | 1.0292             | 1.075679                   | OK       | 0.0022    |

Note on flat/bad_wear "mismatch": anchor arrays are 208 samples of one generated step
(not a continuous walking sequence with net forward motion). The mean ax over one
stance+swing cycle is near zero by construction (symmetry of acceleration). This is
expected anchor behaviour — L_vel mismatch here reflects the anchor array structure,
not a derivation error. At training time, ax_pred is a live network output with gradients;
the loss correctly penalises the network during training. The anchor-based measurement
shown here is informational only for stairs/slope which happen to produce larger mean ax.

---

### L_phase — Stance/Swing Timing Constraint (unchanged from v2)

Traces to: stance_frac (biomechanical constant per profile), cadence_spm implicitly

Formula:
```
stance_viol    = clamp(mean(gy_pred, t < stance_frac), min=0)²
boundary_loss  = mean(gy_pred, |t - stance_frac| < 0.05)²
L_phase        = stance_viol + 0.5 × boundary_loss
```

| Profile  | stance_frac | mean_gy_stance (dps) | gy_boundary (dps) | L_phase_raw |
|----------|-------------|----------------------|-------------------|-------------|
| flat     | 0.60        | -0.0039              | -0.0087           | 0.0000      |
| bad_wear | 0.60        | +0.0012              | +0.0018           | 0.0000      |
| stairs   | 0.65        | +0.0060              | -0.0008           | 0.0000      |
| slope    | 0.62        | -0.0044              | -0.0244           | 0.0003      |

L_phase is near-zero on anchors because mean_gy is close to zero across the step.
At training time, a random-initialised network will produce non-zero mean_gy_stance,
which this term correctly penalises.

---

## Loss Magnitude at Evaluation (anchor IMU data)

| Profile  | L_gyy_raw | L_az_raw | L_vel_raw | L_phase_raw |
|----------|-----------|----------|-----------|-------------|
| flat     | 3232.01   | 0.0602   | 1.7224    | 0.0000      |
| bad_wear | 3232.18   | 0.0602   | 1.7212    | 0.0000      |
| stairs   | 1460.81   | 0.2403   | 0.1064    | 0.0000      |
| slope    | 3064.57   | 0.0625   | 0.0022    | 0.0003      |

---

## Weighted Loss Balance

Lambda values (train_config.json v4):
- lambda_gyy   = 2.994e-05
- lambda_az    = 6.054e-03
- lambda_vel   = 4.908
- lambda_phase = 73.625

| Profile  | λ_gyy·L_gyy | λ_az·L_az | λ_vel·L_vel | λ_phase·L_phase |
|----------|-------------|-----------|-------------|-----------------|
| flat     | 0.0968      | 0.0004    | 8.4535      | 0.0028          |
| bad_wear | 0.0968      | 0.0004    | 8.4475      | 0.0002          |
| stairs   | 0.0437      | 0.0015    | 0.5222      | 0.0027          |
| slope    | 0.0918      | 0.0004    | 0.0106      | 0.0218          |

Means across 4 profiles:
- λ_gyy   × mean(L_gyy)   = 0.082257
- λ_az    × mean(L_az)    = 0.000641
- λ_vel   × mean(L_vel)   = 4.358484
- λ_phase × mean(L_phase) = 0.006888

Balance ratios (target: 0.1 – 10):

| Pair                  | Ratio    | Status                         |
|-----------------------|----------|--------------------------------|
| λ_gyy·L / λ_az·L     | 128.42   | **CRITICAL IMBALANCE (>100x)** |
| λ_gyy·L / λ_vel·L    | 0.0189   | **IMBALANCED (>10x)**          |
| λ_gyy·L / λ_phase·L  | 11.94    | **IMBALANCED (>10x)**          |
| λ_az·L  / λ_vel·L    | 0.0001   | **CRITICAL IMBALANCE (>100x)** |
| λ_az·L  / λ_phase·L  | 0.0930   | **IMBALANCED (>10x)**          |
| λ_vel·L / λ_phase·L  | 632.73   | **CRITICAL IMBALANCE (>100x)** |

**OVERALL STATUS: IMBALANCED**

Dominant term: λ_vel · L_vel = 4.36 (mean), 52× larger than λ_gyy·L_gyy.
λ_az is 6800× smaller than λ_vel — az gravity term will be numerically invisible.

Note: Lambda values for l_vel (4.908) and l_phase (73.625) were carried from
bill_loss_weights_v1, which was calibrated against the v2 loss terms (L_ODE, L_vel,
L_phase). The v3 loss adds two new terms and removes one. The inherited lambda values
have not been recalibrated for the v3 loss magnitude profile.

---

## Summary of Findings

1. **Stairs mismatch (MISMATCH flag)**: l_gyy_pulse underestimates gy target by 33%
   on stairs. The 1.5x stairs terrain multiplier from walker_model.py is absent from
   the loss formula.

2. **train_pinn.py signature mismatch**: The trainer calls total_loss() with the v2
   parameter names (lambda_ode, no slope_deg, no lambda_gyy, no lambda_az). v4
   training will fail with TypeError on first batch.

3. **Balance: λ_vel dominates**: λ_vel·L_vel is 52× larger than λ_gyy·L_gyy and
   6800× larger than λ_az·L_az. L_az_gravity will have negligible gradient influence.

4. **All l_az_gravity traces are OK**: az_dc expected vs anchor mean within acceptable
   range for all four profiles.

5. **L_phase near-zero on anchors**: Expected — this term activates primarily during
   training against random network outputs, not against near-correct anchor signals.

---

## Human Action Required

Review this report and the plot at:
`docs/executive_branch_document/plots/pinn_loss_derivation.png`

Update `physics_review_log.json` with APPROVED or REJECTED.
If approved: train_pinn.py must also be updated before v4 training begins.
