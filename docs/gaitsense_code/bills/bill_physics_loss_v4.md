### BILL: Physics Loss v4 — L_phase peak angular velocity enforcement

Proposed by: Justice direction (Option C, 2026-04-05)
Date drafted: 2026-04-05
Change type: software (physics_loss.py — l_phase() amplitude enforcement)

Mandated by: v7 diagnostic — L_phase soft hinge satisfied by flat gyr_y ≈ 0

---

**Problem statement:**
The current L_phase implementation enforces only the *sign* of gyr_y during stance
via a soft hinge loss: `max(0, mean_gy_stance)²`. A flat gyr_y ≈ 0 satisfies this
trivially (mean ≈ 0 < 0 passes the hinge). Even with λ_phase = 500, the phase loss
converges to ~0.0005 within 30 epochs while gyr_y remains flat at ±0.2 dps — far
from the expected ±185 dps swing peak. The step detector requires zero-crossings at
the stance/swing boundary at ±30 dps threshold.

**Physical justification:**
peak_angvel_dps is a first-order quantity derived from cadence_spm and step_length_m
(Article I — cadence × step_length ∝ angular velocity of the shank during swing).
It is already present in every WalkerProfile and anchor profile. The PINN must produce
a gyr_y signal whose peak magnitude matches this primitive — otherwise it is generating
a signal that is physically inconsistent with the walking pattern it was conditioned on.

---

**Proposed change — l_phase() in physics_loss.py:**

Add two amplitude enforcement terms alongside the existing stance hinge:

1. **Swing peak enforcement:**
   During swing phase (t > stance_frac), gyr_y should reach peak_angvel_dps.
   Loss: `(max(|gy_pred[swing]|) - peak_angvel_dps)²`
   Normalised by peak_angvel_dps² to make dimensionless.

2. **Stance trough enforcement:**
   During stance phase (t < stance_frac), gyr_y should reach a negative trough
   of approximately -(peak_angvel_dps × stance_depth_frac) where stance_depth_frac
   ≈ 0.35 (physiological constant — ankle dorsiflexion reaches ~35% of swing peak).
   Loss: `(min(gy_pred[stance]) + peak_angvel_dps × 0.35)²`
   Normalised by (peak_angvel_dps × 0.35)².

Both terms are normalised so their scale is dimensionless (ratio, not raw dps²).
This makes them comparable to the existing stance hinge and boundary loss terms.

New l_phase signature adds `peak_angvel_dps` parameter (traces to cadence_spm
and step_length_m via walker_model derivation — Article I satisfied).

```python
def l_phase(
    self,
    gy_pred:          torch.Tensor,  # (N,) predicted gyr_y [dps]
    t:                torch.Tensor,  # (N,) normalised time [0,1]
    stance_frac:      torch.Tensor,  # (N,) [dimensionless]
    peak_angvel_dps:  torch.Tensor,  # (N,) expected swing peak [dps]
) -> torch.Tensor:
```

`peak_angvel_dps` is column index 9 in the X conditioning vector... no — it is not
currently in the X vector. It is derived from cadence_spm and step_length_m which
ARE in X (indices 0 and 1). The training loop computes it from X_batch at call time:

    peak_angvel = cadence_spm / 60.0 * step_length_m * 2 * π * (shank_length ≈ 0.42m)

However: the anchor_profiles.json already contains `peak_angvel_dps` as a stored
derived quantity. For the training loop, pass it from X_batch columns 0 and 1:

    peak_angvel_dps ≈ cadence_spm / 60 * step_length_m / 0.42 * (180/π)
    (shank_length 0.42m is a physiological constant — Amendment 15 documented)

Simpler: read `peak_angvel_dps` directly from a new X column (index 9 replaces
terrain int, terrain moves to index... no — do not change the X vector).

**Decision: compute peak_angvel_dps inside l_phase from cadence_spm and
step_length_m passed from train_pinn.py.** No X vector change. No architecture
change. The physics loss already receives cadence_spm and step_length_m.

---

**Implementation:**

In `physics_loss.py` `total_loss()`, pass cadence_spm and step_length_m to l_phase:

```python
loss_phase = self.l_phase(
    gy_pred, t, stance_frac,
    cadence_spm=cadence_spm,
    step_length_m=step_length_m,
)
```

In `l_phase()`:

```python
SHANK_LENGTH_M = 0.42   # physiological constant, Amendment 15
DEG_PER_RAD = 180.0 / math.pi
STANCE_DEPTH_FRAC = 0.35  # ankle dorsiflexion reaches ~35% of swing peak

peak_angvel = (cadence_spm / 60.0) * step_length_m / SHANK_LENGTH_M * DEG_PER_RAD
# (B*T,) — expected swing peak in dps, traces to cadence_spm + step_length_m

# Swing peak enforcement
swing_mask = (t >= stance_frac).float()
n_swing = swing_mask.sum() + 1e-6
gy_swing_max = (gy_pred.abs() * swing_mask).max()
swing_peak_loss = ((gy_swing_max - peak_angvel.mean()) / (peak_angvel.mean() + 1e-6)) ** 2

# Stance trough enforcement
gy_stance_vals = gy_pred * stance_mask
# min over stance — use negative clamp to find trough
gy_stance_min = (gy_stance_vals - (1 - stance_mask) * 1e6).min()
trough_target = -(peak_angvel.mean() * STANCE_DEPTH_FRAC)
stance_trough_loss = ((gy_stance_min - trough_target) / (peak_angvel.mean() * STANCE_DEPTH_FRAC + 1e-6)) ** 2

# Existing: stance sign hinge + boundary loss (unchanged)
...

return stance_violation + 0.5 * boundary_loss + swing_peak_loss + stance_trough_loss
```

---

**Article/Amendment grounding:**
- Article I: peak_angvel_dps computed from cadence_spm (primitive 2) and
  step_length_m (primitive 3) via shank_length physiological constant.
  Full derivation chain preserved — no raw IMU value used without primitive trace.
- Amendment 15: SHANK_LENGTH_M = 0.42m and STANCE_DEPTH_FRAC = 0.35 are
  physiological constants documented under the statistical derivation standard.
- bill_physics_loss_v3: DC correction and double-integration path unchanged.

**Expected outcome:**
- gyr_y output reaches ≥ ±30 dps zero-crossings (step detector threshold)
- gyr_y swing peak approaches peak_angvel_dps (185 dps flat, 182 dps slope)
- Step count within ±15% of 100 on all 4 anchor profiles
- SI_stance < 3% on all 4 anchor profiles (si_true=0)

**Branch:** constitution-style-management
