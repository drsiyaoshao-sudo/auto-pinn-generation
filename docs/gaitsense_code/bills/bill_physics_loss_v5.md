### BILL: Physics Loss v5 — Revert L_phase to sign hinge + boundary only

Proposed by: Justice direction (Option A, 2026-04-05)
Date drafted: 2026-04-05
Change type: software (physics_loss.py — remove l_phase amplitude enforcement)

Supersedes: bill_physics_loss_v4 (swing peak + stance trough amplitude enforcement)

---

**Problem statement:**
bill_physics_loss_v4 added amplitude enforcement to L_phase to fix gyr_y ≈ 0 dps in PINN
output. That flat signal was caused by the stationary prefix data bug (generate_training_data.py
captured samples 0–207, all near-zero, instead of the walking signal). The data bug is now
fixed — training data has gyr_y range −107 to +304 dps, covering the full walking signal.

v10 training run diagnostic:
- L_phase exploded to 304 at epoch 200 (data_w=1.0), causing total loss 725 (worse than
  random-init baseline of 521)
- Root cause 1: wrong peak_angvel formula in physics_loss.py uses pendulum model:
    (cadence/60) * step_length / 0.42 * (180/π) → 44.6 dps for stairs
  but walker_model.py uses empirical fit:
    (100 + 65 * walking_speed_ms) * slope_factor * stairs_mult → 181.85 dps for stairs
  4× error on stairs profile — creates irreconcilable conflict with data
- Root cause 2: empirical formula requires slope_deg + terrain fields, which are not
  passed to physics_loss.total_loss() — cannot be fixed without API change
- Root cause 3: physics-only warmup (100 epochs) steered model away from data manifold;
  data loss at epoch 200 = 725 > 521 (random init) — model regressed

The amplitude enforcement terms were solving a problem that no longer exists.

---

**Proposed change — l_phase() in physics_loss.py:**

Remove swing peak enforcement and stance trough enforcement (added by bill_physics_loss_v4).
Restore to sign hinge + boundary zero-crossing only.

New l_phase signature (removes cadence_spm and step_length_m parameters):

```python
def l_phase(
    self,
    gy_pred:     torch.Tensor,  # (N,) predicted gyr_y [dps]
    t:           torch.Tensor,  # (N,) normalised time [0,1]
    stance_frac: torch.Tensor,  # (N,) [dimensionless]
) -> torch.Tensor:
```

Remove from total_loss() call:
    cadence_spm=cadence_spm and step_length_m=step_length_m arguments to l_phase.

---

**Article/Amendment grounding:**
- Article I: sign hinge and boundary crossing both trace to cadence_spm (via stance_frac
  which encodes the proportion of step_period in stance). Physical basis preserved.
- Article II: amplitude enforcement was removed by human Justice ruling after diagnostic
  showed phase loss explosion caused by formula error + unavailable terrain primitives.
- bill_physics_loss_v3: DC correction and double-integration path unchanged.

**Why amplitude comes from data, not physics:**
With correct walking data (gyr_y −107 to +304 dps), the MSE data loss enforces that PINN
output matches the walker_model amplitude directly. The physics loss on gyr_y is responsible
only for the structural constraint (sign during stance, zero-crossing at lift-off). This is
consistent with Article I: the sign and timing of the gyr_y zero-crossing trace to
cadence_spm (stance_frac derived from cadence). The absolute amplitude is a projection of
walking speed onto the shank frame — learned most accurately from data directly.

**Expected outcome:**
- L_phase ≈ 1.0–2.0 throughout training (not exploding)
- gyr_y output amplitude learned from data → reaches ±30+ dps zero-crossings
- Data loss converges below random-init baseline (521) after ramp
- Step count within ±15% on all 4 anchor profiles
- SI_stance < 3% on all 4 anchor profiles (si_true=0)

**Branch:** hybrid-model
