### BILL: PINN Training Dataset Configuration v1
Proposed by: synthetic-data-setter (Claude Sonnet 4.6)
Date drafted: 2026-04-03
Change type: simulation

**Status: RATIFIED 2026-04-03**

---

## Problem Statement

The PINN architecture (bill_train_config_v1, architecture.json) is committed and the training loop is ready to run. Before any data generation begins, the full parameter space that will be sampled must be ratified as a Bill. Without this ratification:

- The PINN would train on a parameter distribution that no human has approved (Article II violation)
- The distributions for the 10 input fields (architecture.json `input_fields`) would have no traceable grounding to the three Article I primitives (Article I violation)
- The train/val/test split would be undefined, making the test set unavailable for post-training confirmation (Amendment 18 pre-condition)

This Bill defines all dataset configuration parameters — profile count, sample count, split fractions, terrain distribution, and per-field parameter distributions — before a single sample is generated.

---

## Article/Amendment Grounding

- **Article I — Physics First:** Every parameter distribution in this Bill traces to one of the three walking primitives (vertical_oscillation_cm, cadence_spm, step_length_m) or to a physical terrain geometry quantity derivable from them. The traceability chain is stated explicitly for each field in Section 2 below.
- **Article II — Learner-in-the-Loop:** This Bill is the required human decision gate before data generation begins. The synthetic-data-generator agent may not write training_data/ until this Bill is ratified.
- **Amendment 2 — Three Measurement Primitives:** The three WalkerProfile primary fields (`cadence_spm`, `step_length_m`, `vertical_oscillation_cm`) are the first three input fields in architecture.json and have their own distributions defined here. All other fields are derived or correlated quantities.
- **Amendment 15 — Statistical Derivation Documentation:** Every distribution in this Bill that is derived from a population statistic rather than an algebraic derivation is documented using the required format: distribution (μ, σ), sigma bound applied, excluded population.
- **Amendment 17 — PINN Physics Loss Weight Derivation Lock:** This Bill governs the data domain only, not the loss weights (that is bill_loss_weights_v1). However, Amendment 17 compliance requires that the data domain be fixed before training begins, because the physics loss weight derivation (bill_loss_weights_v1) assumes specific signal amplitude ranges that are only valid within the parameter bounds defined here.

---

## Section 1: Dataset Design

| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_random_profiles | 500 | Sufficient to cover the 10-dimensional parameter space at ~2.5 samples per dimension per side; consistent with surrogate model literature for 8-12 input dimensions |
| n_anchor_profiles | 4 | The four named WalkerProfile entries from walker_model.py PROFILES dict: flat, bad_wear, stairs, slope |
| n_total_profiles | 504 | 500 + 4; anchor profiles are held out from the random split and appear only in the validation set as fixed reference points |
| samples_per_profile | 208 | One step at 208 Hz (ODR_HZ constant from walker_model.py). Each sample is one IMU reading; 208 samples = exactly 1 second = 1 full step cycle at 60 spm to ~1/3 step at 140 spm. This is the minimal complete gait event unit |
| train_fraction | 0.70 | 70% of 500 random profiles = 350 training profiles × 208 samples = 72,800 training samples |
| val_fraction | 0.15 | From train_config.json val_fraction=0.15. 75 random profiles + 4 anchor profiles = 79 validation profiles |
| test_fraction | 0.15 | 75 random profiles withheld from training and validation. Test set owned by this Bill; released only at Amendment 18 boundary confirmation stage |
| seed | 42 | Consistent with train_config.json seed=42 and architecture.json fourier_seed=42. All random state is reproducible from a single seed |

**Total sample counts (random profiles only):**
- Training:   350 profiles × 208 samples = 72,800 samples
- Validation:  75 profiles × 208 samples = 15,600 samples (+4 anchor profiles × 208 = 832)
- Test:         75 profiles × 208 samples = 15,600 samples

---

## Section 2: Parameter Distributions

All 10 fields from `architecture.json input_fields` are defined below. Each field states its Article I primitive traceability and its Amendment 15 population documentation where a statistical derivation is used.

### Field 1: `cadence_spm`
**Distribution:** normal(mu=95, sigma=15, min=60, max=140)
**Article I primitive:** Cadence — the fundamental temporal frequency of gait (first-order primitive)
**Amendment 15 documentation:**
Community ambulation cadence ~ N(95, 15^2) spm.
Population: healthy community-dwelling adults (not running).
2.33-sigma lower tail: 95 - 2.33*15 = 60 spm — very slow, frailty boundary.
3.0-sigma upper tail: 95 + 3.0*15 = 140 spm — brisk walk.
Excluded population: running (>140 spm); out of scope for SI measurement device.

Reference: walker_model.py PROFILES cadence range 70 (stairs) to 105 (flat/bad_wear). mu=95 is the mean of {70, 95, 105, 105} = 93.75, rounded to 95.

### Field 2: `step_length_m`
**Distribution:** normal(mu=0.65, sigma=0.12, min=0.25, max=1.0)
**Article I primitive:** Step Length — the spatial extent of each step (first-order primitive)
**Amendment 15 documentation:**
Adult step length ~ N(0.65, 0.12^2) m.
Population: healthy community-dwelling adults.
Lower tail 0.25 m: very slow or frail gait (stair tread depth is 0.28 m in PROFILES['stairs']).
Upper tail 1.0 m: tall athletic walkers.
Excluded population: running stride (>1.0 m).

Reference: walker_model.py PROFILES step_length range: 0.28 m (stairs) to 0.75 m (flat/slope). mu=0.65 weighted toward level-ground norm; stairs is an outlier that will be captured by the terrain-conditional vertical_oscillation distribution.

### Field 3: `vertical_oscillation_cm`
**Distribution:** terrain_conditional (CRITICAL — stairs is physically distinct)
**Article I primitive:** Vertical Oscillation — amplitude of CoM vertical movement per step (first-order primitive)

**flat branch:** normal(mu=4.5, sigma=1.5, min=1.5, max=8.0)
Amendment 15 doc: Flat CoM oscillation ~ N(4.5, 1.5^2) cm. Population: healthy adults on level ground. Lower tail 1.5 cm: stiff or pathological gait (Parkinson's). Upper tail 8.0 cm: exaggerated bouncy gait. Reference: PROFILES['flat'] = 4.0 cm, PROFILES['bad_wear'] = 4.0 cm; mu=4.5 is 1/3-sigma above to include the 5.0 cm slope reference.

**slope branch:** normal(mu=4.5, sigma=1.5, min=1.5, max=8.0)
Amendment 15 doc: Identical to flat. Physical basis: on moderate slopes (3-15 deg) the per-step CoM vertical oscillation is driven by step length and cadence, not slope grade. The slope's effect on the IMU signal appears in acc_x_dc = G*sin(slope_rad), which is captured by slope_deg, not vertical_oscillation_cm.

**stairs branch:** normal(mu=15.0, sigma=3.0, min=10.0, max=20.0)
Amendment 15 doc: Stair CoM oscillation ~ N(15.0, 3.0^2) cm. Physical basis: standard UK/US residential riser height 17-22 cm; CoM must clear this height per step. mu=15.0 cm derived from mean riser 18 cm minus partial oscillation recovery (~3 cm). Lower tail 10.0 cm: low-rise commercial steps. Upper tail 20.0 cm: high-rise industrial stairs. Reference: PROFILES['stairs'] = 18.0 cm, consistent with 1-sigma above mu.

**Physical consequence of this split:** the vertical_oscillation_cm distribution is the single most important terrain discriminator in the training data. A flat/slope sample with vertical_oscillation_cm > 8.0 cm does not exist in the training set, so the PINN cannot confuse a large-oscillation event with a flat-terrain event. This is an Article I enforcement constraint, not just a statistical choice.

### Field 4: `slope_deg`
**Distribution:** terrain_conditional
**Article I primitive:** Vertical Oscillation (via terrain geometry: acc_x_dc = G*sin(slope_rad))

**flat branch:** fixed=0.0 — flat terrain is by definition zero slope
**slope branch:** uniform(min=3.0, max=15.0)
Amendment 15 doc: Slope angle uniform(3, 15) deg. Lower bound 3 deg: perceptible grade (ADA ramp minimum 1:20 = 2.86 deg, rounded to 3). Upper bound 15 deg: near limit of comfortable ambulation without assistive devices. Uniform chosen: no preferred slope value within physiological range; all grades in [3, 15] are equally likely in community environments. Traces to: vertical_oscillation_cm via acc_x_dc = G*sin(slope_rad) in walker_model.py.
**stairs branch:** fixed=0.0 — stair ascent IMU tilt is captured by the stair vertical_oscillation model, not a continuous incline angle (see walker_model.py `__post_init__` stair branch)

### Field 5: `stance_frac`
**Distribution:** terrain_conditional (fixed per terrain)
**Article I primitive:** Cadence (via stance_dur_s = stance_frac * step_period_s = stance_frac * 60/cadence_spm)

| Terrain | Value | Physical basis |
|---------|-------|---------------|
| flat | 0.60 | Healthy adult consensus: 58-62% stance fraction on level ground; fixed at midpoint to prevent covariance with cadence |
| slope | 0.62 | Uphill ambulation increases stance fraction 2-4 percentage points; 0.62 matches PROFILES['slope'] |
| stairs | 0.65 | Stair ascent requires prolonged single-limb support for riser clearance; 65% measured in instrumented stair studies; matches PROFILES['stairs'] |

Amendment 15 doc: Stance fraction is fixed per terrain (not sampled) to prevent it from becoming a free parameter that confounds the cadence-based timing signals the PINN must learn. A PINN trained with stance_frac varying independently of terrain would need to infer terrain from the joint (cadence, stance_frac, vertical_oscillation) tuple — introducing an unnecessary learning burden. Fixing stance_frac per terrain makes terrain-inference unambiguous.

### Field 6: `si_stance_true_pct`
**Distribution:** normal(mu=5.0, sigma=8.0, min=0.0, max=50.0), clipped at 0
**Article I primitive:** Cadence (via delta_s = SI * stance_nom_s / 200 in walker_model.py)
**Amendment 15 documentation:**
SI stance asymmetry ~ N(5.0, 8.0^2) pct, clipped at 0 (unsigned asymmetry).
Population: community-dwelling adults including sub-clinical asymmetry.
mu=5.0 pct: general adult population shows mean mild asymmetry below the Robinson 1987 10% clinical threshold; mu=5 captures this sub-clinical range.
Upper tail 50 pct: moderate clinical pathology; PROFILES['high_si'] at 25% is within the sampling range.
Clip at 0: SI is a magnitude; negative values have no physical meaning.
Excluded: severe pathology (>50 pct SI); those patients require clinical gait lab, not wearable device.

### Field 7: `mounting_offset_deg`
**Distribution:** normal(mu=0.0, sigma=5.0, min=-15.0, max=15.0)
**Article I primitive:** Vertical Oscillation (via sensor frame projection: rotation_y matrix in walker_model.py)
**Amendment 15 documentation:**
Mounting offset ~ N(0, 5^2) deg.
Population: real-world device self-application by patients.
mu=0.0: ideal application; sigma=5 deg is expected 1-sigma variation.
Upper/lower bound +/-15 deg: chosen at the boundary where acc_x/acc_z cross-projection begins to corrupt the HS impulse peak below the step detection gate threshold. Reference: PROFILES['bad_wear'] uses 20 deg, which is deliberately outside the training distribution to test the PINN's response to out-of-distribution inputs.
Excluded: extreme misapplication (>15 deg); this is a device failure mode, not a patient population characteristic.

### Field 8: `loose_fit_attenuation`
**Distribution:** uniform(min=0.75, max=1.0)
**Article I primitive:** Vertical Oscillation (via HS impact attenuation: effective_impact = hs_impact_ms2 * loose_fit_attenuation)
**Amendment 15 documentation:**
Loose fit attenuation ~ uniform(0.75, 1.0). 1.0 = perfect coupling; 0.75 = 25% attenuation.
Lower bound physical basis: at attenuation below 0.75, the HS impulse for the flat reference walker (hs_impact_ms2 ≈ 8.9 m/s^2 * 0.75 = 6.7 m/s^2) approaches the 5.0 m/s^2 adaptive step detection threshold, causing step detection failure. This is a device boundary, not a physiological bound.
Upper bound 1.0: perfect sensor coupling.
Uniform chosen: no preferred fit quality within the valid range; a loose strap can attenuate by any amount between 0% and 25%.
Reference: PROFILES['bad_wear'] uses 0.55 (45% attenuation), which is deliberately below the training lower bound to test the PINN on out-of-distribution fitting conditions.

### Field 9: `step_variability_ms`
**Distribution:** normal(mu=20.0, sigma=10.0, min=0.0, max=60.0)
**Article I primitive:** Cadence (temporal variance of step period)
**Amendment 15 documentation:**
Step timing variability ~ N(20, 10^2) ms.
Population: healthy adults across terrains.
mu=20 ms: coefficient of variation 2-5% of step period at 100 spm (step_period=600 ms) maps to 12-30 ms; mu=20 ms is a conservative midpoint.
Upper tail 60 ms: elevated variability in healthy elderly or on challenging terrain; PROFILES['stairs'] uses 25 ms and PROFILES['slope'] uses 18 ms, both within the distribution body.
Traces to: cadence_spm via step_period_s = 60/cadence_spm in walker_model.py _generate_step().

### Field 10: `terrain_int`
**Distribution:** categorical {flat: 0.60, slope: 0.20, stairs: 0.20}
**Article I primitive:** All three primitives (terrain determines the conditional distributions for all other fields)
*See Section 3 for full terrain distribution justification.*

---

## Section 3: Terrain Distribution Justification

**60% flat / 20% slope / 20% stairs**

Physical justification for each fraction:

**Flat (60%):** Community ambulation studies (Tudor-Locke, Bohannon) consistently report that >60% of daily steps occur on level surfaces. The PINN must learn the flat reference condition well because the gait algorithm's primary calibration baseline is flat-ground walking. Underweighting flat would cause the PINN to learn the boundary between flat and non-flat terrain less precisely, which is precisely the boundary that Amendment 18 requires it to find.

**Slope (20%):** Community ambulation includes ramps, hills, and uneven outdoor surfaces. ADA-compliant ramps (up to 8.33% grade, ~4.8 deg) are ubiquitous. The 20% weighting ensures the PINN encounters sufficient slope samples to learn the acc_x_dc = G*sin(slope_deg) projection effect that the gait algorithm misinterprets as horizontal acceleration.

**Stairs (20%):** Stair ascent/descent is the most physically discriminative terrain for the gait algorithm because vertical_oscillation_cm shifts by >10 cm (from ~4.5 cm to ~15 cm), fundamentally changing every derived signal parameter (hs_impact_ms2, peak_angvel_dps, stance pattern). Equal weighting with slope ensures the PINN does not underfit stairs. A 10% stairs fraction was considered and rejected: with 500 random profiles, 10% = 50 stair profiles. The vertical_oscillation_cm distribution for stairs has sigma=3 cm over a 10 cm range — 50 profiles would undersample the tails. 20% = 100 stair profiles adequately covers the distribution.

**Terrain-conditional distributions:** Fields 3 (vertical_oscillation_cm), 4 (slope_deg), and 5 (stance_frac) have terrain-conditional distributions. This is an Article I requirement: the distribution of a physical quantity must reflect the physical constraint of the terrain. Sampling vertical_oscillation_cm from a single flat-ground distribution and then assigning a "stairs" label would violate Article I by producing a physically impossible parameter combination.

---

## Section 4: Anchor Profile Role

The four named profiles from walker_model.py PROFILES dict serve as fixed validation references:

| Anchor | Key physical characteristics | Validation role |
|--------|-----------------------------|-----------------| 
| flat | cadence=105, step_length=0.75, vert_osc=4.0, slope=0, si=0% | Baseline: PINN must reproduce the reference walker to within Amendment 19 fidelity threshold (15% peak error) |
| bad_wear | Same as flat but mounting_offset=20 deg, attenuation=0.55 | Tests PINN on out-of-distribution signal quality; both parameters are outside the random sampling bounds |
| stairs | cadence=70, step_length=0.28, vert_osc=18.0, stance_frac=0.65 | Tests the extreme of the vertical_oscillation distribution; vert_osc=18.0 is at the 1-sigma upper tail |
| slope | cadence=95, step_length=0.65, vert_osc=5.0, slope_deg=10 | Tests the slope terrain branch; slope_deg=10 is within the uniform(3,15) training range |

Anchor profiles are generated with seed=42 (matching data_config.json seed) and are held in the validation set. They appear in every validation epoch so that the training curve can be monitored against known physical ground truth. They are never in the training set.

The bad_wear anchor (mounting_offset=20 deg, attenuation=0.55) is deliberately outside the training parameter bounds. This tests whether the PINN has learned the physics of the sensor frame projection (mounting offset) and attenuation effects, or whether it has merely memorised the training distribution. A PINN that fails badly_wear beyond the Amendment 19 threshold is overfitting to the training distribution and must be retrained.

---

## Section 5: Files Written on Ratification

| File | Written by | Status |
|------|-----------|--------|
| `simulator/pinn/data_config.json` | synthetic-data-setter | Written with this Bill; takes effect on ratification |
| `simulator/pinn/training_data/` | synthetic-data-generator | Written only after this Bill is ratified; not yet generated |

data_config.json has been written to disk by the synthetic-data-setter agent as part of this Bill proposal. The file is present but the training_data/ directory is empty. The synthetic-data-generator agent is blocked from generating data until this Bill receives explicit human ratification.

---

## Section 6: Amendment 17 Compliance Declaration

Amendment 17 requires that every parameter entering the PINN training objective traces to the three walking primitives. This Bill governs the data domain, not the loss weights (that is bill_loss_weights_v1), but the same traceability standard applies to the input parameter distributions.

**Primitive traceability map for all 10 input fields:**

| Field | Article I Primitive | Derivation path |
|-------|--------------------|-----------------| 
| cadence_spm | Cadence | Direct primitive |
| step_length_m | Step Length | Direct primitive |
| vertical_oscillation_cm | Vertical Oscillation | Direct primitive |
| slope_deg | Vertical Oscillation | acc_x_dc = G*sin(slope_rad); slope changes CoM vertical path geometry |
| stance_frac | Cadence | stance_dur_s = stance_frac * (60/cadence_spm) |
| si_stance_true_pct | Cadence | delta_s = SI * stance_dur_s / 200 — alternating stance duration perturbation |
| mounting_offset_deg | Vertical Oscillation | rotation_y(offset) rotates the vert_osc signal vector into acc_x |
| loose_fit_attenuation | Vertical Oscillation | effective_impact = hs_impact_ms2 * attenuation; hs_impact derived from vert_osc |
| step_variability_ms | Cadence | Temporal noise on step_period_s = 60/cadence_spm |
| terrain_int | All three | Determines conditional distribution of all three primitives |

Every field traces. Amendment 17 compliance confirmed.

---

## Physical Evidence

The parameter ranges in this Bill are validated against the four existing walker profiles in walker_model.py PROFILES dict:

| Profile | cadence_spm | step_length_m | vert_osc_cm | slope_deg | stance_frac | In training distribution? |
|---------|-------------|---------------|-------------|-----------|-------------|--------------------------|
| flat | 105 | 0.75 | 4.0 | 0.0 | 0.60 | Yes — all values within bounds |
| bad_wear | 105 | 0.75 | 4.0 | 0.0 | 0.60 | Partial — mounting_offset=20 deg and attenuation=0.55 are outside bounds (intentional) |
| stairs | 70 | 0.28 | 18.0 | 0.0 | 0.65 | Yes — vert_osc=18.0 is within stairs N(15,3) distribution |
| slope | 95 | 0.65 | 5.0 | 10.0 | 0.62 | Yes — all values within bounds |

All four anchor profiles can be reproduced from within (or at the edge of) the training distribution, satisfying Amendment 19's requirement that the PINN must reproduce known profiles within the 15% fidelity threshold.

---

## Expected Outcome

On ratification, the synthetic-data-generator will produce:
- 504 WalkerProfile × 208 samples = 104,832 total samples
- 72,800 training samples (350 random profiles)
- 16,432 validation samples (75 random profiles + 4 anchors)
- 15,600 test samples (75 random profiles, withheld)

The PINN trained on this dataset is expected to:
1. Reproduce all 4 anchor profiles within the Amendment 19 fidelity threshold (15% peak error per axis)
2. Interpolate correctly within the training parameter space (verified by test set evaluation)
3. Identify gait algorithm failure boundaries through grid search (Amendment 18 post-condition)

---

## Branch

`main` — data_config.json is a configuration file with no algorithm logic. It is written to main directly on ratification.
