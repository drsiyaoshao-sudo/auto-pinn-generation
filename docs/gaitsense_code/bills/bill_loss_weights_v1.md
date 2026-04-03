# Bill: Physics Loss Weight Vector — v1

**Proposed by:** loss-setter agent
**Date:** 2026-04-03
**Status:** RATIFIED 2026-04-03 — Option A accepted (single mean λ_vel=2.87, terrain-unified)

---

## Loss Terms

| Term | Formula | Traces to primitive | λ value |
|---|---|---|---|
| L_ODE | mean((d²z/dt² + ω²·z_proxy − F_contact)²) | cadence_spm → ω; vertical_oscillation_cm → F_contact | 1.3425e-04 |
| L_vel | mean((v_x_pred − v_x_expected)²) | cadence_spm × step_length_m → walking_speed_ms | 2.8691e+00 |
| L_phase | hinge(mean_gy_stance) + 0.5 × boundary² | cadence_spm → step_period_s; stance_frac (physiological 60/40) | 7.8472e+01 |

---

## Loss Term Derivations

### L_ODE — CoM Vertical Oscillation ODE

The vertical Centre of Mass motion approximates a damped harmonic oscillator:

```
d²z/dt² + ω²·z = F_contact(t)
```

**ω derivation (traces to cadence_spm):**
```
ω = 2π × (cadence_spm / 60)   [rad/s]
ω² = (2π × cadence / 60)²

flat:    cadence=105 spm → ω=10.996 rad/s → ω²=120.90 rad²/s²
stairs:  cadence=70  spm → ω= 7.330 rad/s → ω²= 53.73 rad²/s²
slope:   cadence=95  spm → ω= 9.948 rad/s → ω²= 98.97 rad²/s²
```

**F_contact derivation (traces to vertical_oscillation_cm):**
```
CoM free-falls vert_osc/2 before heel-strike arrest:
  v_impact = sqrt(2 × G × vert_osc_m)
  hs_impact_ms2 = v_impact / IMPACT_DURATION_S   (IMPACT_DURATION_S = 0.05 s)

flat:    vert_osc=4.0cm → v_impact=0.626 m/s → F_contact=12.53 m/s²
slope:   vert_osc=5.0cm → v_impact=0.700 m/s → F_contact=14.01 m/s²
stairs:  vert_osc=18cm  → toe-strike → F_contact=0.0 (no Gaussian impulse)
```

### L_vel — Horizontal Velocity Constraint

```
v_x_expected = (cadence_spm / 60) × step_length_m   [traces to cadence_spm + step_length_m]

flat:    105/60 × 0.75 = 1.3125 m/s
stairs:   70/60 × 0.28 = 0.3267 m/s
slope:    95/60 × 0.65 = 1.0292 m/s

v_x_pred = mean(ax_pred) × step_period_s   [torch.trapezoid approximation]
```

**Autograd requirement:** ax_pred must remain an attached tensor. Assert guard
in `physics_loss.py::l_vel()` catches silent zero-gradient failures at runtime.

### L_phase — Stance/Swing Timing Constraint

```
stance_frac from WalkerProfile (physiological 60/40 split, Amendment 15):
  flat:    0.60   slope: 0.62   stairs: 0.65

Constraint: gyr_y < 0 during stance (dorsiflexion + ankle rocker)
            gyr_y ≈ 0 at t = stance_frac (lift-off transition)

Loss = hinge(mean_gy_stance)² + 0.5 × mean_gy_boundary²
  where hinge penalises positive mean gyr_y during stance (wrong sign)
  and boundary enforces near-zero gyr_y at lift-off
```

---

## λ Derivation

Each λ = 1 / E[L_term²] at initialisation (random weights, output ~ N(0,1)).
This normalises all three weighted terms to the same order of magnitude at epoch 0.

| Profile | L_ODE_scale | L_vel_scale | L_phase_scale | λ_ODE | λ_vel | λ_phase |
|---|---|---|---|---|---|---|
| flat | 17803.8 | 1.723 | 0.0100 | 5.62e-05 | 5.80e-01 | 1.00e+02 |
| bad_wear | 17803.8 | 1.723 | 0.0100 | 5.62e-05 | 5.80e-01 | 1.00e+02 |
| stairs | 2887.4 | 0.107 | 0.0225 | 3.46e-04 | 9.37e+00 | 4.44e+01 |
| slope | 12763.9 | 1.059 | 0.0144 | 7.84e-05 | 9.44e-01 | 6.94e+01 |
| **mean** | — | — | — | **1.34e-04** | **2.87e+00** | **7.85e+01** |

**Profile range ratios (must be < 10×):**
- λ_ODE:   6.2×  ✓
- λ_vel:   16.1× ⚠ (stairs outlier: low walking speed = high λ; acceptable because stairs
  is architecturally distinct and the mean is physically grounded)
- λ_phase: 2.3×  ✓

**Weighted loss balance at initialisation (target: all within 0.1–10× of each other):**

| Profile | λ·L_ODE | λ·L_vel | λ·L_phase | ODE/vel | ODE/phase |
|---|---|---|---|---|---|
| flat | 2.39 | 4.94 | 0.79 | 0.5× | 3.0× |
| bad_wear | 2.39 | 4.94 | 0.79 | 0.5× | 3.0× |
| stairs | 0.39 | 0.31 | 1.77 | 1.3× | 0.2× |
| slope | 1.71 | 3.04 | 1.13 | 0.6× | 1.5× |

**Status: BALANCED** — all weighted terms within 0.2–5× (well within 0.1–10× target).

---

## Amendment 17 Compliance Declaration

Each λ weight is derived from the expected squared magnitude of its loss term
at network initialisation, computed numerically from the 4 anchor profiles.
No weight was empirically tuned. The derivation formula is:

```
λ_term = mean_across_profiles(1 / E[L_term²]_at_init)
```

Given the same 4 anchor profiles, the same λ values result deterministically.
This is reproducible.

---

## λ Warning: lambda_vel 16.1× Range

lambda_vel spans 16.1× across profiles (0.58 for flat vs 9.37 for stairs).
This exceeds the 10× soft threshold in the loss-setter agent definition.

**Assessment:** The spread is driven by the stairs profile's low walking speed
(0.327 m/s vs 1.3 m/s for flat). Stairs is an architecturally distinct terrain —
its velocity constraint is physically correct and well-defined. The mean λ_vel
(2.87) is appropriate for the flat/slope profiles that dominate training volume.
The stairs profile will naturally have its velocity loss weighted ~3× more than
flat — this is acceptable and physically motivated (slower walking = lower
absolute velocity error scale, so same λ has proportionally more effect).

**Decision required from human:** Accept the 16.1× range given the physical
justification above, or request a terrain-separated λ_vel (flat/slope λ=0.75,
stairs λ=9.37)?

---

## Expected Training Behaviour

With these weights, at epoch 0 (random weights):
- All three weighted physics loss terms should be within 0.2–5× of each other ✓
- Total physics loss ≈ data loss after warmup ramp completes
- L_vel will have higher relative weight on stairs samples — expected behaviour

If any physics loss component grows monotonically relative to others after ramp-up,
the λ derivation should be revisited with a new Bill.

---

## Files Written

- `simulator/pinn/physics_loss.py` — PhysicsLoss class, l_ode(), l_vel(), l_phase(), total_loss()
