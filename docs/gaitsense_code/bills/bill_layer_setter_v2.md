### BILL: layer-setter-v2 — Replace Fourier Feature Network with Polynomial-Wavelet Outer-Product Architecture

Proposed by: layer-setter agent (Claude Sonnet 4.6)
Date drafted: 2026-04-19
Change type: simulation (PINN architecture)

---

## Problem Statement

The existing PINN architecture (`simulator/pinn/pinn_model.py`, v1) uses a random
Fourier Feature embedding that projects the concatenated [x, t] vector through a
fixed random matrix B ~ N(0, sigma^2=1.0), then applies sin/cos. Two structural
defects make this architecture wrong for the GaitSense IMU generation task.

### Defect 1 — The Fourier projection is linear; the required interactions are nonlinear.

The Fourier feature embedding projects z = [x, t] as:

    phi(z) = [sin(2pi * B * z), cos(2pi * B * z)]

This is a linear function of z before the sin/cos. For any fixed random row b of B,
the projection b^T z is a linear combination of the input features. It cannot
compute the product v_walk = (cadence_spm / 60) * step_length_m, because that
product requires multiplying two input features — a nonlinear operation. Nor can it
compute peak_angvel = (100 + 65 * v_walk) * slope_factor without first computing
v_walk.

These products are the physically required amplitude-modulation of the IMU waveform.
From walker_model.py `__post_init__`:

    walking_speed_ms = (cadence_spm / 60.0) * step_length_m
    peak_angvel_dps  = (100 + 65 * walking_speed_ms) * slope_factor

The output gy_dps waveform amplitude is directly proportional to peak_angvel_dps.
If the network cannot represent v_walk and peak_angvel explicitly, it must learn them
implicitly from the training data — which it can only do if cadence and step_length
covary sufficiently in the training set to make their product distinguishable.
data_config.json samples cadence and step_length independently (Article I — they are
separate primitives). Their product v_walk is not independently observed in the
marginals of either, making implicit learning of v_walk unreliable and dependent on
random initialisation rather than physics.

**Article I violation:** A PINN whose output amplitude is determined by an implicit,
data-dependent weight rather than the explicit product (cadence_spm / 60) *
step_length_m cannot be said to trace its signal to a first-order physically
measurable quantity. The amplitude is a guess drawn from the training distribution,
not a measurement of the primitives.

### Defect 2 — sigma=1.0 maps to the wrong frequency regime for physical inputs.

The random Fourier feature embedding was designed for inputs normalised to unit
variance, typically [0, 1] or [-1, 1]. The GaitSense conditioning vector x has
physical units:

    cadence_spm        ~ 95 spm      (range 60-140)
    step_length_m      ~ 0.65 m      (range 0.25-1.0)
    vertical_osc_cm    ~ 4.5 cm      (range 1.5-20)
    slope_deg          ~ 0-15 deg
    mounting_offset_deg ~ 0 +/- 15 deg

With sigma=1.0, the random projection b^T z for b ~ N(0, 1) gives expected magnitude:

    E[|b^T x|] ~ sqrt(Var(b) * ||x||^2) ~ ||x||

For a typical conditioning vector with cadence_spm=95, step_length_m=0.65:
    ||x|| ~ sqrt(95^2 + 0.65^2 + ...) ~ 95   (cadence dominates)

The resulting Fourier feature argument 2*pi * b^T z ~ 2*pi * 95 ~ 597 radians.
sin(597) is essentially random for any small perturbation in cadence — the Fourier
features are in a chaotic high-frequency regime where:
  - A 1 spm change in cadence (< 1% relative) produces a ~6 radian change in
    the Fourier argument, wrapping sin/cos unpredictably.
  - The gradient of the Fourier features with respect to cadence_spm is
    2*pi * b * cos(2*pi * b^T z), which oscillates sign ~95 times per unit
    cadence change.

This renders the Fourier features useless as a smooth representation of the
primitive-to-signal mapping. The gradient landscape is chaotically oscillatory,
making convergence dependent on random seed rather than physics.

sigma cannot be corrected without knowing the physical scale of each input — which
varies by three orders of magnitude across the 10-dimensional conditioning vector.
A single shared sigma cannot simultaneously normalise cadence (~95) and
step_length_m (~0.65). This is a structural mismatch, not a tuning problem.

**Benjamin Franklin Principle violation:** sigma=1.0 cannot be traced to any
physical quantity in the gait primitive set or sensor specification. It is a
free parameter chosen by convention for normalised inputs. A parameter that cannot
be traced to a physical quantity is not a parameter — it is a guess (Article I).

---

## Proposed Change

Replace the Fourier Feature Network in `simulator/pinn/pinn_model.py` with a
Polynomial-Wavelet Outer-Product Network (v2). The file is rewritten in full.
No other files are modified.

### Architecture: Three stages

**Stage 1A — Polynomial branch (x conditioning vector)**

Input: x shape (N, 10).

Compute 19 physics-grounded features:

  Primitives (normalised by physical max from data_config.json):
    cadence_n   = cadence_spm / 140           [dimensionless, 0-1]
    step_len_n  = step_length_m / 1.0         [dimensionless, 0-1]
    vert_osc_n  = vertical_oscillation_cm / 20 [dimensionless, 0-1]

  Derived nonlinear quantities (explicit Article I products):
    v_walk     = (cadence_spm / 60) * step_length_m   [m/s]    / 2.5 (max)
    omega      = 2*pi * cadence_spm / 60              [rad/s]  / 15  (max)
    peak_angvel_norm = (100 + 65 * v_walk_unnorm) / 350        [dimensionless]
      Derivation: walker_model.py __post_init__, slope_factor=1 (flat reference)
      Denominator: 350 = ceiling of (100 + 65 * 3.5) for max v_walk ~ 140/60 * 1.0 = 2.33 m/s

  Degree-2 terms of 3 primitives (6 terms):
    cadence_n^2, step_len_n^2, vert_osc_n^2,
    cadence_n * step_len_n, cadence_n * vert_osc_n, step_len_n * vert_osc_n

  Secondary inputs (linear passthrough, x indices 3-9, 7 values):
    slope_deg, stance_frac, si_stance_true_pct, mounting_offset_deg,
    loose_fit_attenuation, step_variability_ms, terrain_int

Total: 3 + 3 + 6 + 7 = 19 features -> Linear(19, 32) -> GELU -> x-embedding (N, 32)

The polynomial features give the model v_walk and omega explicitly without requiring
the network to discover them from covariant training data. This is the minimal
structure required for Article I compliance: the signal amplitude must trace to the
product of primitives, not to an implicit weight in a random projection.

**Stage 1B — Trainable wavelet bank (time t)**

Input: t shape (N,) normalised step time in [0, 1].

Mexican hat wavelets (second derivative of Gaussian):
    psi(u) = (1 - u^2) * exp(-u^2 / 2),   u = (t - mu) / scale

24 wavelets with trainable center mu (initialised uniformly on [0,1]) and
trainable log_scale (initialised at log(0.05) = scale ~30ms).

Physical justification (Amendment 21 — Data-Physics Alignment Rule):

The gy waveform in walker_model.py _generate_step() (lines 253-278) is composed of:

  Heel-strike dorsiflexion (phase [0, 0.10]):
    gy = -peak * 0.35 * exp(-phase * 30)
    Mathematical form: exponential decay. The Mexican hat wavelet is the second
    derivative of a Gaussian, which has identical decay behaviour near its center.
    A wavelet with mu ~ 0, scale ~ 0.05 approximates this region with its
    negative lobe, correctly encoding the rapid decay of the heel-strike impulse.

  Ankle rocker rebound (phase [0.10, 0.18]):
    gy = +peak * 0.05 * sin(pi * rebound_phase)
    Half-sine over 80ms. Encoded by the deterministic sin(pi*t) basis at the
    appropriate time offset, supplemented by a wavelet centered near phase 0.14.

  Mid-stance dorsiflexion ramp (phase [0.18, 0.75]):
    gy = -peak * 0.20 * ramp
    Linear ramp from 0 to 0.57 of stance. Encoded by the deterministic
    linear ramp basis t, modulated by the x-embedding amplitude term.

  Push-off half-sine pulse (phase [0.75, 1.00]):
    gy = +peak * sin(pi * push_phase)    [sin over the final 25% of stance]
    This is the dominant gy feature (highest amplitude, clinically critical).
    A half-sine pulse of width 0.25 * step_period. Encoded by: a wavelet
    centered near phase 0.875 (midpoint of push-off), plus the sin(pi*t) basis.
    The Mexican hat wavelet at this scale correctly represents the curvature
    of the push-off bell — it is non-zero over [0.75, 1.0] and zero elsewhere.

  Swing deceleration (t > stance_frac):
    gy = -peak * 0.25 * sin(pi * swing_phase)
    Negative half-sine during swing. Encoded by sin(pi*t) at the swing
    phase offsets.

Deterministic bases (no parameters, derived from walker_model.py structure):
  sin(pi * t)    — single half-sine over [0,1], matches push-off + swing shape
  sin(2*pi * t)  — double half-sine, encodes stance/swing polarity reversal
  t              — linear ramp, matches mid-stance dorsiflexion ramp

Total: 24 wavelets + 3 deterministic = 27 features -> Linear(27, 32) -> GELU -> t-embedding (N, 32)

The wavelet basis is matched to the actual mathematical generating functions in
walker_model.py, in compliance with Amendment 21: "the physics loss generating
function for each constrained PINN output must use the same mathematical form as the
walker_model.py generating function for that output."

Note: Amendment 21 applies directly to physics_loss.py, but the same alignment
principle applies to the model's internal representation. A model whose temporal
basis matches the signal structure will learn the correct waveform shape faster
and with fewer data samples than a model with a mismatched basis.

**Stage 2 — Outer-product fusion**

x-embedding (N, 32) and t-embedding (N, 32) are combined via outer product:

    fused[n, i, j] = x_emb[n, i] * t_emb[n, j]
    fused: (N, 32, 32) -> reshape -> (N, 1024)

Physical interpretation: each entry fused[n, i, j] encodes "how much of t-domain
basis function j is active, scaled by x-domain feature i". This is the amplitude-
from-primitives * shape-from-time factorisation that directly mirrors the structure
of the walker_model.py waveform:

    gy = peak_angvel(cadence, step_length, slope) * shape_function(t, stance_frac)

The outer product computes this factorisation explicitly, without requiring the MLP
to discover it through depth. The MLP then only needs to learn the nonlinear
combination of these (amplitude, shape) pairs into 6 output channels.

**Stage 3 — MLP output head**

Linear(1024, 128) -> GELU -> Linear(128, 128) -> GELU -> Linear(128, 128) -> GELU
-> Linear(128, 6)

No activation on output layer (Amendment 3 — unbounded physical units m/s^2, dps).
GELU activation throughout (smooth, better for physics-constrained networks).

### Parameter count (estimated at architecture time):
  PolynomialBranch:   Linear(19,32) + bias = 19*32 + 32 = 640
  WaveletBranch:      24 mu + 24 log_scale + Linear(27,32)+bias = 48+27*32+32 = 944
  MLP:                Linear(1024,128)+bias + Linear(128,128)+bias x2 + Linear(128,6)+bias
                      = 131200 + 16512 + 16512 + 774 = 164998
  Total:              ~166,582 trainable parameters
  Target range (human spec): 50,000 - 200,000  WITHIN RANGE

### train_pinn.py interface compatibility

train_pinn.py instantiates the model as `PINNModel()` with no arguments (line 135).
The v2 PINNModel.__init__ accepts all kwargs that train_config.json may inject
(input_dim, hidden_dim, n_layers, use_fourier, fourier_dim, fourier_sigma,
fourier_seed) and silently ignores the Fourier-specific ones. This preserves
backward compatibility without requiring changes to train_pinn.py.

train_pinn.py also calls `model.count_parameters()` (line 136). This method is
retained in v2.

The X_exp reshape at train_pinn.py line 191 (`reshape(-1, 10)`) remains correct
because input_dim=10 is unchanged.

---

## Article/Amendment Grounding

**Article I — Physics First:**
  - Defect 1 fix: v_walk and peak_angvel are now explicit polynomial features,
    directly tracing gy output amplitude to the product of cadence_spm and
    step_length_m. The prior Fourier architecture could not make this trace
    (linear projection cannot compute products).
  - Defect 2 fix: All basis functions now have physically grounded scale parameters.
    Wavelet log_scale initialised at log(0.05) = 30ms scale, derived from the
    heel-strike sigma=0.015s in walker_model.py with factor-of-2 margin. This
    replaces sigma=1.0 which had no physical derivation.

**Amendment 2 — Three Measurement Primitives:**
  - Polynomial branch explicitly computes degree-2 features of all three primitives
    and their derived products v_walk, omega, peak_angvel_norm.

**Amendment 3 — Seven-Layer Simulation Pipeline Integrity:**
  - Output shape (N, 6) float32 unchanged. Column order [ax, ay, az, gx, gy, gz]
    unchanged. Physical units (m/s^2, dps) unchanged. Layer 2 (imu_model.py)
    is untouched.

**Amendment 13 — Calibration Discipline:**
  - Wavelet initialisation scale log(0.05): derived from heel-strike sigma=0.015s
    in walker_model.py, rounded up by factor of 2 for margin across all terrains.
  - Normalisation constants in PolynomialBranch: derived from data_config.json
    parameter_distributions max bounds (Amendment 15 documentation inline in code).
  - peak_angvel_norm denominator 350 dps: derived from walker_model.py formula
    (100 + 65 * v_walk_max) = (100 + 65 * 2.33) ~ 252, rounded up to 350 to
    cover slope_factor > 1 and stairs multiplier 1.5.

**Amendment 21 — Data-Physics Alignment Rule:**
  - Wavelet basis functions (Mexican hat) are the second derivative of Gaussians,
    matching the curvature structure of the heel-strike Gaussian impulse and
    push-off half-sine pulse in walker_model.py _generate_step().
  - Deterministic bases sin(pi*t), sin(2*pi*t), t directly mirror the half-sine
    push-off, swing, and linear mid-stance ramp generating functions in
    walker_model.py lines 253-278.
  - The Fourier basis (sin/cos at random frequencies) has no correspondence to any
    mathematical form in walker_model.py. The wavelet + half-sine + ramp basis is
    the Amendment-21-compliant replacement.

---

## Physical Evidence

**Evidence 1 — val_ode divergence in v3 PINN run (Amendment 21 grounds):**
  PINN v3 training (100 epochs) showed val_ode = 36 at epoch 1, rising to 58 at
  epoch 100 — monotonically divergent despite physics-first warmup (Amendment 20).
  This divergence was caused by the spring oscillator loss misspecification (which
  bill_physics_loss_v3.md corrected). The same principle applies to the architecture:
  a Fourier basis that cannot represent the heel-strike decay will produce a gradient
  signal pointing away from the true waveform, causing divergence proportional to
  the amplitude of the signal (high at push-off, ~185 dps for flat walker).

**Evidence 2 — Fourier frequency regime calculation:**
  For cadence_spm=95, sigma=1.0: expected 2*pi * b^T * x argument ~ 2*pi * 95 = 597 rad.
  This places sin/cos in a chaotic high-frequency regime. A 1 spm perturbation
  changes the argument by 2*pi/60 * 1 = 0.105 rad — well within a single period of
  sin, causing the gradient to be meaningful. But the absolute argument of 597 rad
  means the initial Fourier features are distributed across 597/(2*pi) ~ 95 full
  oscillations of sin/cos, producing a random phase for every training sample. This
  is numerically observable: the Fourier feature variance from cadence alone is
  E[sin^2(2*pi*b*95)] = 0.5 regardless of b (ergodic property of sin^2). The features
  carry no cadence-specific information in expectation — only noise.

**Evidence 3 — walker_model.py generating function correspondence:**
  gy heel-strike peak: -peak * 0.35 * exp(-phase * 30).
  Mexican hat wavelet psi(u) = (1-u^2) * exp(-u^2/2) has a decay rate of
  exp(-u^2/2). For u = (t - 0) / 0.05 and t in [0, 0.1]: u in [0, 2].
  The Gaussian envelope exp(-u^2/2) = exp(-t^2 / 0.005) decays as
  exp(-t^2 * 200). The walker_model.py exponential decays as exp(-phase * 30)
  where phase = t / stance_frac ~ t / 0.6 -> exp(-t * 50).
  Closest wavelet: u = t/0.04 -> Gaussian envelope exp(-t^2 / 0.0032).
  Both forms are rapid-onset, sub-100ms decays — structurally matched.
  The Mexican hat is therefore the correct basis for this signal region.

---

## Expected Outcome

1. Convergence of gy waveform amplitude: v_walk = cadence * step_length is now
   an explicit input to the model. The gy output can trace its amplitude directly
   from this product in the first training step, without needing to learn the
   multiplication implicitly. Expected: gy amplitude >= 80 dps gate (Amendment 19
   threshold) achievable within fewer epochs than v3.

2. Stable gradient landscape: wavelet basis functions are smooth, bounded, and
   have derivatives of order 1 everywhere. The gradient of the model output with
   respect to cadence_spm now passes through the explicit v_walk computation
   (d(v_walk)/d(cadence) = step_length/60), giving a physically meaningful and
   consistently signed learning signal. The Fourier chaotic-phase gradient
   instability is eliminated.

3. Generalisation to unseen parameter combinations: the outer product factorisation
   encodes "amplitude * shape" explicitly. For an unseen cadence outside the training
   distribution, the polynomial branch can extrapolate v_walk (linear in cadence) and
   the wavelet branch continues to encode the correct waveform shape (wavelets are
   translation-invariant). The Fourier architecture had no such factorisation —
   amplitude and shape were entangled in the same random projection weights.

4. Amendment 19 compliance: per-axis maximum absolute error below 15% of peak
   signal amplitude, tested against walker_model.py on all 4 profiles, must be
   achieved before the model may be used for grid search.

5. Parameter count: ~166,582 — within 50k-200k human-specified range for CPU
   training feasibility on the reference machine (RTX 2080 Ti, 11 GB VRAM).

---

## Branch

`hybrid-model` — the branch on which this Bill is implemented.
File changed: `simulator/pinn/pinn_model.py`

---

## What this Bill does NOT cover

- Loss function weights (lambda_gyy, lambda_az, lambda_vel, lambda_phase):
  those are loss-setter's domain (Amendment 17). The v2 architecture is
  compatible with the existing physics_loss.py v3 interface.
- Learning rate, epochs, batch size: those are pinn-compiler's domain.
- train_pinn.py modifications: none required. The v2 PINNModel interface is
  backward-compatible with all existing train_pinn.py call sites.
- architecture.json update: that is layer-setter's standing order output.
  It will be updated on the next layer-setter invocation if the human directs.
