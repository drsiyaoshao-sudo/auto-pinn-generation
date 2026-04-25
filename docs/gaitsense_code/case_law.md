# GaitSense Case Law

All rulings are binding on future agents and humans per the Judicial Process in CLAUDE.md. To deviate from a precedent, a new hearing must be declared — unilateral deviation is not permitted.

*Format for each case:*
```
### [CASE NAME] — [DATE]
Competing Positions: A (Amendment N) vs B (Amendment M)
Physical/Empirical Basis: [the measurement or signal that decided the case]
Ruling: [which position prevailed and the condition under which it applies]
Precedent Effect: [future situations this ruling governs]
Files Changed: [list]
```

---

### The Stair Walker Case — 2026-03-27

**Competing Positions:**
- Position A (Amendment 5): Maintain the dual-confirmation gate; it is correct for all profiles that passed validation.
- Position B (Article I): The dual-confirmation gate embeds a terrain-specific assumption not derived from any walking primitive — it must be replaced.

**Physical/Empirical Basis:**
Signal diagnostic (docs/executive_branch_document/plots/stair_walker_signal_check.png): gyr_y zero-crossing at 53ms, acc_filt peak at 188ms — temporal gap of 135ms on stairs. The 40ms confirmation window was derived from flat-ground heel-strike kinematics where both events are co-incident. This assumption was not derivable from the three walking primitives. Push-off plantar-flexion traces directly to cadence and step_length (push-off angular velocity = f(step_length, cadence)) and is present on every terrain without exception.

**Ruling:**
Position B prevails. The dual-confirmation gate is replaced by the push-off primary detector with retrospective ring-buffer heel-strike inference. Any future step detector primary trigger must be a signal feature that is biomechanically required on all terrains and derivable from at least one walking primitive.

**Precedent Effect:**
Time-gated co-occurrence windows that assume simultaneous signal events are not permitted unless the simultaneity is derived from and bounded by a walking primitive.

**Files Changed:** `src/gait/step_detector.c`, `simulator/terrain_aware_step_detector.py`

---

### The VABS.F32 Case — 2026-03-28

**Competing Positions:**
- Position A (Amendment 5): Healthy walkers passing all Stage 3 exit criteria (SI < 3%) is sufficient; the VABS.F32 discrepancy is a simulator artifact with no clinical consequence.
- Position B (Article I + Amendment 4): A failure mode is only confirmed caught if it is tested under conditions where the correct answer is non-zero. SI correctness for healthy walkers does not constitute a test of the SI computation under asymmetric input.

**Physical/Empirical Basis:**
Pathological walker test: true SI = 25% injected via ±45ms alternating stance offset. Firmware reported SI = 0.0% across all 9 snapshots on all 4 profiles despite 100/100 steps detected. DBG_SNAP diagnostic confirmed n_odd=9, n_even=10, stance_odd=482ms, stance_even=388ms — expected SI ≈ 21%, reported SI = 0.0%. Root cause: `VABS.F32` ARM FPU instruction returns ≈0 instead of |x| in Renode 1.16.1 for computed FPU-register values. `fabsf(m_odd - m_even)` was silently zeroing every SI computation where the result was non-zero.

**Ruling:**
Position B prevails. The pathological walker test (true SI = 25%, all four profiles, all above 10% clinical threshold) is now a mandatory Stage 3 exit criterion. `fabsf()` on FPU-register values is banned in this codebase; the conditional subtraction pattern `(diff >= 0.0f) ? diff : -diff` is the required replacement. Any future function that computes a quantity that could silently return a "correct-looking" zero must be validated under input conditions where the correct answer is non-zero.

**Precedent Effect:**
"No crash" is not a passing criterion. "Correct output for inputs where the correct answer is known to be non-zero" is the required criterion for any clinical-output computation.

**Files Changed:** `src/gait/rolling_window.c`

---

### The CNN Prior Seeding Case — 2026-03-28

**Competing Positions:**
- Position A (Amendment 13): Pre-filling the rolling window with synthetic records is a calibration that permanently biases the window; one calibration per algorithmic iteration must be justified.
- Position B (Article I + Amendment 5): The cold-start artifact (SI_swing = 200% at first snapshot) has a known physical cause (ring buffer ghost step from calibration period); the synthetic prior is derived from the three walking primitives at 105 spm using the physiological 60/40 stance/swing constant, not tuned empirically.

**Physical/Empirical Basis:**
Renode simulation (all 4 profiles): SI_swing = 200% at first snapshot regardless of actual asymmetry. Diagnostic: ring buffer entry from stationary calibration period produced heel-strike timestamp ≈ 4.8ms, yielding stance ≈ 1534ms (3× normal). Synthetic prior derivation: stance=343ms = 60% of 571ms step period at 105 spm. The 60/40 stance/swing split is a measured physiological constant, not a tuned value. Priors are symmetric (identical odd/even) → contribute exactly 0% SI. Priors evict naturally from the 200-entry buffer after 200 real steps.

**Ruling:**
Position B prevails with the constraint of Amendment 13: this prior derivation is the one calibration for this algorithmic iteration. The cadence convergence transient (first 3–4 snapshots show cadence closer to 105 spm than actual) is documented in `docs/executive_branch_document/handoff.md` Section 8 as an expected and predicted deviation.

**Precedent Effect:**
Synthetic priors are permitted as a cold-start mechanism if and only if: (a) prior values are derived from walking primitives using documented physiological constants; (b) priors are symmetric and contribute zero SI; (c) they evict naturally within one full window cycle; (d) the convergence transient is documented in the handoff document.

**Files Changed:** `src/gait/rolling_window.c`, `src/gait/phase_segmenter.c`

---

### The Terrain Gate Case — 2026-03-27

**Competing Positions:**
- Position A (Amendment 3): The LOADING→MID_STANCE gate references `acc_mag` which is also used in the step detector — this may be a cross-layer coupling.
- Position B (Article I): The `acc_mag` gate (`|acc_mag − 9.81| < 2.94`) was derived from flat-ground physics and must be replaced with a terrain-agnostic gate derivable from a walking primitive.

**Physical/Empirical Basis:**
Stair walker stuck permanently in LOADING phase. `acc_mag` at stair mid-stance ≈ 20 m/s² due to heel-strike impact. Gate: |20 − 9.81| = 10.2 >> 2.94 — never fires. Physical measurement: heel-strike arrest decays from 37–60 dps to near-zero in ~100ms on all terrains; early ankle rocker is 10–13 dps. The bisection point of 20 dps is terrain-invariant because it is derived from the gyr_y decay dynamics of foot-floor contact, which is governed by stance mechanics, not surface type.

**Ruling:**
Position B prevails (Article I takes precedence). Gate replaced with `|gyr_y| < 20 dps`. The VSQRT.F32 workaround was also removed because the acc_mag computation that required it is eliminated. Any phase transition gate that references a computed quantity must justify its terrain-invariance; if it cannot, it must be replaced by a raw axis gate derivable from walking primitive mechanics.

**Precedent Effect:**
Computed quantities used in phase transition gates require explicit terrain-invariance justification traceable to a walking primitive. Flat-ground-derived thresholds are not terrain-invariant by default.

**Files Changed:** `src/gait/phase_segmenter.c`

---

### The Dual-Confirmation Architecture Case — 2026-03-29

**Competing Positions:**
- Position A (Amendment 13 + Amendment 5): The 40ms co-occurrence window is a physically derivable calibration constant, not an architectural error. The correct remediation is to derive the window from `cadence_spm` and `stance_frac` as `stance_frac × (60000 / cadence_spm) × 0.12`, yielding a terrain-adaptive constant. The ring buffer fallback in Position C introduces a stance-duration timing error when no acc_filt crossing is found.
- Position C (Article I + Amendment 2 + Stair Walker Case precedent): Push-off plantar-flexion (gyr_y_hp) derives directly from cadence and step length. The 40ms window cannot derive from any walking primitive because it encodes flat-ground heel-strike co-incidence — a terrain-specific biomechanical assumption, not a primitive.

**Physical/Empirical Basis:**
IMU diagnostic plot (`docs/executive_branch_document/plots/stair_vs_flat_imu_diagnostic.png`), generated 2026-03-29 from `diagnostic_imu_analysis.py` against both profiles, 30 steps, seed=42:

- acc_filt peaks on stairs: mean 21.4 m/s² — **57.6% higher than flat** (13.6 m/s²). Peaks are not degraded; both profiles clear the 5.0 m/s² adaptive threshold with margin. Position A's ring buffer fallback risk (from missing acc_filt crossings) is not supported by the signal evidence.
- Timing gap between acc_filt peak and gyr_y zero-crossing: flat = ~19ms (4/5 steps fire the 40ms window); stairs = ~100ms (0/5 steps fire, including one step with no gyr_y zero-crossing detected at all).
- The ~100ms structural gap on stairs is not a constant calibration error — it is caused by the sigmoid toe-roll loading geometry of stair contact, which shifts the acc_filt peak rightward by the full body-weight loading interval relative to flat heel-strike. No single window constant can cover both because the two peaks occupy structurally different positions in the gait cycle.
- The missing gyr_y zero-crossing on stair Step 2 confirms that the confirmation event itself is unreliable on stairs, independent of window width. Even a widened window would not fire when the confirmation event does not occur.

**Ruling:**
Position C prevails. The Justice ruled that Position A poses a fundamental error conflicting with the Benjamin Franklin Principle (Physics First): stair walker biomechanics differ structurally from flat walker (longer toe-to-heel strike interval, near-single push-off loading pattern). The 40ms window failure is not a miscalibrated constant — it is an architectural assumption that encodes flat-ground heel-strike co-incidence and cannot be corrected by derivation without embedding a hidden terrain classifier. This violates Article I. Enacting Position A would also violate the Thomas Jefferson Principle: a patient ascending stairs with genuine gait asymmetry would receive an undefined SI output (0/5 steps fire), not an inaccurate one. Position C (terrain-aware push-off primary) is confirmed as the constitutionally required architecture.

**Precedent Effect:**
Any future proposal to restore a co-occurrence timing window as the primary step detection mechanism must first demonstrate, from physical IMU measurements, that the two co-occurring events occupy the same phase of the gait cycle across all terrain profiles under test. Simulation evidence (acc_filt and gyr_y timing tables) is the minimum required evidence for such a claim. A derivation argument alone, without measured timing data, is insufficient under the Benjamin Franklin Principle.

**Files Changed:** No new implementation — ruling confirms existing `src/gait/step_detector.c` (Option C, terrain-aware) and `simulator/terrain_aware_step_detector.py`. Diagnostic script: `diagnostic_imu_analysis.py`.

---

### The Algorithm Comparison Case — 2026-03-28

**Competing Positions:**
- Position A (Amendment 9): Three algorithm options (A: threshold tuning; B: filter redesign; C: push-off primary with ring buffer) were evaluated; Option C was selected after exhausting A and B. This is a valid domain search under Amendment 9.
- Position B (Amendment 10): Option C adds firmware complexity (ring buffer, extra FSM state); the hardware alternative (shoe-dorsum sensor repositioning for cleaner forefoot-to-flat terrain discrimination) was not formally evaluated as required by Amendment 9.

**Physical/Empirical Basis:**
Options A and B failed on stair profile (0/100 steps). Option C: 100/100 steps, SI = 0.41% on stairs. RAM overhead: 32 bytes for ring buffer (0.03% of 118KB SRAM). Flash overhead: < 200 bytes. Shoe-dorsum mounting assessed: requires different form factor, different strap BOM, different user calibration — cost exceeds 32 bytes of firmware complexity. Additionally, algorithm comparison GUI revealed Option C also resolves poor device fit (bad_wear) SI underestimation in pathological mode — a second failure mode resolved by the same architectural change.

**Ruling:**
Position A prevails. Option C is accepted. However, the shoe-dorsum mounting option is not closed — it is documented in `docs/executive_branch_document/hw_bom.md` as an open hardware iteration item. An agent may not remove this item without a new hearing. BOM alternatives are never silently closed by algorithm success alone.

**Precedent Effect:**
When an algorithm fix is accepted, the hardware alternative that was considered but not selected must be explicitly documented as an open option. Hardware iteration optionality survives algorithm success.

**Files Changed:** `src/gait/step_detector.c`, `simulator/terrain_aware_step_detector.py`

---

### The z_proxy Collapse Case — 2026-04-03

**Competing Positions:**
- Position A (Amendment 14): The algebraic substitution `z_proxy = d2z_dt2 / ω²` is a valid simplification under dimensional analysis and reduces training instability by eliminating a poorly-conditioned double integration path. cadence_spm is still present in ω and the ODE residual retains physical grounding.
- Position B (Article I + Amendment 14): The substitution `z_proxy = d2z_dt2 / ω²` algebraically collapses the ODE residual to `2×(az_pred - G) - F_contact = 0`, eliminating the `ω²·z` restoring term entirely. cadence_spm contributes zero gradient to the training loss. The resulting PINN does not enforce the spring-mass ODE — it enforces a static force balance that is not the governing equation of walking gait.

**Physical/Empirical Basis:**
The algebraic collapse `z_proxy = d2z_dt2 / ω²` substituted into the spring-mass ODE `d2z_dt2 + ω²·z = F_contact / m - G` yields:

```
d2z_dt2 + ω²·(d2z_dt2 / ω²) = F_contact / m - G
d2z_dt2 + d2z_dt2 = F_contact / m - G
2·d2z_dt2 = F_contact / m - G
```

The `ω²·z` restoring term cancels identically. cadence_spm, which enters only through ω, is multiplied by zero and contributes no gradient. Three training runs confirm that `val_ode` never converged: 190 epochs (v1), 500 epochs (v2), 100 epochs (v3 with 768× weight increase). The failure is structural — a zero-gradient path cannot be resolved by scaling weights or increasing epoch count.

**Ruling:**
Position B prevails. `z_proxy = d2z_dt2 / ω²` is a constitutionally impermissible simplification under Article I: cadence_spm (a first-order walking primitive) is rendered inert in the ODE residual, making the PINN training loss independent of the fundamental temporal frequency of gait. The three training runs constitute empirical confirmation under the Benjamin Franklin Principle that the failure is structural, not a weight-scale or epoch-count problem. The Thomas Jefferson Principle also governs: the `z` term in the true spring-mass ODE encodes vertical CoM oscillation frequency, which is the physical basis of step counting peak detection. A PINN enforcing the correct ODE produces `az` signals whose oscillatory structure is cadence-consistent, enabling reliable step count peak detection. A PINN trained on the collapsed residual produces `az` signals driven only by data fitting, whose peaks are data artefacts not physics-grounded — an unacceptable clinical output.

**Precedent Effect:**
No algebraic substitution that eliminates a walking primitive (cadence, vertical oscillation, or step length) from an ODE residual is permitted, regardless of dimensional correctness. A simplification that appears valid under dimensional analysis must be verified to preserve non-zero gradient contribution from all three walking primitives before it may be used in a PINN loss function. The correct implementation path is true double-integration of `(az_pred - G)` over time to obtain `z_pred(t)`, preserving the `ω²·z_pred` restoring term with cadence_spm as an active training signal. A new Bill is required before any implementation of the corrected `l_ode()` begins.

**Files Changed:** No implementation — ruling prohibits the collapsed residual and requires a new Bill before `physics_loss.py` `l_ode()` is modified.

---

### The PINN Data Loss Dominance Case — 2026-04-03

**Competing Positions:**
- Position A (Amendment 14): Data loss dominance is expected and correct — the PINN should fit data first; physics constraints are secondary regularizers that gradually tighten the solution space. High data loss relative to physics loss at epoch 1 indicates the model is learning the training distribution, which is the prerequisite for any generalisation.
- Position B (Article I + Amendment 14): Data loss dominance at epoch 1 indicates physics loss weight miscalibration. A PINN whose training signal is dominated by data fitting from the outset learns a function that interpolates data artefacts, not one that enforces the governing ODE. Physics must actively shape the function from epoch 1 — not regularize a data-fitted function post hoc.

**Physical/Empirical Basis:**
The v3 diagnostic training run (100 epochs, lambda_ode ×768, no physics warmup phase) was executed and its epoch-by-epoch loss table was recorded. `val_ode` declined from 48.66 (epoch 1) to 43.72 (epoch 30) — the first measurable downward trend in physics loss convergence across all three training runs: v1 (190 epochs, baseline lambda_ode) produced structural plateau in `val_ode`; v2 (500 epochs, intermediate lambda_ode increase) produced structural plateau in `val_ode`; v3 at ×768 weight produced the first confirmed physics convergence signal. The 768× weight amplification gave physics loss sufficient gradient magnitude to overcome data loss dominance and produce a measurable ODE residual reduction. This confirmed empirically that physics IS learnable when given gradient priority from epoch 1 — the prior plateau was not a model-capacity failure but a gradient-priority failure.

The Justice's direction, verbatim: "we have to make sure the franklin principle prevails, i.e., the full system runs on physical primitives and pinn can know these primitives. Thus, a quick training session on very low data loss driven (<0.05) should be operated to check whether the model really learns physics before data kicks in."

**Ruling:**
Position B direction prevailed. The Justice ruled that the Benjamin Franklin Principle requires the full system to run on physical primitives, and the PINN must demonstrably learn those primitives before data fitting is permitted to dominate. The v3 empirical result — first-ever measurable physics convergence produced only when physics loss received gradient priority — constitutes the physical basis for the ruling. The Thomas Jefferson Principle also governs: a PINN that learns physics first will extrapolate along the ODE manifold at unseen parameter combinations (out-of-distribution patients in real deployment); a PINN that learns data first extrapolates data artefacts. Real patients occupy parameter points not in the training set — the manifold extrapolation property is the clinical outcome being protected. Physics-first training order is now constitutionally mandated by Amendment 20 (PINN Physics-First Training Order, ratified 2026-04-03). All future PINN training Bills must document a physics-dominant warmup phase (physics loss >= 80% of total loss) with verified downward trend in all three physics terms before the data-fitting phase begins.

**Precedent Effect:**
Any PINN training run that begins with data loss dominant (>50% of total loss at epoch 1) without a preceding physics-dominant warmup phase that satisfies Amendment 20 criterion 2 (verified downward trend in all three physics loss terms during warmup) is constitutionally invalid. A checkpoint produced by such a run is not eligible for use in pinn-validator, grid search, or any downstream clinical inference step. The physics warmup criterion must be documented as a passed exit gate in the training Bill before any checkpoint advances to the next pipeline stage.

**Files Changed:** No implementation — ruling establishes the constitutional training order requirement. Amendment 20 is the operative rule. A new Bill is required before any training script is modified to implement the physics-first warmup phase.

---

### The Polynomial-Wavelet Architecture Enactment — 2026-04-19

**Bill:** bill_layer_setter_v2.md — "Replace Fourier Feature Network with Polynomial-Wavelet Outer-Product Architecture"
**Proposed by:** layer-setter agent
**Ratified by:** Human (Justice), 2026-04-19

**Problem Statement:**
The Fourier Feature Network (v1, sigma=1.0) had two structural defects: (1) the random projection is linear and cannot compute v_walk = cadence_spm/60 × step_length_m — a nonlinear product required by Article I to trace gy amplitude to the primitives; (2) sigma=1.0 with physical inputs (cadence_spm ~ 95) placed Fourier arguments at ~600 radians, producing chaotically phase-distributed features whose gradients oscillate sign unpredictably with respect to cadence. Both defects are structural — not tunable — constituting Article I and Benjamin Franklin Principle violations.

**Physical/Empirical Basis:**
Frequency regime calculation: for cadence_spm=95, sigma=1.0, the Fourier argument 2π × b^T × x ~ 597 rad. sin(597) is effectively random over any 1 spm perturbation in cadence. E[sin²(2π × b × 95)] = 0.5 regardless of b — Fourier features carry no cadence-specific information in expectation. Additionally, val_ode divergence (36→58 over 100 epochs in v3 run) was caused by the same mismatch principle applied to the physics loss (corrected by bill_physics_loss_v3.md / Amendment 21). The same alignment principle governs the model basis: a Fourier basis has no correspondence to any mathematical form in walker_model.py's gy generating function.

**Ruling (Enactment):**
Bill ratified. The Polynomial-Wavelet Outer-Product Network (v2) is the constitutionally required PINN architecture for this codebase from this date forward. The Fourier Feature Network is prohibited.

Architecture enacted:
- Polynomial branch: explicit v_walk, omega, peak_angvel_norm + degree-2 cross-terms (19 features → Linear(19,32)) — Article I compliant
- Trainable wavelet bank: 24 Mexican hat wavelets (scale init ~30ms, matching heel-strike Gaussian impulse) + sin(πt), sin(2πt), t deterministic bases (matching push-off and ramp generating functions) — Amendment 21 compliant
- Outer product fusion: (N,32) × (N,32) → (N,1024) — explicit amplitude × shape factorisation
- MLP output head: Linear(1024→128→128→128→6), GELU, no output activation (Amendment 3)
- 166,582 trainable parameters (within 50k–200k human-specified CPU-feasible range)
- Smoke test: forward(shape(32,10), shape(32,)) → shape(32,6) float32 PASS

**Precedent Effect:**
Any future PINN architecture proposal must: (a) demonstrate that all primitive product interactions (v_walk, omega, peak_angvel) are represented as explicit features or explicit computations — not implicit learned weights in a linear projection; (b) justify the frequency/scale regime of any basis function from physical units, not from normalised-input convention; (c) align temporal basis functions with the mathematical form of the walker_model.py generating function for the relevant output (Amendment 21).

**Files Changed:** `simulator/pinn/pinn_model.py` (full rewrite to v2), `simulator/pinn/architecture.json` (updated metadata), `docs/gaitsense_code/bills/bill_layer_setter_v2.md` (Bill archived)
