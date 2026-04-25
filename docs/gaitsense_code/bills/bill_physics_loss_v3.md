### BILL: Physics Loss v3 — Replace spring oscillator with data-aligned waveform constraints
Proposed by: loss-setter agent
Date drafted: 2026-04-19
Change type: software (physics_loss.py)

Problem statement:
l_ode() modelled CoM vertical motion as a spring oscillator (d²z/dt² + ω²·z = F_contact).
The walker_model.py generating function for gy_dps is not a spring oscillator — it is a
composite of: exponential decay at heel strike, a linear ramp during mid-stance, and a
half-sine push-off pulse. The ODE residual diverged from val_ode=36 to val_ode=58 over
100 training epochs (v3 run). Amendment 21 (ratified 2026-04-18) prohibits physics loss
terms whose mathematical form contradicts the data-generating function.

Proposed change:
  REMOVE: l_ode() — spring oscillator on az_pred
  ADD:    l_gyy_pulse() — MSE between gy_pred and primitive-derived composite waveform
  ADD:    l_az_gravity() — per-profile mean az_pred vs theoretical DC baseline
  KEEP:   l_vel(), l_phase() — unchanged

Files changed:
  simulator/pinn/physics_loss.py — full rewrite
  simulator/pinn/physics_review_log.json — reset to PENDING

Article/Amendment grounding:
  Article I — amplitude and timing of all new terms trace to cadence_spm,
               step_length_m, vertical_oscillation_cm
  Amendment 21 — mathematical form must match walker_model.py generating function
  Amendment 17 — all lambda values derived from primitive magnitudes (see below)

Physical evidence:
  v3 training log: val_ode 36.0 (epoch 15, best) → 58.0 (epoch 100) — structural divergence
  walker_model.py lines 253-278: gy generating function is NOT a spring oscillator
  Spring oscillator residual d²z/dt² + ω²·z cannot converge to half-sine push-off pulse

Lambda derivation (Amendment 17):
  l_gyy_pulse: reference = mean(peak_angvel_dps²) across 4 anchor profiles
    flat:     peak = (100+65×1.85×0.72)×1.0 = 186.5 dps → 34342 dps²
    bad_wear: same profile → 34342 dps²
    stairs:   peak = same × 1.5 → 279.8 dps → ~33069 (slope=0, stairs×1.5)
    slope:    peak = (100+65×1.65×0.68)×1.073 = 178.9 dps → 31985 dps²
    mean reference = 33435 dps² → lambda_gyy = 1/33435 = 2.994e-05

  l_az_gravity: reference = mean(az_dc²) across 4 anchor profiles
    az_dc = G×cos(slope_rad) + (vert_osc_cm/100) × (cadence_spm/60)² × 2π²
    flat:     9.81 + 0.04×(100/60)²×19.74 = 9.81+2.19 = 12.00 → 144.0 (m/s²)²
    bad_wear: same → 144.0
    stairs:   9.81 + 0.05×(112/60)²×19.74 = 9.81+4.30 = 14.11 → 199.1
    slope:    9.81×cos(5°) + 0.04×(100/60)²×19.74 = 9.77+2.19 = 11.96 → 143.1
    mean reference = 157.6 (m/s²)² → lambda_az = 1/157.6 = 6.054e-03

  l_vel:   lambda_vel = 4.908   (carried from bill_loss_weights_v1, unchanged)
  l_phase: lambda_phase = 73.625 (carried from bill_loss_weights_v1, unchanged)

Expected outcome:
  val_gyy_pulse converges to <100 dps² (MSE on gy waveform) within 100 warmup epochs
  val_az_gravity converges to <0.5 (m/s²)² within 50 epochs
  No divergence of ODE-equivalent term (structural mismatch eliminated)

Branch: hybrid-model
