### BILL: layer-setter-v3 — Physics-grounded wavelet initialisation

Proposed by: main session + human direction
Date drafted: 2026-04-19
Change type: simulation (PINN architecture — wavelet initialisation only)
Status: RATIFIED — 2026-04-19

---

## Problem Statement

v7 training (100 epochs pure-physics warmup) showed model gy output ≈ 0.3 dps versus
physics target ≈ 185 dps. The wavelet bank was not converging to the gy waveform shape.

Root cause: the v2 initialisation places 24 wavelets uniformly at t=[0,1] with
identical scale=0.05 for all. The push-off pulse (the dominant gy feature at
t≈0.45–0.65 for stance_frac 0.55–0.70) has only 1–2 wavelets near it, each with
the same scale as the narrow heel-strike impulse. The gradient from l_gyy_pulse
must simultaneously move 24 wavelets from random positions to the correct physics
positions in 100 epochs — an impossible convergence task with lambda_gyy=3.651e-04.

## Proposed Change

Replace uniform random wavelet initialisation with physics-grounded seed positions
and scales derived from walker_model.py _generate_step() structure.

No change to:
- Wavelet functional form (Mexican hat remains)
- Number of wavelets (24)
- Deterministic bases (sin(pi*t), sin(2*pi*t), t)
- Any other model component

Wavelet groups and initialisation rationale (Article I trace):

| Group | Count | mu range | scale | Source in walker_model.py |
|-------|-------|----------|-------|--------------------------|
| Heel-strike | 6 | 0.02–0.13 | 0.02 | exp(-phase*30) decay, hs_sigma~15ms → t~0.05 |
| Mid-stance | 3 | 0.20–0.37 | 0.07 | linear ramp [0.18,0.75] of stance |
| Push-off | 10 | 0.40–0.66 | 0.05 | half-sine over [0.75,1.0] of stance; peak at 0.875*stance_frac; stance_frac range [0.55,0.70] → push-off peak t in [0.48,0.61] |
| Swing | 5 | 0.70–0.94 | 0.10 | negative half-sine over full swing |

Push-off group concentrated at t=[0.40,0.66] with 10 wavelets (vs 2 in v2) to
cover the full stance_frac distribution across 504 training profiles.

## Amendment Grounding

- Amendment 21: wavelet centres derived from walker_model.py generating function
  event locations. This is the Amendment 21-compliant initialisation that the
  v2 Bill described in principle but did not implement numerically.
- Amendment 13: each scale constant traces to walker_model.py timing parameters.
  heel-strike scale=0.02 ← hs_sigma=15ms / step_period_ms ≈ 15/600 = 0.025.
  push-off scale=0.05 ← push-off width = 0.25*stance_frac ≈ 0.15 / 3 = 0.05.
  swing scale=0.10 ← swing duration = 1 - stance_frac ≈ 0.40 / 4 = 0.10.
- Article II: parameters remain trainable — this is initialisation, not a hard
  constraint. The physics provides the starting point; gradient refines from there.

## Files Changed

- simulator/pinn/pinn_model.py — WaveletBranch.__init__ only (3 lines changed)
- docs/gaitsense_code/bills/bill_layer_setter_v3.md — this document
