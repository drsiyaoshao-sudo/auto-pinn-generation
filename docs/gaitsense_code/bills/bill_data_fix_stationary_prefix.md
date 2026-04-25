### BILL: Remove stationary prefix from PINN training data window

Proposed by: human + main session (2026-04-19)
Date drafted: 2026-04-19
Change type: software (generate_training_data.py)
Status: RATIFIED — 2026-04-19

---

## Problem Statement

`generate_training_data.py` calls `generate_imu_sequence(prof, n_steps=1)` and takes
`seq[:SAMPLES_PER_PROFILE]` (first 208 samples). `walker_model.py` defines:

    STATIONARY_PREFIX_SAMPLES = int(ODR_HZ) = 208

The first 208 samples of every generated sequence are a **stationary rest/calibration
prefix** — the device is at rest before walking begins. No push-off pulse occurs.

All 504 training profiles in Y_train therefore contain only the stationary prefix:
  - gy range in Y_train: −0.18 to 0.17 (noise floor, no step signal)
  - Expected gy peak at push-off: 178–185 dps

The physics loss `l_gyy_pulse` targets the full step waveform (t=[0,1] normalised over
208 samples) including a push-off pulse of ~185 dps at t≈0.875. The data target has
no push-off pulse. The two losses are physically incompatible — the model cannot satisfy
both simultaneously, producing the val_gyy plateau observed in v4 and v5 training.

**Article I violation:** Training data that contains no gait signal cannot support the
claim that the PINN is learning the three walking primitives.

---

## Proposed Change

In `simulator/pinn/generate_training_data.py`:

1. Increase `n_steps` from 1 to **5** — ensures all terrain profiles have ≥208 walking
   samples after the prefix:
   - flat:   step period 119 samples, push-off at sample 61  ✓
   - stairs: step period 178 samples, push-off at sample 101 ✓
   - slope:  step period 133 samples, push-off at sample 70  ✓

2. After calling `generate_imu_sequence`, **skip the stationary prefix**:
   ```python
   seq = seq[STATIONARY_PREFIX_SAMPLES:]
   ```
   Then truncate/pad to SAMPLES_PER_PROFILE (208) as before.

No other files change. train_config.json, physics_loss.py, pinn_model.py all unchanged.

---

## Physical Evidence

- `STATIONARY_PREFIX_SAMPLES = 208` confirmed in `walker_model.py` line 297
- Y_train gy max: 0.17 (measured). Physics target peak: 185.1 dps (measured).
- With n_steps=5 + prefix skip, push-off peaks confirmed within 208-sample window
  for all terrain types (measured via `scipy.signal.find_peaks`, height=50 dps).

---

## Expected Outcome

Y_train gy peak ≥ 150 dps across all profiles. Physics loss `l_gyy_pulse` and data
loss MSE no longer conflict. `val_gyy` descends from epoch 1 in v6 training run.

---

## Article/Amendment Grounding

- **Article I:** Training data must contain actual gait signal traceable to the three
  walking primitives. A stationary prefix contains no cadence, step length, or
  vertical oscillation signal.
- **Amendment 21:** Data-physics alignment — the training data window must include
  the same waveform features that the physics loss constrains.
