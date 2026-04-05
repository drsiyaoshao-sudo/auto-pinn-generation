### BILL: Architecture v21a — Chebyshev-t Embedding (T1–T5) + 4-layer MLP

Proposed by: Justice direction (2026-04-05)
Date drafted: 2026-04-05
Change type: software (pinn_model.py — ChebyshevEmbedding layer; train_pinn.py — config wiring; train_config.json — v21 trial run)

---

**Problem statement:**

v20 sanity check (1038-param, 2-layer MLP, raw t input) converged but output is a
monotonic ramp for all channels. Signal plots confirm: gy range [-0.1, +0.4] dps vs
expected ±185 dps. az: slow linear ramp vs heel-strike impulse spike.

Root cause: raw t scalar fed to a 2-layer MLP has function class ≈ linear in t.
Two affine transforms cannot represent the oscillatory, asymmetric gait waveform
(push-off spike at t≈0.6–0.7, heel-strike impulse at t≈0).

---

**Proposed change:**

1. Add ChebyshevEmbedding module to pinn_model.py:
   Maps t ∈ [0,1] → [T₁(t'), T₂(t'), T₃(t'), T₄(t'), T₅(t')] where t' = 2t − 1.
   Implemented via recurrence T_{n+1}(x) = 2x·Tₙ(x) − T_{n−1}(x), T₀=1, T₁=x.
   Output: 5-dim t-feature vector (T₁ through T₅, T₀ dropped as constant).

2. Combined MLP input: [x (10), T₁..T₅(t) (5)] = 15-dim (was 11-dim with raw t).

3. PINNModel new params: use_cheby=bool, cheby_degree=int (default 5).
   When use_cheby=True and use_fourier=False: Chebyshev path active.
   Both False: raw t concatenated (v20 behaviour preserved).

4. train_pinn.py: read model_use_cheby and model_cheby_degree from config.

5. train_config.json: run_id=v21 trial, 100 epochs, model_use_cheby=true,
   model_cheby_degree=5, model_n_layers=4, model_hidden_dim=16 (~1,174 params).

---

**Physical / Article I grounding:**

Chebyshev polynomials of degrees 1–5 on t' ∈ [−1,1] span up to 5 oscillation cycles
per step. Cadence primitive (cadence_spm) defines the fundamental period; harmonics 1–5
of that period cover the push-off spike (≈3rd harmonic bandwidth) and heel-strike
impulse (≈5th harmonic bandwidth). This traces directly to Article I's cadence primitive.

Odd degrees (T₁, T₃, T₅) capture anti-symmetric components (gy: negative stance,
positive push-off). Even degrees (T₂, T₄) capture symmetric components (az: gravity DC
+ bilateral heel-strike impulse). Together they provide a physically grounded basis for
the within-step waveform.

**Amendment 21 compliance:**
- Item 1 (physics match): not addressed here — lambda_ode=lambda_vel=lambda_phase=0
  for this trial. Physics rederivation deferred to next Bill.
- Item 2 (embedding basis): Chebyshev T₁–T₅ spans 5 cycles/step, covering the
  push-off spike bandwidth. This is the embedding redesign required by Item 2.
- Item 3 (capacity): 1,174 params / 350 profiles = 3.4 params/profile. Acceptable.

---

**Expected outcome (100-epoch trial):**
- gy output shows non-monotonic shape: zero-crossings visible in plot
- val loss floor breaks below v20's 0.86
- No overfitting (train ≈ val throughout, as in v20)

**Branch:** constitution-style-management
