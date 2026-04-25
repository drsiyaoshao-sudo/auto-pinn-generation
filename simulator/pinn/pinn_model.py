"""
PINN architecture v2 — written by layer-setter agent.
Bill reference: bill_layer_setter_v2.md
Date: 2026-04-19

Architecture: Polynomial-Wavelet Outer-Product Network
  input_dim:    10  (WalkerProfile conditioning vector + terrain encoding)
  output_dim:   6   ([ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps])
  activation:   GELU throughout (smooth, better for physics-constrained networks)
  output act:   none (unbounded physical quantities)

Design rationale (full derivation in bill_layer_setter_v2.md):

  The previous Fourier Feature Network projected the concatenated [x, t] vector
  through a fixed random matrix B ~ N(0, sigma^2=1.0) and applied sin/cos. Two
  structural defects made this wrong for this problem:

  DEFECT 1 — Linear frequency projection, nonlinear interaction needed.
    The Fourier projection is linear in the frequency domain. It cannot discover
    nonlinear primitive interactions such as v_walk = (cadence_spm / 60) x step_length_m
    or peak_angvel = (100 + 65 x v_walk) x slope_factor. These interactions are the
    physiologically required amplitude-modulation of the time-domain waveform.
    A linear projection can only learn them if they happen to appear in the training
    data marginals — which they do not, because cadence and step_length are independently
    distributed. The Fourier projection must be replaced by a branch that computes these
    products explicitly (Article I; Amendment 2).

  DEFECT 2 — Wrong frequency regime.
    sigma=1.0 was chosen without reference to the physical units of x. cadence_spm ~ 95,
    step_length_m ~ 0.65. The random projection B z gives 2pi * B * cadence ~ 600 rad,
    placing the dominant Fourier features far outside the [0, 2pi] regime where sin/cos
    encode meaningful variation. The resulting features are chaotically distributed in
    phase, producing a gradient landscape that makes convergence dependent on random seed
    rather than physics. This violates the Benjamin Franklin Principle (CLAUDE.md): a
    parameter whose value cannot be traced to a physical quantity is not a parameter.
    sigma=1.0 cannot be traced to any gait primitive or sensor characteristic.

  REPLACEMENT: Polynomial x Wavelet outer-product architecture.

  Branch 1 — Polynomial (x primitives):
    Computes degree-2 polynomial features from the 3 Article I primitives
    (cadence_spm, step_length_m, vertical_oscillation_cm), then explicitly
    prepends the derived nonlinear quantities:
      v_walk            = (cadence_spm / 60) x step_length_m           [m/s]
      omega             = 2pi x cadence_spm / 60                        [rad/s]
      peak_angvel_norm  = (100 + 65 x v_walk) / 350                    [dimensionless]
    The remaining 7 secondary conditioning inputs (slope_deg, stance_frac,
    si_stance_true_pct, mounting_offset_deg, loose_fit_attenuation,
    step_variability_ms, terrain_int) are passed through linearly.
    Total: 19 features -> Linear -> 32-dim x-embedding.

  Branch 2 — Trainable wavelet bank (t):
    Mexican hat wavelets psi(u) = (1 - u^2) x exp(-u^2 / 2), where
    u = (t - mu) / scale, with trainable center mu in [0,1] and
    trainable log_scale (initialised near log(0.05) for sub-stance
    time resolution). These are second derivatives of Gaussians, matching
    the exponential-decay heel-strike and half-sine push-off pulse shapes
    in the gy waveform (Amendment 21 alignment).
    Also includes: sin(pi * t) and sin(2*pi * t) as deterministic
    half-sine bases (matching the push-off pulse and swing deceleration),
    and a linear ramp basis t (matching mid-stance dorsiflexion ramp).
    Total: n_wavelets + 3 deterministic bases -> Linear -> 32-dim t-embedding.

  Outer-product fusion:
    x-embedding (N, 32) x t-embedding (N, 32) -> outer product (N, 32, 32)
    -> flatten to (N, 1024). This gives the model explicit amplitude-from-
    primitives x shape-from-time factorisation. Each entry of the fused
    representation encodes "how much of this time-domain basis is active
    given this primitive combination", which is exactly the structure of
    the walker_model.py signal (peak_angvel x sin(pi x push_phase)).

  MLP output head:
    Linear(1024, 128) -> GELU -> Linear(128, 128) -> GELU ->
    Linear(128, 128) -> GELU -> Linear(128, 6).
    No activation on output (Amendment 3 — unbounded physical units).

Input conditioning vector x (dim=10):
  0  cadence_spm              [spm]
  1  step_length_m            [m]
  2  vertical_oscillation_cm  [cm]
  3  slope_deg                [deg]
  4  stance_frac              [dimensionless]
  5  si_stance_true_pct       [%]
  6  mounting_offset_deg      [deg]
  7  loose_fit_attenuation    [dimensionless, 0.75–1.0]
  8  step_variability_ms      [ms]
  9  terrain                  [int: flat=0, slope=1, stairs=2]

Input t: normalised time within one step, shape (N,) or (N,1), range [0, 1]

Output: (N, 6) float32 in physical units — m/s² for accel, dps for gyro.
        Layer 1 interface contract (Amendment 3).
        Identical column order to generate_imu_sequence() in walker_model.py.

Terrain scope: stairs present (data_config.json terrain_weights: stairs=0.20 > 0).
Architecture: human-directed wavelet network (bill_layer_setter_v2.md enacted).
Wavelet init: physics-grounded seed positions (bill_layer_setter_v3.md enacted).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Constants — matching walker_model.py (Article I)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIM       = 10
N_WAVELETS      = 24
PROJ_DIM        = 32    # per-branch embedding dimension
OUTER_DIM       = PROJ_DIM * PROJ_DIM   # 1024
HIDDEN_DIM      = 128
N_HIDDEN_LAYERS = 3
OUTPUT_DIM      = 6

# Indices into the x conditioning vector
_IDX_CADENCE    = 0
_IDX_STEP_LEN   = 1
_IDX_VERT_OSC   = 2
_IDX_SECONDARY  = slice(3, 10)   # slope_deg through terrain — 7 inputs


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial branch
# ─────────────────────────────────────────────────────────────────────────────

class PolynomialBranch(nn.Module):
    """
    Extracts 19 physics-grounded features from x (dim=10) and projects to
    PROJ_DIM=32.

    Feature construction (Article I — all features trace to primitives):
      Primitives (indices 0,1,2):
        cadence_spm, step_length_m, vertical_oscillation_cm
      Derived nonlinear quantities (prepended explicitly):
        v_walk           = (cadence_spm / 60) x step_length_m
        omega            = 2*pi * cadence_spm / 60
        peak_angvel_norm = (100 + 65 x v_walk) / 350
          — normalised push-off angular velocity (walker_model.py __post_init__)
          — denominator 350 dps: max of (100 + 65 * 3.5 m/s) rounded to 350
      Degree-2 features of the 3 primitives:
        [c, l, v] -> [c^2, l^2, v^2, c*l, c*v, l*v]   (6 cross/square terms)
      Secondary inputs (linear passthrough, indices 3-9): 7 values
      ---
      Total: 3 (primitives) + 3 (derived) + 6 (degree-2) + 7 (secondary) = 19
    """

    # Normalisation constants derived from data_config.json distributions
    # and walker_model.py physical bounds (Article I, Amendment 15):
    #   cadence_spm:              mu=95, sigma=15  -> scale to [0,1] by /140 (max)
    #   step_length_m:            mu=0.65, sigma=0.12 -> /1.0 (max)
    #   vertical_oscillation_cm:  up to 20 (stairs max) -> /20
    #   v_walk max: (140/60)*1.0 = 2.33 m/s -> /2.5 (with margin)
    #   omega max: 2*pi*140/60 = 14.66 -> /15
    #   peak_angvel_norm: already [0,1] by construction
    _C_NORM = 140.0
    _L_NORM = 1.0
    _V_NORM = 20.0
    _VW_NORM = 2.5
    _OM_NORM = 15.0

    def __init__(self, proj_dim: int = PROJ_DIM):
        super().__init__()
        n_raw_features = 3 + 3 + 6 + 7  # = 19
        self.proj = nn.Linear(n_raw_features, proj_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 10)
        Returns:
            (N, PROJ_DIM) — x-embedding
        """
        cadence  = x[:, _IDX_CADENCE]    / self._C_NORM    # (N,)
        step_len = x[:, _IDX_STEP_LEN]   / self._L_NORM    # (N,)
        vert_osc = x[:, _IDX_VERT_OSC]   / self._V_NORM    # (N,)

        # Derived nonlinear quantities — explicit Article I products
        cadence_hz = x[:, _IDX_CADENCE] / 60.0             # [steps/s]
        v_walk     = cadence_hz * x[:, _IDX_STEP_LEN]      # [m/s]  = cadence*step_length
        omega      = 2.0 * math.pi * cadence_hz             # [rad/s]
        peak_angvel_norm = (100.0 + 65.0 * v_walk) / 350.0 # [dimensionless]

        v_walk_n   = v_walk  / self._VW_NORM
        omega_n    = omega   / self._OM_NORM
        # peak_angvel_norm already dimensionless [0,1]

        # Degree-2 of 3 primitives
        c2  = cadence  ** 2
        l2  = step_len ** 2
        v2  = vert_osc ** 2
        cl  = cadence  * step_len
        cv  = cadence  * vert_osc
        lv  = step_len * vert_osc

        # Secondary inputs (linear passthrough) — already reasonable scale
        # slope_deg/90, stance_frac (already ~0.6), si_stance_true_pct/50,
        # mounting_offset_deg/15, loose_fit_attenuation (already ~0.75-1),
        # step_variability_ms/60, terrain_int/2
        sec = x[:, _IDX_SECONDARY]    # (N, 7) — already finite, passed through

        features = torch.stack([
            cadence, step_len, vert_osc,            # 3 primitives (normalised)
            v_walk_n, omega_n, peak_angvel_norm,    # 3 derived nonlinear
            c2, l2, v2, cl, cv, lv,                 # 6 degree-2 terms
        ], dim=1)                                   # (N, 12)

        features = torch.cat([features, sec], dim=1)  # (N, 19)
        return F.gelu(self.proj(features))             # (N, PROJ_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Wavelet branch
# ─────────────────────────────────────────────────────────────────────────────

class WaveletBranch(nn.Module):
    """
    Trainable Mexican-hat wavelet bank over normalised step time t in [0,1].

    Mexican hat wavelet (second derivative of Gaussian):
      psi(u) = (1 - u^2) * exp(-u^2 / 2),   u = (t - mu) / scale

    Physical justification (Amendment 21):
      The gy waveform in walker_model.py is composed of:
        - Exponential decay at heel-strike: psi'(t) structure at t~0
        - Half-sine push-off pulse at stance end: sin(pi*u) over [0.75, 1.0]
        - Negative half-sine during swing: sin(pi*u) over [0, 1] of swing
        - Linear ramp during mid-stance
      Mexican hat wavelets are second derivatives of Gaussians. A Gaussian
      bell is the envelope of the push-off pulse and heel-strike impulse.
      Its second derivative therefore encodes the curvature of these pulses —
      the correct basis for representing their shape, and the correct basis for
      constraining a network output to match them.

    Deterministic bases (physics-derived, no parameters):
      sin(pi*t)       — single half-sine over full step (push-off + swing)
      sin(2*pi*t)     — double half-sine (stance + swing separately)
      t               — linear ramp (mid-stance dorsiflexion ramp, [0.18,0.75])

    Total output features: n_wavelets + 3 = 27 -> projected to PROJ_DIM.
    """

    def __init__(self, n_wavelets: int = N_WAVELETS, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.n_wavelets = n_wavelets

        # Physics-grounded initialisation (bill_layer_setter_v3, ratified 2026-04-19).
        # Centers and scales derived from walker_model.py _generate_step() event timing.
        # Heel-strike (6): t~0.02-0.13, scale=0.02 (sharp exp decay, hs_sigma~15ms)
        # Mid-stance  (3): t~0.20-0.37, scale=0.07 (linear ramp [0.18,0.75] of stance)
        # Push-off   (10): t~0.40-0.66, scale=0.05 (half-sine, covers stance_frac 0.55-0.70)
        # Swing       (5): t~0.70-0.94, scale=0.10 (neg half-sine, full swing duration)
        hs_mu    = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.10, 0.13])
        ms_mu    = torch.tensor([0.20, 0.28, 0.37])
        push_mu  = torch.tensor([0.40, 0.44, 0.47, 0.50, 0.52, 0.54, 0.57, 0.60, 0.63, 0.66])
        swing_mu = torch.tensor([0.70, 0.76, 0.82, 0.88, 0.94])
        init_mu  = torch.cat([hs_mu, ms_mu, push_mu, swing_mu])   # (24,)

        hs_ls    = torch.full((6,),  math.log(0.02))
        ms_ls    = torch.full((3,),  math.log(0.07))
        push_ls  = torch.full((10,), math.log(0.05))
        swing_ls = torch.full((5,),  math.log(0.10))
        init_log_scale = torch.cat([hs_ls, ms_ls, push_ls, swing_ls])  # (24,)

        self.mu        = nn.Parameter(init_mu)
        self.log_scale = nn.Parameter(init_log_scale)

        n_features = n_wavelets + 3   # wavelet outputs + sin(pi*t) + sin(2pi*t) + t
        self.proj = nn.Linear(n_features, proj_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (N,) or (N,1) normalised step time in [0,1]
        Returns:
            (N, PROJ_DIM) — t-embedding
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)     # (N, 1)

        # Clamp scale to avoid collapse or explosion; exp range: ~[0.01, 0.5]
        scale = torch.clamp(torch.exp(self.log_scale), min=1e-2, max=0.5)  # (W,)
        mu    = torch.clamp(self.mu, min=-0.2, max=1.2)                     # (W,)

        # Broadcast: t is (N,1), mu/scale are (W,)
        u = (t - mu.unsqueeze(0)) / scale.unsqueeze(0)   # (N, W)

        # Mexican hat: psi(u) = (1 - u^2) * exp(-u^2/2)
        u2        = u ** 2
        wavelets  = (1.0 - u2) * torch.exp(-0.5 * u2)   # (N, W)

        # Deterministic half-sine and ramp bases
        sin1  = torch.sin(math.pi  * t)                  # (N, 1)  push-off + swing
        sin2  = torch.sin(2.0 * math.pi * t)             # (N, 1)  stance + swing sep.
        ramp  = t                                         # (N, 1)  mid-stance ramp

        features = torch.cat([wavelets, sin1, sin2, ramp], dim=1)  # (N, W+3)
        return F.gelu(self.proj(features))                          # (N, PROJ_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network for IMU signal generation — v2 architecture.

    forward(x, t) -> (N, 6) float32
      x: (N, input_dim=10) — WalkerProfile conditioning vector
      t: (N,) or (N,1)     — normalised step time in [0, 1]
    Output columns: [ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]

    Amendment 3 compliance: output shape and units identical to
    walker_model.py::generate_imu_sequence() — Layer 2 (imu_model.py) unchanged.

    Architecture summary:
      PolynomialBranch(x)  -> (N, 32)   x-embedding
      WaveletBranch(t)     -> (N, 32)   t-embedding
      outer product        -> (N, 32x32=1024)
      MLP: 1024->128->128->128->6  (GELU, no output activation)

    train_pinn.py interface compatibility:
      - Accepts input_dim, hidden_dim, n_layers, use_fourier kwargs so that
        existing train_config.json keys do not cause TypeError at construction.
      - The wavelet/polynomial hyperparameters (n_wavelets, proj_dim) are set
        here and are not exposed to train_pinn.py — they are architecture
        constants, not training hyperparameters (pinn-compiler's domain).
      - count_parameters() retained for train_pinn.py print statement.
    """

    def __init__(
        self,
        input_dim:     int   = INPUT_DIM,
        hidden_dim:    int   = HIDDEN_DIM,
        n_layers:      int   = N_HIDDEN_LAYERS,
        use_fourier:   bool  = False,   # kept for API compat — ignored in v2
        fourier_dim:   int   = 256,     # kept for API compat — ignored in v2
        fourier_sigma: float = 1.0,     # kept for API compat — ignored in v2
        fourier_seed:  int   = 42,      # kept for API compat — ignored in v2
        n_wavelets:    int   = N_WAVELETS,
        proj_dim:      int   = PROJ_DIM,
    ):
        super().__init__()

        if input_dim != INPUT_DIM:
            raise ValueError(
                f"PINNModel v2: input_dim={input_dim} does not match expected "
                f"INPUT_DIM={INPUT_DIM}. walker_model.py may have changed — "
                f"escalate to layer-setter per standing order."
            )

        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.use_fourier = False   # v2 does not use Fourier features

        # Branch networks
        self.poly_branch   = PolynomialBranch(proj_dim=proj_dim)
        self.wavelet_branch = WaveletBranch(n_wavelets=n_wavelets, proj_dim=proj_dim)

        # MLP output head — takes flattened outer product (proj_dim^2)
        outer_dim = proj_dim * proj_dim
        mlp_layers = []
        in_dim = outer_dim
        for _ in range(n_layers):
            mlp_layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        mlp_layers.append(nn.Linear(hidden_dim, OUTPUT_DIM))  # no activation
        self.mlp = nn.Sequential(*mlp_layers)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all MLP linear layers, zero bias on output."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 10)  WalkerProfile conditioning vector
            t: (N,) or (N,1)  normalised step time [0, 1]
        Returns:
            (N, 6) float32  [ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)       # (N, 1)

        x_emb = self.poly_branch(x)       # (N, proj_dim)
        t_emb = self.wavelet_branch(t)    # (N, proj_dim)

        # Outer product: (N, proj_dim, proj_dim) -> (N, proj_dim^2)
        fused = torch.bmm(
            x_emb.unsqueeze(2),   # (N, proj_dim, 1)
            t_emb.unsqueeze(1),   # (N, 1, proj_dim)
        ).reshape(x_emb.shape[0], -1)    # (N, proj_dim^2)

        return self.mlp(fused)    # (N, 6)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(**kwargs) -> PINNModel:
    """Convenience constructor — passes kwargs to PINNModel."""
    return PINNModel(**kwargs)


if __name__ == "__main__":
    model = PINNModel()
    n_params = model.count_parameters()
    print("Architecture: Polynomial-Wavelet Outer-Product Network (v2)")
    print(f"  input_dim={model.input_dim}")
    print(f"  polynomial branch: 19 features -> {PROJ_DIM}-dim embedding")
    print(f"  wavelet branch: {N_WAVELETS} Mexican-hat + 3 deterministic -> {PROJ_DIM}-dim embedding")
    print(f"  outer product: {PROJ_DIM}x{PROJ_DIM} = {PROJ_DIM*PROJ_DIM}-dim fused")
    print(f"  MLP: {PROJ_DIM*PROJ_DIM} -> {HIDDEN_DIM} x {N_HIDDEN_LAYERS} -> {OUTPUT_DIM}")
    print(f"  use_fourier=False (v2 — replaced by wavelet bank)")
    print(f"  Trainable parameters: {n_params:,}")

    # Smoke test — Amendment 3 compliance
    N = 32
    x = torch.zeros(N, INPUT_DIM)
    x[:, 0] = 95.0    # cadence_spm
    x[:, 1] = 0.65    # step_length_m
    x[:, 2] = 4.5     # vertical_oscillation_cm
    x[:, 7] = 1.0     # loose_fit_attenuation
    t = torch.linspace(0, 1, N)
    out = model(x, t)
    assert out.shape == (N, 6), f"Expected (N,6), got {out.shape}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"
    print(f"  Smoke test: forward({x.shape}, {t.shape}) -> {out.shape} float32  PASS")
