"""
PINN architecture — written by layer-setter agent.

Architecture: Fourier Feature Network
  input_dim:    10  (WalkerProfile conditioning vector + terrain encoding)
  fourier_dim:  256 (random Fourier features, sigma=1.0)
  hidden_dim:   256
  n_layers:     4   (hidden layers after Fourier projection)
  activation:   GELU
  output_dim:   6   ([ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps])
  output act:   none (unbounded physical quantities)

Input conditioning vector x (dim=10):
  0  cadence_spm              [spm]
  1  step_length_m            [m]
  2  vertical_oscillation_cm  [cm]
  3  slope_deg                [deg]
  4  stance_frac              [dimensionless]
  5  si_stance_true_pct       [%]
  6  mounting_offset_deg      [deg]
  7  loose_fit_attenuation    [dimensionless, 0.5–1.0]
  8  step_variability_ms      [ms]
  9  terrain                  [int: flat=0, slope=1, stairs=2]

Input t: normalised time within one step, shape (N,) or (N,1), range [0, 1]

Output: (N, 6) float32 in physical units — m/s² for accel, dps for gyro.
        This is the Layer 1 interface contract (Amendment 3).
        Identical to generate_imu_sequence() output column order.

Terrain selection rationale (layer-setter):
  data_config.json absent at architecture time → default Fourier (conservative).
  Stairs terrain produces non-sinusoidal sigmoid toe-roll loading that
  benefits from frequency-domain input encoding. Fourier features allow
  the network to represent sharp transitions without spectral bias.
"""

import math
import torch
import torch.nn as nn


TERRAIN_FLAT   = 0
TERRAIN_SLOPE  = 1
TERRAIN_STAIRS = 2

INPUT_DIM    = 10
HIDDEN_DIM   = 256
N_LAYERS     = 4
FOURIER_DIM  = 256
FOURIER_SIGMA = 1.0
OUTPUT_DIM   = 6


class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier Feature embedding for the (x, t) input.

    Projects concatenated [conditioning_vector, t] through a fixed random
    matrix B ~ N(0, sigma²) and returns [sin(2π B z), cos(2π B z)].

    The projection matrix B is registered as a non-trainable buffer —
    it is fixed at construction and does not change during training.
    Output dimension: 2 * fourier_dim.
    """

    def __init__(self, input_dim: int, fourier_dim: int, sigma: float, seed: int = 42):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        B = torch.randn(input_dim, fourier_dim, generator=rng) * sigma
        self.register_buffer("B", B)  # non-trainable, moves with .to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (N, input_dim)
        proj = 2.0 * math.pi * z @ self.B   # (N, fourier_dim)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*fourier_dim)


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network for IMU signal generation.

    forward(x, t) → (N, 6) float32
      x: (N, input_dim) — WalkerProfile conditioning vector (see module docstring)
      t: (N,) or (N, 1)  — normalised step time in [0, 1]
    Output columns: [ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]

    Amendment 3 compliance: output shape and units are identical to
    walker_model.py::generate_imu_sequence() — Layer 2 (imu_model.py) is unchanged.
    """

    def __init__(
        self,
        input_dim:    int   = INPUT_DIM,
        hidden_dim:   int   = HIDDEN_DIM,
        n_layers:     int   = N_LAYERS,
        use_fourier:  bool  = True,
        fourier_dim:  int   = FOURIER_DIM,
        fourier_sigma: float = FOURIER_SIGMA,
        fourier_seed: int   = 42,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.use_fourier = use_fourier

        # Combined input: conditioning vector x (input_dim) + time t (1)
        combined_dim = input_dim + 1

        if use_fourier:
            self.fourier = FourierFeatureEmbedding(combined_dim, fourier_dim, fourier_sigma, fourier_seed)
            first_dim = 2 * fourier_dim
        else:
            self.fourier = None
            first_dim = combined_dim

        # Hidden layers
        layers = []
        in_dim = first_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        self.hidden = nn.Sequential(*layers)

        # Output layer — no activation (unbounded physical quantities)
        self.output_layer = nn.Linear(hidden_dim, OUTPUT_DIM)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for hidden layers, zero bias on output."""
        for m in self.hidden.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)  WalkerProfile conditioning vector
            t: (N,) or (N, 1)  normalised step time [0, 1]
        Returns:
            (N, 6) float32  [ax_ms2, ay_ms2, az_ms2, gx_dps, gy_dps, gz_dps]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)          # (N, 1)

        z = torch.cat([x, t], dim=-1)    # (N, input_dim + 1)

        if self.use_fourier:
            z = self.fourier(z)          # (N, 2*fourier_dim)

        h = self.hidden(z)               # (N, hidden_dim)
        return self.output_layer(h)      # (N, 6)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(**kwargs) -> PINNModel:
    """Convenience constructor — passes kwargs to PINNModel."""
    return PINNModel(**kwargs)


if __name__ == "__main__":
    model = PINNModel()
    print(f"Architecture: Fourier Feature Network")
    print(f"  input_dim={model.input_dim}, hidden_dim={model.hidden_dim}, "
          f"n_layers={model.n_layers}, use_fourier={model.use_fourier}")
    print(f"  Trainable parameters: {model.count_parameters():,}")

    # Smoke test
    N = 32
    x = torch.randn(N, INPUT_DIM)
    t = torch.linspace(0, 1, N)
    out = model(x, t)
    assert out.shape == (N, 6), f"Expected (N,6), got {out.shape}"
    assert out.dtype == torch.float32
    print(f"  Smoke test: forward({x.shape}, {t.shape}) → {out.shape} ✓")
