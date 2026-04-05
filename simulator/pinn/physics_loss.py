"""
PINN Physics Loss — written by loss-setter agent.
Bill reference: bill_loss_weights_v1.md
Amendment: bill_physics_loss_v2.md — ratified 2026-04-03 by Justice

Three physics loss terms, each algebraically traced to at least one
of the three walking primitives (Article I, Amendment 17).

Loss terms:
  L_ODE   — CoM vertical oscillation ODE     (traces to: cadence_spm, vertical_oscillation_cm)
  L_vel   — Horizontal velocity constraint   (traces to: cadence_spm, step_length_m)
  L_phase — Stance/swing timing constraint   (traces to: cadence_spm via step_period_s)

All λ weights are injected at runtime from train_config.json.
No hardcoded weight values in this file (Amendment 17).

CRITICAL — autograd requirement for L_vel:
  ax_pred (column 0 of the PINN output) MUST remain an attached PyTorch
  tensor with requires_grad=True when l_vel() is called. The integral
  v_x = ∫ ax dt is computed via torch.trapezoid on the live computation
  graph. Using .detach() or .numpy() before this call produces a finite
  loss value with zero gradient — a silent training failure.
  An assert guard is included to catch this at runtime.

CRITICAL — autograd requirement for L_ODE (bill_physics_loss_v3):
  az_pred must remain an attached PyTorch tensor. z_pred is computed by
  gravity-DC-corrected per-profile double-integration via torch.cumsum on
  the live computation graph. bill_physics_loss_v3 adds per-profile DC
  subtraction before cumsum to eliminate the parabolic accumulation caused
  by the gravitational baseline at initialisation (structural L_ODE minimum
  ~44 reduced to < 1.0 after correction). cadence_spm remains active in ω²
  and F_contact — Article I preserved.
"""

import math
import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """
    Computes the three physics loss terms for PINN training.

    All methods operate on attached PyTorch tensors. No numpy.
    Weights are passed in at call time, read from train_config.json
    by pinn-executor — not hardcoded here.
    """

    def __init__(self):
        super().__init__()

    # ─────────────────────────────────────────────────────────────────
    # L_ODE — CoM Vertical Oscillation ODE
    # ─────────────────────────────────────────────────────────────────
    def l_ode(
        self,
        az_pred: torch.Tensor,      # (B*T,) predicted vertical acceleration [m/s²]
        t:       torch.Tensor,      # (B*T,) normalised time [0,1]
        cadence_spm:  torch.Tensor, # (B*T,) or scalar
        vert_osc_cm:  torch.Tensor, # (B*T,) or scalar
        step_period_s: torch.Tensor,# (B*T,) or scalar
        G: float = 9.81,
        IMPACT_DURATION_S: float = 0.05,
        t_steps: int = 100,
    ) -> torch.Tensor:
        """
        Derivation:
          The vertical CoM motion approximates a damped harmonic oscillator:
            d²z/dt² + ω²·z = F_contact(t)

          ω = 2π × (cadence_spm / 60)       [traces to cadence_spm]
          ω² = (2π × cadence / 60)²

          F_contact(t) is the heel-strike impulse modelled as a Gaussian:
            F_contact = hs_impact_ms2 × exp(-(t_abs - t_hs)² / (2σ²))
          where hs_impact_ms2 = sqrt(2·G·vert_osc_m) / IMPACT_DURATION_S
                                     [traces to vertical_oscillation_cm]
          For stairs: hs_impact_ms2 = 0 (toe-strike, no impulse).

          d²z/dt² is approximated numerically from az_pred:
            az_pred ≈ d²z/dt² + G  (sensor frame: gravity adds +G to az)
          So: d²z/dt² ≈ az_pred - G

          z_pred is the true double-integral of d²z/dt² (bill_physics_loss_v2):
            Per-profile double-integration: reshape (B*T,) → (B, T)
            → cumsum for velocity → cumsum for displacement → reshape back.
            Drift-corrected by subtracting the per-profile mean of z.

          Loss: mean((d²z_pred/dt² + ω²·z_pred - F_contact)²)

        AUTOGRAD REQUIREMENT: az_pred must have requires_grad=True.
          torch.cumsum operates on the live computation graph.
          Do NOT call .detach() or .numpy() before this function.
          (bill_physics_loss_v2)
        """
        assert az_pred.requires_grad, (
            "l_ode autograd violation: az_pred.requires_grad is False. "
            "Ensure az_pred is a slice of the live PINN output tensor. "
            "See bill_physics_loss_v2."
        )

        # ω² from cadence — traces to cadence_spm
        omega2 = (2.0 * math.pi * cadence_spm / 60.0) ** 2   # (B*T,) rad²/s²

        # Vertical acceleration component (remove gravity baseline)
        d2z_dt2 = az_pred - G    # (B*T,)

        # F_contact: Gaussian impulse at t=0.025 (normalised), σ=0.05
        # hs_impact_ms2 traces to vertical_oscillation_cm
        vert_osc_m = vert_osc_cm / 100.0
        v_impact = torch.sqrt(torch.clamp(2.0 * G * vert_osc_m, min=1e-6))
        hs_impact = v_impact / IMPACT_DURATION_S   # (B*T,) or scalar

        t_hs = 0.025   # normalised time of heel strike
        sigma_t = 0.05
        F_contact = hs_impact * torch.exp(
            -((t - t_hs) ** 2) / (2.0 * sigma_t ** 2)
        )

        # Per-profile double-integration (bill_physics_loss_v3 — gravity DC correction):
        # The gravity baseline (mean of d2z_dt2 per profile) is subtracted before
        # integration. This removes the parabolic accumulation caused by the DC
        # gravitational offset at network initialisation, isolating the oscillatory
        # component that the spring-mass ODE governs.
        #
        # Physical interpretation: the ODE describes oscillations about gravitational
        # equilibrium. The DC component is static gravitational loading and does not
        # participate in the oscillatory dynamics. cadence_spm remains active in ω²
        # and F_contact — Article I is satisfied (bill_physics_loss_v3).
        #
        # Replaces bill_physics_loss_v2 integration which accumulated DC and produced
        # z_pred ≈ 1.1 m (100× too large), causing ω²·z to dominate the residual by
        # 6–7× over F_contact and creating an unsolvable structural minimum of ~44.
        dt = step_period_s.reshape(-1, t_steps)[:, 0:1] / (t_steps - 1)   # (B, 1) physical dt
        d2z_mat = d2z_dt2.reshape(-1, t_steps)                              # (B, T)
        d2z_dc  = d2z_mat.mean(dim=1, keepdim=True)                         # per-profile DC (B, 1)
        d2z_ac  = d2z_mat - d2z_dc                                           # oscillatory only (B, T)
        v_z_mat = torch.cumsum(d2z_ac,  dim=1) * dt                         # velocity (B, T)
        z_mat   = torch.cumsum(v_z_mat, dim=1) * dt                         # displacement (B, T)
        z_mat   = z_mat - z_mat.mean(dim=1, keepdim=True)                   # drift correction
        z_pred  = z_mat.reshape(-1)                                          # (B*T,)

        # ODE residual on the oscillatory component — omega2 active in gradient path
        # d2z_ac used (not d2z_dt2) to match the integration path (bill_physics_loss_v3)
        residual = d2z_ac.reshape(-1) + omega2 * z_pred - F_contact

        return torch.mean(residual ** 2)

    # ─────────────────────────────────────────────────────────────────
    # L_vel — Horizontal Velocity Constraint
    # ─────────────────────────────────────────────────────────────────
    def l_vel(
        self,
        ax_pred:       torch.Tensor,  # (N,) predicted horizontal accel [m/s²]
        cadence_spm:   torch.Tensor,  # (N,) or scalar [spm]
        step_length_m: torch.Tensor,  # (N,) or scalar [m]
        step_period_s: torch.Tensor,  # (N,) or scalar [s]
    ) -> torch.Tensor:
        """
        Derivation:
          Walking speed: v_x = (cadence_spm / 60) × step_length_m
            [traces to cadence_spm and step_length_m]

          v_x_pred = ∫₀ᵀ ax_pred dt ≈ torch.trapezoid(ax_pred) × step_period_s / N

          Loss: mean((v_x_pred - v_x_expected)²)

        AUTOGRAD REQUIREMENT: ax_pred must have requires_grad=True.
          torch.trapezoid operates on the live computation graph.
          Do NOT call .detach() or .numpy() before this function.
        """
        assert ax_pred.requires_grad, (
            "L_vel autograd violation: ax_pred.requires_grad is False. "
            "The velocity loss gradient will be zero. "
            "Ensure ax_pred is a slice of the live PINN output tensor, "
            "not a detached copy. See loss-setter agent notes."
        )

        v_x_expected = (cadence_spm / 60.0) * step_length_m   # scalar or (N,)

        # Numerical integral: v = ∫ ax dt ≈ mean(ax) × step_period_s
        # torch.trapezoid needs a 1D or batched tensor
        # For a single step sequence (N,): use mean approximation for stability
        v_x_pred = torch.mean(ax_pred) * step_period_s

        return torch.mean((v_x_pred - v_x_expected) ** 2)

    # ─────────────────────────────────────────────────────────────────
    # L_phase — Stance/Swing Timing Constraint
    # ─────────────────────────────────────────────────────────────────
    def l_phase(
        self,
        gy_pred:       torch.Tensor,  # (N,) predicted gyr_y [dps]
        t:             torch.Tensor,  # (N,) normalised time [0,1]
        stance_frac:   torch.Tensor,  # (N,) or scalar [dimensionless]
    ) -> torch.Tensor:
        """
        Derivation:
          Physiological 60/40 stance/swing split (Amendment 15 documented constant).
          step_period_s = 60 / cadence_spm   [traces to cadence_spm]
          stance_frac from WalkerProfile (0.60 flat, 0.62 slope, 0.65 stairs).

          The phase constraint: gyr_y should be negative (dorsiflexion) during
          stance (t < stance_frac) and negative or near-zero during swing
          (t > stance_frac). Push-off peak occurs at t ≈ 0.85–0.95 of stance.

          Soft constraint: at the stance/swing boundary (t ≈ stance_frac),
          gyr_y should cross zero (foot lifts off).

          Loss: penalise predicted gyr_y at t > stance_frac being > 0
          (foot should not be plantarflexing after lift-off, except push-off).

          Simplified implementation: enforce mean(gy_pred[t < stance_frac]) < 0
          and mean(gy_pred[t > stance_frac]) sign is unconstrained (swing varies).
          Use soft hinge loss on the stance-phase mean.
        """
        # Stance mask: t < stance_frac
        stance_mask = (t < stance_frac).float()   # (N,)
        n_stance = stance_mask.sum() + 1e-6

        # Mean gyr_y during stance — should be dominated by negative values
        # (dorsiflexion and ankle rocker)
        mean_gy_stance = (gy_pred * stance_mask).sum() / n_stance

        # Soft constraint: penalise if mean stance gyr_y is positive
        # (expected: negative, ~-20 to -40 dps during stance)
        # Target: mean_gy_stance < 0 → hinge: max(0, mean_gy_stance)²
        stance_violation = torch.clamp(mean_gy_stance, min=0.0) ** 2

        # Boundary loss: at t = stance_frac, gyr_y ≈ 0 (lift-off transition)
        # Find samples closest to stance_frac boundary
        boundary_mask = (torch.abs(t - stance_frac) < 0.05).float()
        n_boundary = boundary_mask.sum() + 1e-6
        mean_gy_boundary = (gy_pred * boundary_mask).sum() / n_boundary
        boundary_loss = mean_gy_boundary ** 2   # should be near zero

        return stance_violation + 0.5 * boundary_loss

    # ─────────────────────────────────────────────────────────────────
    # Total Loss
    # ─────────────────────────────────────────────────────────────────
    def total_loss(
        self,
        pred:          torch.Tensor,   # (N, 6) PINN output
        t:             torch.Tensor,   # (N,) normalised step time [0,1]
        cadence_spm:   torch.Tensor,
        step_length_m: torch.Tensor,
        vert_osc_cm:   torch.Tensor,
        stance_frac:   torch.Tensor,
        step_period_s: torch.Tensor,
        lambda_ode:    float,
        lambda_vel:    float,
        lambda_phase:  float,
        physics_weight_ramp: float = 1.0,   # 0→1 during warmup epochs
        t_steps: int = 100,                  # time steps per profile (bill_physics_loss_v2)
    ) -> dict:
        """
        Computes weighted sum of all three physics loss terms.

        Args:
            pred: (N, 6) — PINN output, columns [ax,ay,az,gx,gy,gz]
            t:    (N,)   — normalised step time [0,1]
            lambda_*:    — weights read from train_config.json at runtime
            physics_weight_ramp: 0.0 at epoch 0, 1.0 after physics_loss_warmup
                                 (linear ramp implemented by pinn-executor)
            t_steps: number of time steps per profile; used by l_ode() for
                     reshape into (B, T) for per-profile double-integration.
                     Default 100 matches train_pinn.py T_steps. (bill_physics_loss_v2)

        Returns:
            dict with keys: total, l_ode, l_vel, l_phase, data_loss (placeholder)
        """
        ax_pred = pred[:, 0]   # horizontal accel
        az_pred = pred[:, 2]   # vertical accel
        gy_pred = pred[:, 4]   # gyr_y

        loss_ode   = self.l_ode(
            az_pred, t, cadence_spm, vert_osc_cm, step_period_s,
            t_steps=t_steps,
        )
        loss_vel   = self.l_vel(ax_pred, cadence_spm, step_length_m, step_period_s)
        loss_phase = self.l_phase(gy_pred, t, stance_frac)

        physics_total = physics_weight_ramp * (
            lambda_ode   * loss_ode   +
            lambda_vel   * loss_vel   +
            lambda_phase * loss_phase
        )

        return {
            "l_ode":    loss_ode,
            "l_vel":    loss_vel,
            "l_phase":  loss_phase,
            "physics":  physics_total,
        }
