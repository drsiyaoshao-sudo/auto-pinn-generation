"""
inference.py — shared PINN inference API.

Imported by all plot scripts. Never run directly.
Loads a checkpoint + norm stats, exposes pinn_one_step() and load_model().

Usage:
    from inference import load_model, pinn_one_step

    model, Y_mean, Y_std = load_model("checkpoints/best_v21_2k.pt",
                                       "checkpoints/Y_norm_stats_v21_2k.npy",
                                       hidden_dim=16, n_layers=4,
                                       use_cheby=True, cheby_degree=5)
    pred = pinn_one_step(model, Y_mean, Y_std, profile)  # (T_STEPS, 6)
"""

import os
import json
import numpy as np
import torch

# ── path helpers ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
CHECKPOINT_DIR   = os.path.join(_HERE, "checkpoints")
TRAIN_CONFIG     = os.path.join(_HERE, "train_config.json")

TERRAIN_INT = {"flat": 0, "slope": 1, "stairs": 2}
T_STEPS     = 208
OUTPUT_COLS = ["ax_ms2", "ay_ms2", "az_ms2", "gx_dps", "gy_dps", "gz_dps"]
COL         = {name: i for i, name in enumerate(OUTPUT_COLS)}


def _resolve_checkpoint(run_id):
    # type: (str) -> tuple
    """Return (ckpt_path, norm_path) for a given run_id.
    If run_id is None, read run_id from train_config.json."""
    if run_id is None:
        with open(TRAIN_CONFIG) as f:
            run_id = json.load(f)["run_id"]
    ckpt = os.path.join(CHECKPOINT_DIR, f"best_{run_id}.pt")
    norm = os.path.join(CHECKPOINT_DIR, f"Y_norm_stats_{run_id}.npy")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not os.path.exists(norm):
        raise FileNotFoundError(f"Norm stats not found: {norm}")
    return ckpt, norm


def load_model(run_id: str = None,
               hidden_dim = None,
               n_layers = None,
               use_cheby = None,
               cheby_degree = None,
               use_fourier = None,
               fourier_dim = None):
    """Load PINN model + norm stats.

    Architecture params default to train_config.json values if not supplied.
    Returns (model, Y_mean, Y_std_safe).
    """
    import sys
    sys.path.insert(0, _HERE)
    from pinn_model import PINNModel

    with open(TRAIN_CONFIG) as f:
        cfg = json.load(f)
    if run_id is None:
        run_id = cfg["run_id"]

    hidden_dim   = hidden_dim   if hidden_dim   is not None else cfg.get("model_hidden_dim",   16)
    n_layers     = n_layers     if n_layers     is not None else cfg.get("model_n_layers",     4)
    use_cheby    = use_cheby    if use_cheby    is not None else cfg.get("model_use_cheby",    False)
    cheby_degree = cheby_degree if cheby_degree is not None else cfg.get("model_cheby_degree", 5)
    use_fourier  = use_fourier  if use_fourier  is not None else cfg.get("model_use_fourier",  False)
    fourier_dim  = fourier_dim  if fourier_dim  is not None else cfg.get("model_fourier_dim",  24)

    ckpt_path, norm_path = _resolve_checkpoint(run_id)

    model = PINNModel(
        hidden_dim=hidden_dim,
        fourier_dim=fourier_dim,
        n_layers=n_layers,
        use_fourier=use_fourier,
        use_cheby=use_cheby,
        cheby_degree=cheby_degree,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    norm = np.load(norm_path)
    Y_mean     = norm[0]   # (6,)
    Y_std_safe = norm[1]   # (6,)

    return model, Y_mean, Y_std_safe


def pinn_one_step(model, Y_mean, Y_std_safe, profile) -> np.ndarray:
    """Run model for one canonical step. Returns (T_STEPS, 6) in physical units."""
    x_vec = np.array([
        profile.cadence_spm,
        profile.step_length_m,
        profile.vertical_oscillation_cm,
        profile.slope_deg,
        profile.stance_frac,
        profile.si_stance_true_pct,
        profile.mounting_offset_deg,
        profile.loose_fit_attenuation,
        profile.step_variability_ms,
        TERRAIN_INT.get(profile.terrain, 0),
    ], dtype=np.float32)

    t = torch.linspace(0, 1, T_STEPS)
    x = torch.from_numpy(x_vec).unsqueeze(0).expand(T_STEPS, -1)
    with torch.no_grad():
        pred_norm = model(x, t).numpy()
    return pred_norm * Y_std_safe + Y_mean   # (T_STEPS, 6)


def gt_one_step(profile, seed: int = 42) -> np.ndarray:
    """Generate one canonical step from walker_model ground truth. Returns (T_STEPS, 6)."""
    import sys
    sys.path.insert(0, os.path.join(_HERE, ".."))
    from walker_model import generate_imu_sequence
    rng = np.random.default_rng(seed)
    seq = generate_imu_sequence(profile, n_steps=2, rng=rng)
    stationary = T_STEPS   # 1 s prefix at 208 Hz
    walking = seq[stationary:]
    return walking[:T_STEPS]
