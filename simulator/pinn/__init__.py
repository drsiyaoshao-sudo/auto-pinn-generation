"""
simulator/pinn — GaitSense PINN package.

Public API (import from here or from the individual modules):

    Inference
    ---------
    load_model(run_id=None, ...)   -> (model, Y_mean, Y_std_safe)
    pinn_one_step(model, Y_mean, Y_std, profile) -> np.ndarray (T_STEPS, 6)
    gt_one_step(profile, seed=42)  -> np.ndarray (T_STEPS, 6)

    Pipeline
    --------
    run_anchor_sim(run_id=None)    -> list of result dicts (one per anchor profile)

    Plotting (called by plotter agent — return path to saved .png)
    --------
    plot_signal_check.run(profile_name, mode="healthy")
    plot_step_unit.run(run_id=None)
    plot_training_loss.run(run_id=None)
    plot_training_data.run(data_dir=None)

Agent entry points
------------------
All modules above expose a run() function callable as:
    python -m simulator.pinn.<module> [args]
or directly:
    python simulator/pinn/<module>.py [args]
"""

from .inference import load_model, pinn_one_step, gt_one_step, T_STEPS, COL, OUTPUT_COLS
from .run_anchor_sim import run as run_anchor_sim

__all__ = [
    "load_model",
    "pinn_one_step",
    "gt_one_step",
    "T_STEPS",
    "COL",
    "OUTPUT_COLS",
    "run_anchor_sim",
]
