Invoke the plotter agent to generate a PINN training loss diagnostic plot from an existing run log.

Usage: /plot-training <run_id>

Arguments:
- run_id: the run identifier, e.g. run_20260403_141523
  If omitted, uses the most recent log file in simulator/pinn/training_logs/

What this does:
1. Dispatches the plotter agent pointed at simulator/pinn/training_logs/<run_id>.jsonl
2. Plotter generates a 4-panel loss curve plot:
   - Panel 1: total loss (train + val)
   - Panel 2: physics loss components (l_ode, l_vel, l_phase)
   - Panel 3: physics ramp schedule + learning rate
   - Panel 4: summary table (best epoch, best val loss, early stop status)
3. Saves to docs/executive_branch_document/plots/pinn_training/loss_curve_<run_id>.png
4. Prints the summary table to stdout
5. If GAITSENSE_DEMO=1 is set, opens the plot in Preview automatically

Constitutional grounding:
- Amendment 11: signal/training plots are mandatory after any algorithm parameter change
- Amendment 20: plot must show whether physics loss terms converged before data loss dominated
  (warmup phase physics contribution ≥ 80% of total loss, net downward trend over 10 epochs)
- Bureaucracy Signal Plotting Standing Order: no Bill or hearing required

The plotter agent does NOT:
- Modify pinn_model.py, physics_loss.py, or train_config.json
- Propose changes to loss weights or architecture based on what it observes
- Archive checkpoints (that is pinn-archivist's role)
- Rule on whether the run satisfies Amendment 20 (that is the Justice's role)

Example invocations:
  /plot-training
  /plot-training run_20260403_141523

Now invoke the plotter agent with the run_id "$ARGUMENTS".
If no run_id is given, find the most recent .jsonl file in simulator/pinn/training_logs/ and use that.
If the training_logs directory is empty or does not exist, report it and stop.
