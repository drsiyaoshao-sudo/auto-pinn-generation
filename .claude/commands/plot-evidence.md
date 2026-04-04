Invoke the plot-orchestrator to collect and present evidence during a Judicial Hearing or validation run.

Usage: /plot-evidence <type> [args]

Evidence types:
  signal   <profile> [mode]     — signal diagnostic plot for one walker profile
  uart     <log_path>           — UART session output from a Renode log file
  training <run_id>             — PINN training loss curves + Amendment 20 assessment
  sim      <profile> [mode]     — full simulation evidence: UART + signal plot (all-simulation)
  pinn     <run_id>             — full PINN evidence: loss curves + Amendment 20 (all-pinn)

Arguments:
  profile:  flat | bad_wear | stairs | slope
  mode:     healthy (default) | pathological
  log_path: path to UART log file (e.g. simulator/logs/renode_flat.log)
  run_id:   training run identifier (e.g. run_20260403_141523)
            omit to use the most recent log in simulator/pinn/training_logs/

Examples:
  /plot-evidence signal stairs
  /plot-evidence signal bad_wear pathological
  /plot-evidence uart simulator/logs/renode_flat.log
  /plot-evidence training run_20260403_141523
  /plot-evidence sim flat
  /plot-evidence pinn

Constitutional grounding:
  Amendment 11: signal plots mandatory after any walker_model or algorithm change
  Amendment 20: training evidence must show physics-dominant warmup before data phase
  Bureaucracy Signal Plotting Standing Order: no Bill or hearing required

Now invoke the plot-orchestrator agent with evidence type and arguments: "$ARGUMENTS"
Parse the first word as the evidence type. Pass remaining words as the target (profile, log_path, or run_id).
If no type is given, print the usage above and stop.
