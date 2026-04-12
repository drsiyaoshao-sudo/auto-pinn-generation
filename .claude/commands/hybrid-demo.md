Run the full hybrid cloud-local intelligence demo.

Usage: /hybrid-demo [good_run_id] [bad_run_id]

Arguments (both optional):
- good_run_id: trained PINN checkpoint to use as the "good" model (default: v1)
- bad_run_id:  checkpoint or "random" for untrained random-init model (default: random)

What this demo shows:
The hybrid pipeline splits intelligence by data sensitivity:

  STEP 1 — Local agent (qwen2.5:0.5b + Python)
    Reads: walker_model.py + checkpoint .pt  [PRIVATE — never leaves local]
    Does:  generate IMU signal shapes for good vs bad model across 4 profiles
    Writes: model_compare_<good>_vs_<bad>.png  [DERIVED-OK — safe to forward]

  STEP 2 — Cloud agent (claude-sonnet-4-6)
    Reads: comparison PNG (shapes only)  [DERIVED-OK]
           Article I physics primer      [PUBLIC]
    Does:  reason about what the shape difference means in terms of gait primitives
    Writes: physics insight text  [PUBLIC → stdout]

Terminal output includes:
- Hybrid boundary accounting block (which model runs where, per-file token counts)
- Total PRIVATE tokens shielded from cloud (~23,873 tokens of IP)
- What is actually sent (PNG image + ~674 token primer)
- Shield ratio (shielded tokens / sent tokens)
- Physics insight from Claude grounded in Article I primitives
- Post-call token summary

Prerequisites:
- ANTHROPIC_API_KEY must be in repo .env file (git-ignored) or exported to env
- qwen2.5:0.5b available via Ollama (only needed for LLM parse path; explicit args bypass it)
- best_<good_run_id>.pt checkpoint must exist in simulator/pinn/checkpoints/

Skill contract:
- Local step reads PRIVATE; cloud step reads DERIVED-OK + PUBLIC only
- No PRIVATE content (equations, weights, training data) ever reaches the cloud model
- The derivation chain (walker_model.py → IMU shapes → PNG) is the privacy boundary

Now run the hybrid demo. Execute both steps in sequence:

STEP 1: Run the local model comparison agent.
Call this exact command from the repo root:
    GAITSENSE_DEMO=1 python hybrid/local_agents/run_model_compare.py --good $GOOD --bad $BAD

Where $GOOD is the first argument (default: v1) and $BAD is the second argument (default: random).
Parse $ARGUMENTS: first word = good_run_id, second word = bad_run_id. If empty, use defaults.

Wait for step 1 to complete and confirm the plot was saved.

STEP 2: Run the cloud physics insight agent.
Call this exact command from the repo root:
    python hybrid/cloud_agents/run_physics_insight.py

Do NOT pass --plot; it auto-detects the latest model_compare_*.png.

Do NOT write any inline code. Both modules are the standard callables.
If ANTHROPIC_API_KEY is missing, the script will print a clear setup message and exit — report that to the user and stop.
