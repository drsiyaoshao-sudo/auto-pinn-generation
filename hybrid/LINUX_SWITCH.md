# GaitSense Hybrid — Linux Switch Guide

**Branch:** hybrid-model  
**Current state:** macOS M1 PoC — fixed scripts working, Ollama wrapper written, 7B LLM too slow on CPU  
**Target:** Linux with GPU — full hybrid pipeline demo before YC application (~2026-04-24)

---

```
  LINUX SWITCH FLOWCHART
  ══════════════════════════════════════════════════════════════════

  CURRENT STATE (macOS M1)
  ─────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  ✓ hybrid/scripts/plot_profile.py  — fixed, runs on M1      │
  │  ✓ hybrid/wrapper/ollama_client.py — wrapper written        │
  │  ✓ qwen2.5-coder:7b pulled via Ollama                       │
  │  ✗ 7B generation too slow on M1 CPU (no Metal acceleration) │
  │  ✗ LLM tasks (parse_to_json, summarise) not yet validated   │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 1 — LINUX ENVIRONMENT SETUP
  ──────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  OS:    Ubuntu 22.04 LTS (recommended) or Debian 12         │
  │  GPU:   NVIDIA — any with ≥8 GB VRAM (RTX 3080+ preferred)  │
  │  CUDA:  12.1+ with cuDNN 8+                                 │
  │                                                              │
  │  $ sudo apt install python3.11 python3.11-venv git curl     │
  │  $ python3.11 -m venv .venv && source .venv/bin/activate    │
  │  $ pip install torch==2.2.2 scipy matplotlib requests       │
  │  $ pip install scikit-learn numpy                            │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 2 — INSTALL OLLAMA (Linux)
  ────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  $ curl -fsSL https://ollama.com/install.sh | sh             │
  │  $ ollama serve &                                            │
  │                                                              │
  │  Verify GPU detected:                                        │
  │  $ ollama ps  ← should show GPU column after a model loads  │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 3 — PULL MODEL
  ────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  M1 PoC model (kept for compat):                            │
  │  $ ollama pull qwen2.5-coder:7b                              │
  │                                                              │
  │  Linux GPU target (≥8 GB VRAM):                             │
  │  $ ollama pull qwen2.5-coder:7b     ← same, now runs on GPU │
  │                                                              │
  │  Linux GPU upgrade (≥24 GB VRAM):                           │
  │  $ ollama pull qwen2.5-coder:32b    ← handles long context  │
  │                                                              │
  │  Verify GPU execution:                                       │
  │  $ ollama run qwen2.5-coder:7b "reply: ok"                  │
  │  $ ollama ps   ← check SIZE and PROCESSOR columns           │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 4 — CLONE AND CONFIGURE
  ─────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  $ git clone <repo> && cd auto-pinn-generation               │
  │  $ git checkout hybrid-model                                 │
  │                                                              │
  │  Set env vars (override defaults in ollama_client.py):       │
  │  $ export OLLAMA_HOST="http://localhost:11434"               │
  │  $ export OLLAMA_MODEL="qwen2.5-coder:7b"                   │
  │                                                              │
  │  Upgrade model on 24GB machine:                             │
  │  $ export OLLAMA_MODEL="qwen2.5-coder:32b"                  │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 5 — VALIDATE FIXED SCRIPTS
  ─────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Run fixed plotter (no LLM — should work immediately):      │
  │  $ python hybrid/scripts/plot_profile.py --profile flat     │
  │  $ python hybrid/scripts/plot_profile.py --profile stairs   │
  │                                                              │
  │  Expected output:                                           │
  │  ─────────────────────────────────────────────────────      │
  │  SIGNAL DATA — profile=flat  mode=healthy                   │
  │  acc_z peak (raw):      XX.XX m/s²                          │
  │  acc_z peak (filtered): XX.XX m/s²                          │
  │  gyr_y peak:            XXX.X dps                           │
  │  steps detected:        20                                  │
  │  PLOT_PATH: docs/.../flat_signal_check.png                  │
  │                                                              │
  │  PASS criterion: plot file exists + steps_detected == 20    │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 6 — VALIDATE LLM WRAPPER
  ──────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Smoke test (parse + stream):                               │
  │  $ python hybrid/wrapper/ollama_client.py                   │
  │                                                              │
  │  Expected:                                                  │
  │  [test 1] parse_to_json: {...az_peak_raw, steps...}         │
  │  [test 2] stream summarise: <one sentence>                  │
  │  [ollama_client] smoke test PASS                            │
  │                                                              │
  │  PASS criterion: both tests complete in < 30s on GPU        │
  │  FAIL on M1: generation too slow — expected, not a bug      │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 7 — VALIDATE CORPUS BOUNDARY
  ────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Confirm no PRIVATE content reaches the LLM:                │
  │                                                              │
  │  $ grep -r "physics_loss\|walker_model\|lambda_ode" \       │
  │      hybrid/wrapper/                                         │
  │  → must return empty (no PRIVATE refs in wrapper)           │
  │                                                              │
  │  $ grep -r "PRIVATE" hybrid/scripts/plot_profile.py         │
  │  → should show only the "not forwarded" comment             │
  │                                                              │
  │  Corpus contract: wrapper receives PUBLIC + DERIVED-OK only │
  │  Fixed scripts read PRIVATE locally and strip before return │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP 8 — FULL EVIDENCE SESSION TEST
  ────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  With Renode simulation running (from FW repo):              │
  │                                                              │
  │  $ /plot-evidence sim flat                                  │
  │    → plot-orchestrator dispatches uart-reader + plotter      │
  │    → both use fixed scripts + wrapper                       │
  │    → consolidated evidence block printed                    │
  │                                                              │
  │  $ /plot-evidence pinn                                      │
  │    → plot-orchestrator dispatches train-sum                  │
  │    → Amendment 20 assessment printed                        │
  │                                                              │
  │  PASS: evidence block matches expected format from           │
  │        docs/gaitsense_code/evidence_session.md              │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ══════════════════════════════════════════════════════════════════
  FUTURE — MODEL DISTILLATION PATH
  ══════════════════════════════════════════════════════════════════
  ┌──────────────────────────────────────────────────────────────┐
  │  Once Linux pipeline is validated:                          │
  │                                                              │
  │  Task boundary:   LLM handles parsing + routing only        │
  │                   Fixed scripts handle all computation       │
  │                                                              │
  │  Distillation target:                                       │
  │    Input:  DERIVED-OK tables (loss values, UART summaries)  │
  │    Output: structured JSON dicts                            │
  │    Size:   ~1B params sufficient for parse + format tasks   │
  │                                                              │
  │  Candidate base models for distillation:                    │
  │    Qwen2.5-0.5B  — smallest, fast, good JSON formatting     │
  │    Phi-3.5-mini  — 3.8B, strong structured output          │
  │    SmolLM2-1.7B  — designed for on-device deployment        │
  │                                                              │
  │  Training data:                                             │
  │    Generate (input, output) pairs from the fixed scripts    │
  │    Teacher: qwen2.5-coder:32b on Linux GPU                  │
  │    Student: target ~1B model                                │
  │                                                              │
  │  Goal: model small enough to run on nRF9160 companion       │
  │        processor or Raspberry Pi 5 alongside firmware       │
  └──────────────────────────────────────────────────────────────┘

  ══════════════════════════════════════════════════════════════════
  QUICK REFERENCE — COMMANDS TO RUN ON LINUX
  ══════════════════════════════════════════════════════════════════

  # 1. environment
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install torch==2.2.2 scipy matplotlib requests scikit-learn numpy

  # 2. Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  ollama serve &
  ollama pull qwen2.5-coder:7b

  # 3. validate
  python hybrid/scripts/plot_profile.py --profile flat
  python hybrid/scripts/plot_profile.py --profile stairs --mode pathological
  python hybrid/wrapper/ollama_client.py

  # 4. env vars
  export OLLAMA_MODEL="qwen2.5-coder:32b"   # if 24GB VRAM available
  export GAITSENSE_DEMO=1                    # auto-open plots after save
```

---

## Files Changed by This Work (hybrid-model branch)

| File | Status | Purpose |
|---|---|---|
| `hybrid/scripts/plot_profile.py` | New | Fixed plotter — no LLM |
| `hybrid/wrapper/ollama_client.py` | New | LocalLLM wrapper for agents |
| `hybrid/wrapper/__init__.py` | New | Package export |
| `hybrid/local_agents/run_plotter.py` | New | LLM-based plotter (deprecated for M1, target for Linux) |
| `.claude/agents/plot-orchestrator.md` | Updated | Skill contract added |
| `.claude/agents/plotter.md` | Updated | Skill contract added |
| `.claude/agents/uart-reader.md` | Updated | Skill contract added |
| `.claude/agents/train-sum.md` | Updated | Skill contract added |
| `docs/gaitsense_code/corpus_classification.md` | New | Tier classification for all files |
| `docs/gaitsense_code/skill_contract_spec.md` | New | Contract format spec |
| `simulator/pinn/data_config_public.json` | New | PUBLIC split of data config |
| `simulator/pinn/data_config_private.json` | New | PRIVATE split of data config |
