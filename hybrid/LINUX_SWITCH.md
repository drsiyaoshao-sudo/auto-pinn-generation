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

## Local Document Processor — Gemma 3 12B QAT Implementation Plan

**Purpose:** Process mixed-tier and ambiguous-sensitivity documents locally. Redacts PRIVATE content, produces structured JSON specs/datasheets for customer discovery. Human gate (Article II) mandatory before any file write.

**Constitutional grounding:** Human Demands Alignment — data sovereignty, IP protection, privacy enforced structurally, not by policy.

```
  GEMMA DOC PROCESSOR — BUILD STEPS
  ══════════════════════════════════════════════════════════════════

  PRE-REQUISITE — PULL GEMMA 3 12B QAT
  ──────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  $ ollama pull gemma3:12b-it-qat                             │
  │                                                              │
  │  If QAT tag unavailable on your Ollama version:             │
  │  $ ollama pull gemma3:12b                                    │
  │  $ export GEMMA_MODEL="gemma3:12b"                          │
  │                                                              │
  │  Verify GPU load:                                            │
  │  $ ollama run gemma3:12b-it-qat "reply: ok"                 │
  │  $ ollama ps  ← check PROCESSOR column shows GPU            │
  │                                                              │
  │  VRAM check: model ~6.6 GB + ~4 GB KV cache = ~10.6 GB     │
  │  RTX 2080 Ti (11 GB) fits with ~400 MB headroom             │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  TWO-TIER ARCHITECTURE
  ──────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Tier 1 — 0.5B gatekeeper (existing gaitsense-dispatch):    │
  │    Input:  sensitivity_gate metadata dict only               │
  │    Output: {"decision":"local|cloud|skip", "reason":...,    │
  │             "token_count":N, "tier_detected":"..."}          │
  │    NEVER sees document content — PRIVATE protected           │
  │                                                              │
  │  Tier 2 — Gemma 3 12B QAT (new):                            │
  │    Input:  document content (any tier — stays local)         │
  │    Output: structured JSON with PRIVATE content redacted     │
  │    Uses Ollama /api/chat with format="json"                  │
  │    num_ctx: 32768 (conservative for QAT VRAM benefit)        │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP A — PHASE 1: PURE PYTHON (no model needed)
  ─────────────────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Create hybrid/schemas/doc_processor_schema.json            │
  │  ─────────────────────────────────────────────              │
  │  JSON Schema (draft-07). Required top-level fields:         │
  │    document_type: spec|datasheet|user_manual|closeout|mixed │
  │    classification: "DERIVED-OK"  ← const, always            │
  │    generated_at: ISO datetime                                │
  │    source_documents: [paths only — no content]               │
  │    model_used: string                                        │
  │    content: { type-specific block }                          │
  │    redactions: [{field, reason}]                             │
  │    human_review_required: true  ← const, Article II         │
  │    human_decision: PENDING|APPROVED|REJECTED|...             │
  │    alignment_check: {physics_aligned, human_demands_aligned} │
  │                                                              │
  │  Type-specific content sub-schemas:                         │
  │    spec:         title, features[], interfaces[],            │
  │                  performance_targets[], limitations[]        │
  │    datasheet:    electrical_specs{}, mechanical_specs{},     │
  │                  communication_interfaces[]                  │
  │    user_manual:  setup_steps[], usage_instructions[],        │
  │                  troubleshooting[], safety_notes[]           │
  │    closeout:     what_was_decided, next_stage_must[],        │
  │                  next_stage_must_not[], reopens_only_if      │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Create hybrid/wrapper/sensitivity_gate.py                  │
  │  ─────────────────────────────────────────                   │
  │  Pure Python — NO LLM. Determines doc tier before any       │
  │  model sees the document.                                    │
  │                                                              │
  │  def check_document_tier(doc_paths, corpus_index_path)       │
  │    → tier: PRIVATE|DERIVED-OK|PUBLIC                         │
  │    → any_private: bool                                       │
  │    → any_mixed: bool                                         │
  │    → heuristic_triggers: [str]                               │
  │    → token_count_estimate: int  (len/4 BPE approx)          │
  │                                                              │
  │  Detection methods:                                          │
  │  1. Exact + glob match against corpus_index.json             │
  │  2. Heuristic regex scan for PRIVATE indicators:            │
  │       LaTeX: \frac \int \lambda \partial                     │
  │       ODE: d[a-z]/dt, double integration                     │
  │       Private bills: bill_loss_weights, bill_physics_loss    │
  │       Arrays: .npy, X_train, Y_train                        │
  │  3. Conservative fallback: paths containing "customer",     │
  │     "clinical", "patient" → DERIVED-OK if no other triggers │
  │                                                              │
  │  Smoke test against known files:                             │
  │  $ python -c "from hybrid.wrapper.sensitivity_gate import   │
  │    check_document_tier; print(check_document_tier(          │
  │    ['docs/gaitsense_code/bills/bill_loss_weights_v1.md']))"  │
  │  → tier: PRIVATE  (contains derivation formulas)            │
  │                                                              │
  │  $ python -c "... check_document_tier(                      │
  │    ['docs/gaitsense_code/bills/bill_train_config_v3.md']))" │
  │  → tier: PUBLIC   (hyperparameters only)                     │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Append 6 entries to docs/gaitsense_code/corpus_index.json  │
  │  ──────────────────────────────────────────────────────────  │
  │  {"path": "hybrid/wrapper/gemma_client.py",   "tier":"PUBLIC"}│
  │  {"path": "hybrid/wrapper/sensitivity_gate.py","tier":"PUBLIC"}│
  │  {"path": "hybrid/local_agents/run_doc_processor.py",       │
  │            "tier": "PUBLIC"}                                 │
  │  {"path": "hybrid/schemas/doc_processor_schema.json",       │
  │            "tier": "PUBLIC"}                                 │
  │  {"path": ".claude/agents/doc-processor.md",  "tier":"PUBLIC"}│
  │  {"path": "docs/executive_branch_document/processed/*.json", │
  │            "tier": "DERIVED-OK"}                             │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP B — PHASE 2: GEMMA WRAPPER (model must be pulled)
  ────────────────────────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Create hybrid/wrapper/gemma_client.py                      │
  │  ─────────────────────────────────────                       │
  │  Mirror LocalLLM structure from ollama_client.py.            │
  │  Key differences:                                            │
  │    - Always uses /api/chat (Ollama handles Gemma template)   │
  │    - No _RAW_QWEN_MODELS list needed                         │
  │    - Always sets "format": "json" in payload                 │
  │    - May receive PRIVATE content (only wrapper that can)     │
  │    - Separate GEMMA_MODEL env var (not OLLAMA_MODEL)         │
  │    - num_ctx: 32768 default                                  │
  │                                                              │
  │  class GemmaClient:                                          │
  │    __init__(model, host, num_ctx=32768)                      │
  │    extract_structured(content, system, stream_progress=True) │
  │    process_document(content, document_type, source_paths,   │
  │                     detected_tier) → schema-conformant dict  │
  │    is_available() → bool                                     │
  │                                                              │
  │  5 embedded system prompts (one per document_type):          │
  │    All prompts share the same redaction rules:               │
  │    1. Extract PUBLIC + DERIVED-OK content                    │
  │    2. Replace PRIVATE derivations with                       │
  │       "[REDACTED — proprietary derivation]"                  │
  │    3. Abstract clinical parameters → functional terms only   │
  │    4. Null + redaction_reason for fully-PRIVATE fields       │
  │                                                              │
  │  process_document() validates output against schema          │
  │  required keys before returning. On fail:                   │
  │    {"status": "SCHEMA_VIOLATION", "raw": <raw_output>}      │
  │  Never writes to file — returns dict to caller.              │
  │                                                              │
  │  Smoke test:                                                 │
  │  $ export GEMMA_MODEL="gemma3:12b-it-qat"                   │
  │  $ python hybrid/wrapper/gemma_client.py                     │
  │  → should print: [gemma_client] smoke test PASS              │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP C — PHASE 3: DISPATCH AGENT + CONTRACT
  ────────────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Create hybrid/local_agents/run_doc_processor.py            │
  │  ───────────────────────────────────────────────             │
  │  Two-tier pipeline. Pattern: run_uart_reader.py              │
  │                                                              │
  │  Critical invariant: 0.5B never sees document content.       │
  │  It receives only sensitivity_gate metadata dict.            │
  │                                                              │
  │  def dispatch(doc_paths, document_type="mixed",             │
  │               output_path=None, llm=None, gemma=None)        │
  │    1. _gate_check(doc_paths) → sensitivity_gate result       │
  │    2. _route_with_0_5b(gate_result, doc_paths, llm)          │
  │         0.5B input: gate metadata JSON only (no file content)│
  │         0.5B output: {decision, reason, token_count, tier}   │
  │    3. Route:                                                 │
  │         skip  → return {status:SKIP, reason:...}            │
  │         cloud → print warning, return {status:CLOUD_BLOCKED} │
  │         local → _process_with_12b() → human gate            │
  │    4. _human_confirmation_gate(result, output_path)          │
  │         prints full JSON to stdout                           │
  │         blocks on input("Type 'confirm' to approve: ")       │
  │         writes to file ONLY if confirmed + path given        │
  │                                                              │
  │  def dispatch_explicit(doc_paths, document_type,            │
  │                         detected_tier, output_path, gemma)   │
  │    Bypasses 0.5B. Human gate remains unconditional.          │
  │                                                              │
  │  Token budget check: if estimate > 24000 tokens:            │
  │    warn + suggest chunk size + prompt continue/abort         │
  │                                                              │
  │  Output destination (on confirmed write):                    │
  │    docs/executive_branch_document/processed/                │
  │      <document_type>_<YYYY-MM-DD>.json                       │
  │    Directory created on first use (mkdir parents ok)         │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Create .claude/agents/doc-processor.md                     │
  │  ──────────────────────────────────────                      │
  │  YAML frontmatter contract:                                  │
  │    name: doc-processor                                       │
  │    tools: Bash, Read, Write                                  │
  │    model: haiku                                              │
  │    contract:                                                 │
  │      execution: local                                        │
  │      retrieves: PRIVATE + DERIVED-OK + PUBLIC (all local)    │
  │      produces: DERIVED-OK JSON → stdout + optional file      │
  │      must_not_forward: PRIVATE                               │
  │      note: "written only after explicit human confirmation"  │
  │                                                              │
  │  Modify hybrid/wrapper/__init__.py — add:                   │
  │    from .gemma_client import GemmaClient                     │
  │    from .sensitivity_gate import (check_document_tier,      │
  │                                    estimate_token_count)     │
  │                                                              │
  │  Modify hybrid/local_agents/__init__.py — add:              │
  │    from .run_doc_processor import (                          │
  │        dispatch as dispatch_doc_processor,                   │
  │        dispatch_explicit as doc_processor_explicit)          │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  STEP D — PHASE 4: INTEGRATION SMOKE TESTS
  ───────────────────────────────────────────
  ┌──────────────────────────────────────────────────────────────┐
  │  Test 1 — PUBLIC doc (expect cloud routing, no 12B):        │
  │  $ python hybrid/local_agents/run_doc_processor.py \        │
  │      docs/gaitsense_code/bills/bill_train_config_v3.md       │
  │  → decision=cloud (PUBLIC only, <4000 tokens)               │
  │  → 12B NOT invoked                                           │
  │                                                              │
  │  Test 2 — PRIVATE doc (expect local routing + redaction):   │
  │  $ python hybrid/local_agents/run_doc_processor.py \        │
  │      docs/gaitsense_code/bills/bill_loss_weights_v1.md       │
  │  → decision=local (PRIVATE detected)                         │
  │  → 12B processes, formula lines replaced with [REDACTED...] │
  │  → human gate fires: type 'confirm' to write JSON            │
  │  → alignment_check.human_demands_aligned == true             │
  │                                                              │
  │  Test 3 — Corpus boundary check (PRIVATE never in wrapper): │
  │  $ grep -r "lambda_ode\|physics_loss\|walker_model" \       │
  │      hybrid/wrapper/gemma_client.py                          │
  │  → empty (model sees content, not formula names in code)     │
  │                                                              │
  │  Test 4 — rag_router compatibility:                         │
  │  $ python -c "from hybrid.wrapper.rag_router import          │
  │    get_permitted_chunks; print(get_permitted_chunks(         │
  │    'doc-processor', 'case law'))"                            │
  │  → returns PUBLIC + DERIVED-OK + PRIVATE chunks             │
  │    (doc-processor contract declares all 3 tiers)             │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
  ══════════════════════════════════════════════════════════════════
  ENV VARS FOR DOC PROCESSOR
  ══════════════════════════════════════════════════════════════════

  # Required
  export GEMMA_MODEL="gemma3:12b-it-qat"   # or gemma3:12b if QAT tag unavailable

  # Optional overrides
  export OLLAMA_HOST="http://localhost:11434"
  export GEMMA_NUM_CTX=32768               # increase to 65536 if docs exceed 24K tokens

  ══════════════════════════════════════════════════════════════════
  SUMMARY OF FILES TO CREATE / MODIFY
  ══════════════════════════════════════════════════════════════════

  CREATE (5 new files):
    hybrid/wrapper/gemma_client.py
    hybrid/wrapper/sensitivity_gate.py
    hybrid/local_agents/run_doc_processor.py
    hybrid/schemas/doc_processor_schema.json        ← new dir: hybrid/schemas/
    .claude/agents/doc-processor.md

  MODIFY (3 existing files):
    hybrid/wrapper/__init__.py                      ← add GemmaClient exports
    hybrid/local_agents/__init__.py                 ← add dispatch_doc_processor export
    docs/gaitsense_code/corpus_index.json           ← append 6 tier entries
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
