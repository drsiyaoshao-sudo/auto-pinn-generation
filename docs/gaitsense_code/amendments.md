# GaitSense Amendments

All amendments derive from Article I (Physics First) and/or Article II (Learner-in-the-Loop) in CLAUDE.md. The governing Articles are unconditional and cannot be amended. New amendments are added through the Amendment Ratification Process defined in CLAUDE.md.

---

### Amendment 1 — Five-Stage Development Order
*Traces to: Article I + II*

Development proceeds in exactly this order — Firmware, Software, Simulation, Edge Cases, Hardware Deployment — and no stage begins until the previous stage's exit criteria are explicitly confirmed by the human.

Expansion: This order exists because each stage's errors become exponentially more expensive to fix in later stages. An agent must not begin Stage N+1 work while Stage N has any open failure, even one that appears unrelated to the next stage's work. Hardware cannot be used as a debugging tool.

*Technical reference: Appendix A — Stage Definitions and Exit Criteria*

---

### Amendment 2 — Three Measurement Primitives
*Traces to: Article I*

Walker profiles must specify `vertical_oscillation_cm`, `cadence_spm`, and `step_length_m` as primary fields. All other signal parameters are derived from these. No other parameters may be set directly.

Expansion: The derivation chain is mandatory, not optional. A walker profile that specifies `hs_impact_g` directly without deriving it from vertical oscillation and cadence violates Article I regardless of whether the resulting signal looks plausible.

*Technical reference: Appendix F — Measurement Philosophy Reference*

---

### Amendment 3 — Seven-Layer Simulation Pipeline Integrity
*Traces to: Article I*

The seven simulation layers are never collapsed. Each layer owns exactly one transformation and must not touch the transformation owned by any other layer.

Expansion: Layer ownership is defined in Appendix B. The boundary table is normative. An agent that passes biomechanical quantities into the IMU model layer, or performs algorithm-level computation in the display layer, has violated this amendment regardless of whether the output is numerically correct.

*Technical reference: Appendix B — Simulation Infrastructure Reference*

---

### Amendment 4 — Stage Gate Confirmation
*Traces to: Article II*

Before advancing from any stage to the next, an agent must state each exit criterion, confirm explicitly whether it is met, and record the human's confirmation verbatim. Advancement without this record is not permitted.

Expansion: Assumed confirmation is not confirmation. The agent states the criteria. The human confirms. The agent records the confirmation. This protects against the most common failure mode in hardware development: a stage that passes without anyone verifying what was actually tested.

---

### Amendment 5 — Simulation is the Hardware Proxy
*Traces to: Article I + II*

If something cannot be tested in simulation, a simulation test must be written first. Hardware is a validation tool, not a debugging tool.

Expansion: A hardware result that deviates from the simulation prediction is evidence of a hardware or mounting problem, not a firmware problem — unless the corresponding simulation test was never written. The handoff document (`docs/executive_branch_document/handoff.md`) is the binding prediction set against which hardware results are compared.

---

### Amendment 6 — Hardware Deployment Irreversibility
*Traces to: Article II*

No agent may initiate or recommend a firmware flash without explicit human approval in the same conversation turn that the flash is requested.

Expansion: "Flash" means any action that writes firmware to physical hardware. The agent provides the flash command and bring-up checklist. The human executes. The agent's role ends at handing the human the verified command.

---

### Amendment 7 — Three-Strike Escalation Rule
*Traces to: Article II*

If a simulation, unit test, or iterative fix process fails to meet exit criteria within three attempts, the agent must stop, report the full status to the human, and wait for a human determination before any further action.

Expansion: Continuing past three failures compounds token debt and masks the root cause. The three-strike report must include: what was attempted, what was observed on each attempt, and what the agent does not know. The agent must not propose a fourth approach without human direction.

---

### Amendment 8 — Bug Triage and Documentation
*Traces to: Article II*

All bugs that require more than one fix attempt must be categorized and documented in `docs/executive_branch_document/bug_receipt.md` using the seven-category taxonomy before the session ends.

The seven categories: walker profile bug, gait algorithm bug, firmware generation bug, Python simulation bug, bare-metal C simulation bug, dependencies bug, hardware porting bug.

Expansion: A bug that is fixed but not categorized is a traceability gap. Future agents and engineers cannot distinguish it from a known risk without this record.

---

### Amendment 9 — Algorithm Search Honesty
*Traces to: Article I + II*

When an algorithm fix domain has been exhausted without resolution, the agent must explicitly state which domain was searched, why it yielded no result, and offer no more than three alternative domains. The human selects exactly one. The hardware iteration option must always remain on the list.

Expansion: An agent that continues searching within an exhausted domain without disclosure violates Article II. Switching domains unilaterally violates the same Article. The hardware iteration option is never automatically eliminated — the cost of the algorithm fix may exceed the cost of a sensor repositioning or BOM change.

---

### Amendment 10 — BOM Optimization Transparency
*Traces to: Article II*

When an agent identifies that an algorithm change enables lower-cost hardware, it must explicitly state this and the physical reasoning before proceeding. The human decides whether to optimize. Accepted BOM changes must be recorded using versioning of BOMs.

Expansion: BOM changes have supply chain, procurement, and schedule consequences an agent does not possess. BOM changes or hardware specification changes require explicit human authorization.

---

### Amendment 11 — Signal Plot Mandate
*Traces to: Article I + II*

After any change to `walker_model.py` or any filter coefficient in `phase_segmenter.c` or `step_detector.c`, an agent must generate a signal plot, save it to `docs/executive_branch_document/plots/`, and wait for human visual confirmation before proceeding.

Expansion: Signal plots are the primary mechanism for catching silent model errors that pass numerical tests. Human visual review of biomechanical plausibility cannot be substituted by a numerical test. An SI value that looks correct can be produced by a physically implausible signal.

*Technical reference: Appendix C — Signal Plot Template and Review Log*

---

### Amendment 12 — Renode Test Template Invariance
*Traces to: Article I*

When creating a new Renode simulation test, copy `scripts/renode_test_template.py` and replace only Sections 2 (signal generation) and 5 (assertions). Sections 1, 3, and 4 must not be modified.

Expansion: Sections 1, 3, and 4 are the invariant infrastructure — MCU platform, bridge execution, and UART result parsing. Modifying these per-test introduces infrastructure drift. A test that passes because of a customized infrastructure section has not validated the firmware.

*Technical reference: Appendix B — Simulation Infrastructure Reference*

---

### Amendment 13 — Calibration Discipline
*Traces to: Article I*

One new calibration constant may be introduced per algorithmic iteration. Every calibration constant must be documented with its physical derivation in CLAUDE.md before the session ends.

Expansion: Calibration constants that cannot be traced to a physical measurement are tuning knobs, not calibrations. A physically derived constant predicts its own hardware value. A tuned constant requires re-tuning at every hardware configuration change.

---

### Amendment 14 — Interim Results and Decision Logging
*Traces to: Article II*

During any iterative build-debug process, intermediate results must be printed to the console for human review. The agent waits for a human determination before proposing the next action. The specific human decision must be recorded verbatim in bug receipt and project memory files.

Expansion: This rule prevents the most common failure mode in agentic development: an agent that runs five sub-steps autonomously, encounters an anomaly in step 2, compensates in step 3, and delivers a result in step 5 that looks correct but carries a hidden assumption no human ever approved. The record of human decisions is the audit trail.

---

### Amendment 15 — Statistical Derivation Documentation
*Traces to: Article I*
*Ratified: 2026-03-29. Proposed by: Claude Sonnet 4.6. Ratified by: sole human engineer.*

Any constant that cannot be derived algebraically from the three walking primitives but is instead derived from a population distribution of one or more primitives must document: the distribution (μ, σ), the sigma bound applied, and any population explicitly excluded from the bound.

Expansion: This amendment closes the gap between Amendment 13 (algebraically derived constants) and constants derived from population statistics of walking primitives. A constant derived from the cadence distribution still traces to Article I — but the statistical path must be made explicit. The failure mode without this rule: statistically derived constants accumulate as undocumented magic numbers that pass Article I review because a primitive is nominally involved, but carry no derivation that predicts their correct value when the target population changes (paediatric, geriatric, athletic). The required documentation format is:

```
/* CONSTANT_NAME — statistically derived from [primitive].
 * Population: [description], distribution ~ N(μ, σ²) [units]
 * Bound: [N]σ upper/lower tail → [value] [units] → CONSTANT = [value]
 * Excluded population: [any explicitly out-of-scope group and why]
 * Traces to: [primitive] (Article I). */
```

First application: `MIN_STEP_INTERVAL_MS = 250` in `src/gait/step_detector.c`.
Human ambulation cadence ~ N(130, 30²) spm across walking + running population.
2.5σ upper tail: 205 spm → minimum step period 293 ms → 250 ms with margin.
Excluded: running downhill (>210 spm, out of scope for SI measurement device).

---

### Amendment 16 — PINN Model Provenance and Versioning
*Traces to: Article I + Article II*
*Ratified: 2026-04-03. Proposed by: Claude Sonnet 4.6. Ratified by: sole human engineer.*

Every trained PINN checkpoint must be uniquely identified by a SHA-256 hash, stored in a manifest alongside the exact (parameter set, loss weight vector, training epoch count, seed) tuple that produced it, and this manifest entry must exist before the checkpoint is used to generate any signal that enters the pipeline.

Expansion: A PINN model is a compressed encoding of assumptions. Without provenance, two signals generated at different times cannot be compared — the model may have been retrained between them. This amendment closes the same traceability gap that Amendment 15 closes for statistical constants: the derivation chain must be auditable. The failure mode without it: a boundary finding from a grid search run is confirmed via Renode, then a subsequent training run silently changes the model, and the boundary can no longer be reproduced. The manifest entry is the minimum required audit trail.

Manifest location: `simulator/pinn/checkpoints/manifest.json`
Registry location: `docs/gaitsense_code/pinn_registry.md`
Responsible agent: `pinn-archivist`

---

### Amendment 17 — PINN Physics Loss Weight Derivation Lock
*Traces to: Article I*
*Ratified: 2026-04-03. Proposed by: Claude Sonnet 4.6. Ratified by: sole human engineer.*

The weight assigned to each physics loss term in the PINN training objective must be derived from the three walking primitives using the same derivation standard as Amendment 13, documented in a ratified Bill before training begins, and may not be adjusted between training runs without a new Bill and human approval.

Expansion: Amendment 13 governs calibration constants in firmware. Amendment 17 extends that same rule to the PINN's physics loss weights. The failure mode without this rule: a researcher tunes the loss weights empirically until training curves look good, producing a network that fits the observed profiles but whose physics enforcement level is arbitrary. The resulting model may pass all 4 profiles and fail on an unseen parameter combination in a way that is entirely predictable from the physics ODE — but was not enforced because the weight on the ODE loss was reduced to improve training stability. This is precisely the scenario Article I forbids: a parameter that cannot be traced to a physical quantity. Each loss weight must be the ratio of the expected physical quantity magnitude (derivable from the primitives) to a reference scale.

Bill format: `docs/gaitsense_code/bills/bill_loss_weights_v{N}.md`
Responsible agent: `loss-setter` (proposes Bill), `physics-reviewer` (generates derivation evidence for human)

---

### Amendment 18 — Grid Search Boundary Confirmation Standard
*Traces to: Article I + Article II*
*Ratified: 2026-04-03. Proposed by: Claude Sonnet 4.6. Ratified by: sole human engineer.*

A PINN-discovered algorithm failure boundary is not a confirmed finding until: a Renode simulation test (per Amendment 12) has been run at the boundary parameter point, the test has produced a non-trivial output (per VABS.F32 Case Precedent), the signal has been plotted (Amendment 11), and the human has confirmed the finding verbatim. A PINN prediction alone — however well-trained — does not constitute an empirical confirmation.

Expansion: The PINN's purpose is discovery — finding parameter combinations where the gait algorithm fails. But a PINN-predicted failure boundary is a model prediction, not an empirical measurement. The PINN may find a boundary because of a limitation in its own physics enforcement, not because of a real firmware failure. The only valid confirmation is Renode: the same firmware ELF that runs on hardware, executing the PINN-generated signal at the boundary parameter point, producing a measurable result. The VABS.F32 case precedent is directly applicable: a boundary at which SI = 0% is reported must be validated under a pathological input (true SI ≠ 0%) to confirm the result is not a silent zero from a computation error. Confirmed boundaries are recorded in `case_law.md` using the standard case format with Renode evidence.

Responsible agent: `pinn-grid-controller` (proposes search domain Bill), main session (runs Renode confirmation)

---

### Amendment 19 — PINN Surrogate Fidelity Threshold
*Traces to: Article I*
*Ratified: 2026-04-03. Proposed by: Claude Sonnet 4.6. Ratified by: sole human engineer.*

Before a PINN model may be used for grid search, it must reproduce all 4 existing walker profiles with a per-axis maximum absolute error below 15% of the peak signal amplitude for that profile, measured against `walker_model.py` output on the same parameter set and seed. This threshold must be re-derived if the sensor specification or any walker profile primitive changes.

Expansion: The 15% fidelity threshold is physically derived, not tuned. The LSM6DS3TR-C accel sensitivity is 4.786×10⁻³ m/s²/LSB at ±16g. At peak heel-strike impact (~18 m/s²), one LSB = 0.027% of peak — far below 15%. The 15% bound is set by the PINN's ability to represent the signal shape, not by sensor quantisation. Physical interpretation: a PINN within 15% of the true peak on all 4 profiles has correctly learned the primitive-to-signal mapping for the known operating points. At 15% of the heel-strike acc_z peak (~18 m/s²): error = 2.7 m/s² — still above the 5.0 m/s² adaptive detection threshold, so step detection is unaffected at known profiles. A PINN that cannot reproduce the 4 known points within this threshold has not learned the physics and must not be trusted for extrapolation into unseen parameter regions.

Responsible agent: `pinn-validator` (Check 1 of 3 in validation pipeline)
