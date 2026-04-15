# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent LLM validation system built with LangGraph. A 4-agent pipeline (scenario analysis, actor modeling, impact assessment, decision synthesis) with an inter-agent validation layer that checks outputs between each agent handoff. A baseline pipeline (no validators) exists for comparison experiments.

## Environment

- **Python**: 3.11.9
- **Virtual environment**: `masvenv/` (Windows)
- **Activation**: `source masvenv/Scripts/activate`

## Commands

```bash
# Activate venv
source masvenv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_ledger.py

# Verify pipeline scaffold
cd .. && python -c "from mas_validation.pipeline.graph import validated_graph; validated_graph.invoke(test_state)"
```

Note: The `mas_validation` package must be imported from the **parent directory** (`VS_code/`), not from inside this directory, since the folder itself is the Python package.

## Architecture

**Pipeline flow (validated):** `START → agent1 → validator_1_to_2 → agent2 → validator_2_to_3 → agent3 → validator_3_to_4 → agent4 → END`

**Pipeline flow (baseline):** `START → agent1 → agent2 → agent3 → agent4 → END`

Both pipelines are LangGraph `StateGraph` instances compiled from `PipelineState` (defined in `state.py`), a TypedDict carrying seed documents, agent outputs, claim ledger, validation results, and fallback flags through the graph.

- **`pipeline/graph.py`** — validated pipeline (exposed as `validated_graph`)
- **`pipeline/baseline.py`** — baseline pipeline without validators (exposed as `baseline_graph`)
- **`agents/scenario_analysis.py`** — Agent 1 (Scenario Analysis). `run_scenario_analysis(state)` sends the seed document to the LLM with a structured prompt requesting `Agent1Output` JSON. Includes markdown code fence stripping, Pydantic validation, one retry with error feedback on parse/validation failure, and graceful `pipeline_status = "failed"` on unrecoverable error. On success writes `agent1_output` and appends extracted claims to `claim_ledger`. Prompt enforces source-only extraction (no world knowledge) and confidence grading by evidence strength.
- **`agents/actor_modeling.py`** — Agent 2 (Actor Modeling). `run_actor_modeling(state)` deserializes actor profiles from `state["scenario_constraints"]["scenario"]`, retrieves upstream `agent_1` claims from the ledger, and prompts the LLM to produce `Agent2Output` JSON. Prompt enforces behavioral consistency with actor profiles (type-specific priority rules), factual anchoring to Agent 1 output, and confidence grading. On success adds both top-level `extracted_claims` and per-actor `predicted_actions` to the ledger. Same retry and graceful failure pattern as Agent 1.
- **`agents/impact_assessment.py`** — Agent 3 (Impact Assessment). `run_impact_assessment(state)` retrieves upstream `agent_2` claims from the ledger and prompts the LLM to produce `Agent3Output` JSON with `SystemicEffect` objects and primarily `CausalClaim`/`QuantitativeClaim` extracted claims. Prompt enforces causal integrity (cause_chain must reference real upstream claim_ids), claim type constraint (70%+ causal/quantitative), and confidence grading. Early exit with `pipeline_status = "failed"` if `agent2_output` is None. Logs claim type breakdown on success. Same retry and graceful failure pattern.
- **`agents/decision_synthesis.py`** — Agent 4 (Decision Synthesis). `run_decision_synthesis(state)` retrieves upstream `agent_3` claims and all flagged claims from the ledger, prompts the LLM to produce `Agent4Output` JSON. Prompt enforces mandatory audit (every flagged claim must appear individually in `flagged_uncertainties`), evidence constraint (supporting_claim_ids must reference real upstream claim_ids), and confidence grading. Overrides `overall_confidence` with Python-computed `statistics.mean()` of key_findings confidence scores. Sets `pipeline_status = "completed"` on success. Early exit if `agent3_output` is None. Same retry and graceful failure pattern.
- **`validators/deterministic.py`** — deterministic validation engine (Classes 1-3). Entry point `validate_claim(claim_data, scenario, ledger)` runs five sequential checks returning `List[ValidationCheckResult]`: (1) schema validation via `ClaimFactory`, (2) scope matching against `ScenarioConstraint.affected_claim_types`, (3) quantitative bound enforcement using `min_`/`max_` prefixed keys in constraint parameters vs claim parameters, (4) behavioral consistency checking `BehavioralClaim.predicted_action` against `Constraint.prohibited_actions` on the matching `ActorProfile`, (5) ledger contradiction detection comparing numeric parameters of same-type/same-entity claims with `TOLERANCE = 0.01` relative difference threshold. Hard severity → "rejected", soft severity → "flagged". Check 5 always runs regardless of Check 2 result. Early return only on Check 1 rejection.
- **`validators/llm_validator.py`** — LLM-based validation layer (Classes 4-5). `detect_discontinuity(premises, reasoning)` (Check 4) sends claims and a conclusion to the LLM as a logic checker — maps `ENTAILED`→passed, `PARTIAL`→flagged, `DISCONTINUOUS`→rejected. `detect_propagation(upstream_claims, downstream_reasoning, ledger)` (Check 5) marks flagged ledger claims as `[UNVERIFIED]` and checks if downstream reasoning treats them as settled fact — maps `PROPAGATED`→rejected, `UNCERTAIN_ACKNOWLEDGED`→flagged, `CLEAN`→passed; returns early with "passed" if no flagged claims exist. Both checks use `_parse_and_validate_llm_response` as a guardrail enforcing valid JSON, required keys (`classification`, `justification`), classification whitelist, and `CLAIM_ID_PATTERN` (UUID reference) in justification. Requires Ollama running locally for live tests.
- **`validators/validator_nodes.py`** — three LangGraph node functions (`validate_1_to_2`, `validate_2_to_3`, `validate_3_to_4`) wiring validation into the pipeline. Core logic in `_execute_validation_handoff` runs a six-step protocol: reconstruct ledger from state, run deterministic checks per claim, run LLM checks (discontinuity + propagation) with try/except graceful degradation to "flagged" if Ollama is unavailable, update per-claim ledger statuses, compute worst overall status, and write results/fallback flags/reasons back to state. Scenario is deserialized from `state["scenario_constraints"]["scenario"]` if present.
- **`ledger.py`** — `ClaimLedger` class for tracking claims with filtering by agent, step, type, and validation status
- **`config.py`** — LLM backend config; supports `ollama` (default, llama3.1) and `gemini` (via OpenRouter)
- **`llm_client.py`** — `get_llm()` factory that returns a LangChain chat model based on `LLM_BACKEND`
- **`schemas/claims.py`** — `BaseClaim` and four subtypes (`FactualClaim`, `BehavioralClaim`, `CausalClaim`, `QuantitativeClaim`), `ClaimFactory` for dict→subclass reconstruction, and `CLAIM_TYPE_MAP`
- **`schemas/agent_outputs.py`** — output schemas for all four pipeline agents (`Agent1Output`–`Agent4Output`) and helper models (`Constraint`, `ActorResponse`, `SystemicEffect`, `KeyFinding`). `Constraint.prohibited_actions` is validated against the nine `BehavioralClaim.predicted_action` values via a Pydantic field validator. `ActorResponse` keeps its own `extracted_claims` separate from the parent agent output's top-level `extracted_claims`.
- **`schemas/scenario.py`** — ground truth and identity layer: `ActorProfile`, `ScenarioConstraint`, `Scenario` models with `ActorType`, `ConstraintType`, `Severity` enums. `ScenarioConstraint` uses hard/soft severity to control reject vs flag behavior in deterministic validators. `Scenario.injected_errors` is populated only for adversarial test runs.
- **`experiments/`** — experiment runner, metrics, scenario files, actor profiles, and injection data
