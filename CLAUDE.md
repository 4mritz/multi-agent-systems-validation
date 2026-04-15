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
- **`agents/`** — one module per agent node (scenario_analysis, actor_modeling, impact_assessment, decision_synthesis)
- **`validators/deterministic.py`** — deterministic validation engine (Classes 1-3). Entry point `validate_claim(claim_data, scenario, ledger)` runs five sequential checks returning `List[ValidationCheckResult]`: (1) schema validation via `ClaimFactory`, (2) scope matching against `ScenarioConstraint.affected_claim_types`, (3) quantitative bound enforcement using `min_`/`max_` prefixed keys in constraint parameters vs claim parameters, (4) behavioral consistency checking `BehavioralClaim.predicted_action` against `Constraint.prohibited_actions` on the matching `ActorProfile`, (5) ledger contradiction detection comparing numeric parameters of same-type/same-entity claims with `TOLERANCE = 0.01` relative difference threshold. Hard severity → "rejected", soft severity → "flagged". Check 5 always runs regardless of Check 2 result. Early return only on Check 1 rejection.
- **`validators/llm_validator.py`** — LLM-based checks (Classes 4-5), to be implemented
- **`validators/validator_nodes.py`** — validator graph nodes that wire deterministic and LLM checks into the pipeline
- **`ledger.py`** — `ClaimLedger` class for tracking claims with filtering by agent, step, type, and validation status
- **`config.py`** — LLM backend config; supports `ollama` (default, llama3.1) and `gemini` (via OpenRouter)
- **`llm_client.py`** — `get_llm()` factory that returns a LangChain chat model based on `LLM_BACKEND`
- **`schemas/claims.py`** — `BaseClaim` and four subtypes (`FactualClaim`, `BehavioralClaim`, `CausalClaim`, `QuantitativeClaim`), `ClaimFactory` for dict→subclass reconstruction, and `CLAIM_TYPE_MAP`
- **`schemas/agent_outputs.py`** — output schemas for all four pipeline agents (`Agent1Output`–`Agent4Output`) and helper models (`Constraint`, `ActorResponse`, `SystemicEffect`, `KeyFinding`). `Constraint.prohibited_actions` is validated against the nine `BehavioralClaim.predicted_action` values via a Pydantic field validator. `ActorResponse` keeps its own `extracted_claims` separate from the parent agent output's top-level `extracted_claims`.
- **`schemas/scenario.py`** — ground truth and identity layer: `ActorProfile`, `ScenarioConstraint`, `Scenario` models with `ActorType`, `ConstraintType`, `Severity` enums. `ScenarioConstraint` uses hard/soft severity to control reject vs flag behavior in deterministic validators. `Scenario.injected_errors` is populated only for adversarial test runs.
- **`experiments/`** — experiment runner, metrics, scenario files, actor profiles, and injection data
