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
- **`validators/`** — validator nodes in `validator_nodes.py`; deterministic checks (Classes 1-3) in `deterministic.py`, LLM-based checks (Classes 4-5) in `llm_validator.py`
- **`ledger.py`** — `ClaimLedger` class for tracking claims with filtering by agent, step, type, and validation status
- **`config.py`** — LLM backend config; supports `ollama` (default, llama3.1) and `gemini` (via OpenRouter)
- **`llm_client.py`** — `get_llm()` factory that returns a LangChain chat model based on `LLM_BACKEND`
- **`schemas/`** — Pydantic schemas for claims, agent outputs, and scenarios
- **`experiments/`** — experiment runner, metrics, scenario files, actor profiles, and injection data
