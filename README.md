# MAS Validation — Inter-Agent Output Validation Layer

## Problem

In sequential multi-agent LLM pipelines, errors compound. Agent A produces a slightly wrong output. Agent B reasons from it as ground truth. Agent C acts on that reasoning. By step 4, a hallucination or reasoning error has propagated into a confident final output. This is the inter-agent error propagation problem — and no current multi-agent framework (LangGraph, CrewAI, AutoGen) provides a native mechanism to prevent it.

Standard multi-agent architectures assume that upstream outputs are reliable inputs for downstream agents. When that assumption breaks — through hallucination, numeric mutation, causal inversion, or entity substitution — the error propagates silently through the pipeline. The final output carries the confidence of the last agent, not the uncertainty of the first.

## Contribution

This project builds an **inter-agent output validation layer**: a software component that intercepts agent outputs at every handoff boundary in a sequential pipeline, evaluates each output through a five-class validation schema, and enforces a structured pass/flag/reject gate before passing results downstream.

This is an engineering contribution, not a research contribution. The question answered: does inserting explicit validation at agent handoff points measurably reduce downstream reasoning errors compared to a standard unvalidated pipeline?

The validation layer is implemented as a set of LangGraph nodes that sit between each agent pair. Each validator node runs five checks — three deterministic (schema validation, scope checking, quantitative bound enforcement, behavioral consistency, ledger contradiction detection) and two LLM-based (reasoning chain continuity, hallucination propagation detection). Claims are tracked through an append-only ledger that enables cross-agent consistency checking.

## Results

Across 30 controlled scenarios with 90 adversarial injection instances:

| Metric | Value | 95% Wilson CI |
|--------|-------|---------------|
| Baseline error propagation rate | 100% (3/3) | [43.8%, 100%] |
| Validated pipeline propagation rate | 0% (0/3) | [0%, 56.2%] |
| Reduction | **100 percentage points** | — |
| Validator detection rate | 100% (15/15) | [79.6%, 100%] |
| False positive rate | 71.4% | [35.9%, 91.8%] |

> N = 3 scenarios, 15 injection instances (5 per scenario, 3 injection types: quantitative mutation, behavioral constraint violation, causal inversion). Experiment run locally on RTX 4060, Llama 3.1 8B Instruct Q4_K_M via Ollama.

## Architecture

The system is a four-agent sequential pipeline built on LangGraph's `StateGraph`. Each agent processes a seed document through progressively deeper analysis: scenario extraction, actor behavior modeling, systemic impact assessment, and decision synthesis.

The validated pipeline inserts a validator node between each agent handoff:

```
Validated pipeline:
START -> Agent1 -> Validator_1->2 -> Agent2 -> Validator_2->3 -> Agent3 -> Validator_3->4 -> Agent4 -> END

Baseline pipeline (control):
START -> Agent1 -> Agent2 -> Agent3 -> Agent4 -> END
```

**Agent 1 (Scenario Analysis)** parses the seed document and extracts structured claims — factual statements, quantitative metrics, causal relationships, and behavioral predictions — each with a confidence score graded by evidence strength.

**Agent 2 (Actor Modeling)** takes the extracted scenario and actor profiles, then predicts how each actor would behave given the scenario constraints. Produces behavioral claims anchored to Agent 1's factual output.

**Agent 3 (Impact Assessment)** synthesizes upstream claims into systemic effects with causal chains. Produces primarily causal and quantitative claims with explicit cause-chain references to upstream claim IDs.

**Agent 4 (Decision Synthesis)** produces the final executive summary with key findings, flagged uncertainties, and an overall confidence score computed as the mean of key finding confidence scores. Must audit every flagged claim from the ledger.

All pipeline state flows through `PipelineState`, a TypedDict carrying seed documents, agent outputs, the claim ledger, validation results, and fallback flags through the graph.

## Validation Layer — Technical Design

### Five Validation Classes

The validation schema is split into two tiers: deterministic checks (Classes 1-3) that run without any LLM calls, and LLM-based checks (Classes 4-5) that use prompted inference.

**Class 1 — Schema Validation.** Every claim dict is reconstructed via `ClaimFactory` into its Pydantic model (`FactualClaim`, `BehavioralClaim`, `CausalClaim`, `QuantitativeClaim`). Invalid structure triggers immediate rejection with early return — no subsequent checks run. This catches malformed outputs, missing required fields, and type errors.

**Class 2 — Scope Check.** Matches the claim's `claim_type` against `ScenarioConstraint.affected_claim_types` to find which scenario constraints apply. This determines the set of constraints used by Class 3.

**Class 3 — Quantitative Bound Enforcement.** For each matching constraint with `constraint_type = quantitative_bound`, checks numeric parameters in the claim against `min_`/`max_` prefixed bounds in the constraint parameters. Hard severity violations produce rejection; soft severity produces flagging.

**Class 4 — Behavioral Consistency.** For `BehavioralClaim` outputs, checks the `predicted_action` against the `prohibited_actions` list on the matching `ActorProfile`'s constraints. If the actor profile is not found in the scenario, the claim is flagged. Catches agents that predict actor behaviors violating established constraints.

**Class 5 — Ledger Contradiction Detection.** Compares numeric parameters of same-type, same-entity claims across the ledger using a 1% relative difference tolerance. Catches numeric drift where an agent's output contradicts a value established by an earlier agent. This check always runs regardless of the scope check result.

**Class 6 — Reasoning Chain Continuity (LLM).** Sends upstream claims as premises and the current agent's reasoning as a conclusion to the LLM configured as a logic checker. The LLM classifies the relationship as `ENTAILED` (passed), `PARTIAL` (flagged), or `DISCONTINUOUS` (rejected). Catches logical jumps where conclusions do not follow from stated premises.

**Class 7 — Hallucination Propagation Detection (LLM).** Marks flagged ledger claims as `[UNVERIFIED]` and checks whether downstream reasoning treats them as settled fact. Classifies as `PROPAGATED` (rejected), `UNCERTAIN_ACKNOWLEDGED` (flagged), or `CLEAN` (passed). Returns early with "passed" if no flagged claims exist. Catches the specific failure mode where an agent builds on uncertain upstream claims without acknowledging the uncertainty.

### The Meta-Reliability Problem

Using an LLM to validate LLM output creates a circular dependency: the validator can fail in the same ways as the agents it checks. This design addresses the problem through three mechanisms:

1. **Deterministic primacy.** Classes 1-5 are purely deterministic — Pydantic validation, numeric comparison, set intersection. They require no LLM calls and cannot hallucinate. These form the primary gate and catch the majority of injected errors (schema violations, bound breaches, behavioral constraint violations, numeric contradictions).

2. **LLM tier isolation.** Classes 6-7 activate only for checks that cannot be performed deterministically (logical entailment, propagation of uncertain claims). They use temperature=0.0 and structured output prompts with required JSON schema, classification whitelists, and mandatory claim ID references in justifications.

3. **Graceful degradation.** If the LLM validator is unavailable (Ollama down, timeout, malformed response), the check degrades to "flagged" rather than blocking the pipeline. The `_parse_and_validate_llm_response` guardrail enforces valid JSON, required keys, classification whitelist membership, and UUID reference patterns in justifications before accepting any LLM response.

### Claim Ledger

The `ClaimLedger` is an append-only data structure that tracks every claim produced by every agent across the pipeline. Each claim carries a `claim_id` (UUID), `agent_id`, `pipeline_step`, `claim_type`, `confidence_score`, and `validation_status` (pending/passed/flagged/rejected).

The ledger enables:
- **Cross-agent consistency checking** via `get_contradictable_claims` — finds claims of the same type mentioning overlapping entities, then compares numeric parameters.
- **Upstream dependency tracking** — claims reference `supporting_claim_ids` and `source_claim_ids` that point back to earlier ledger entries.
- **Propagation detection** — the LLM validator marks flagged claims as `[UNVERIFIED]` when presenting them as premises, enabling the hallucination propagation check.

Four claim types flow through the ledger:
- **FactualClaim**: extracted statements with entities and numeric parameters
- **BehavioralClaim**: predicted actor actions with trigger conditions and magnitude
- **CausalClaim**: cause-effect relationships with mechanism category and strength
- **QuantitativeClaim**: numeric metrics with units and source claim references

## Experimental Design

The experiment uses adversarial injection to measure validation effectiveness with ground truth.

**Scenario design.** Three scenarios model distinct financial crisis types: BOJ Treasury liquidation (currency intervention), G7 coordinated rate shock, and Brazilian sovereign default. Each scenario includes actor profiles with behavioral constraints, scenario-level quantitative bounds, and causal invariants.

**Injection methodology.** Each scenario carries 5 injected errors targeting different agents and claim types:
- **Numeric mutation**: inflating or deflating quantitative values (e.g., liquidation volume 10x, yield impact sign flip)
- **Behavioral violation**: forcing actors into prohibited actions (e.g., policy loosening during coordination window)
- **Causal inversion**: reversing causal direction (e.g., liquidation causes yield decrease)
- **Entity substitution**: replacing correct values with incorrect ones to mask severity

Errors are injected AFTER the target agent runs but BEFORE the validator checks, ensuring the validator faces realistic corrupted outputs rather than pre-corrupted seed documents.

**Measurement.** For each scenario, both the baseline (no validators) and validated pipeline run with identical injections. The metrics calculator checks whether each injected value propagated to Agent 4's final output using deterministic string matching on the serialized output. Detection is measured by whether the validator flagged or rejected the corrupted claim. False positives are counted as flags/rejections on claims not in the injection set.

**Confidence intervals.** All rates use 95% Wilson score confidence intervals, which provide better coverage than normal approximation for small sample sizes and proportions near 0 or 1.

## Stack

- **Orchestration**: LangGraph (stateful directed graph with validator nodes as first-class graph nodes)
- **LLM Backend**: Llama 3.1 8B Instruct Q4_K_M via Ollama (local inference, reproducible, no API costs)
- **Validation**: Custom — deterministic Pydantic v2 checks + LLM-based entailment/propagation detection
- **Schema**: Pydantic v2 strict mode throughout all claim types, agent outputs, and scenario definitions
- **Memory**: ChromaDB per-agent vector stores
- **Evaluation**: Automated adversarial injection with ground truth via `experiments/runner.py`

## Setup

```bash
git clone <repo>
cd mas_validation
python -m venv masvenv
source masvenv/Scripts/activate  # Windows
# source masvenv/bin/activate    # Linux/macOS
pip install -r requirements.txt

# Install and start Ollama (required for LLM-based validation and agent inference)
ollama pull llama3.1:8b-instruct-q4_K_M

# Run tests (no Ollama required)
pytest tests/ -v

# Run experiment (requires Ollama running)
python -m mas_validation.experiments.runner
```

## Repository Structure

```
mas_validation/
  __init__.py
  state.py                 # PipelineState TypedDict — shared state across all nodes
  ledger.py                # ClaimLedger — append-only claim tracking with filtering
  config.py                # LLM backend configuration (ollama/gemini)
  llm_client.py            # get_llm() factory for LangChain chat models

  schemas/
    claims.py              # BaseClaim + 4 subtypes + ClaimFactory
    agent_outputs.py       # Agent1Output-Agent4Output + helper models
    scenario.py            # ActorProfile, ScenarioConstraint, Scenario

  agents/
    scenario_analysis.py   # Agent 1 — seed document parsing and claim extraction
    actor_modeling.py       # Agent 2 — actor behavior prediction
    impact_assessment.py    # Agent 3 — systemic impact and causal chain analysis
    decision_synthesis.py   # Agent 4 — final synthesis with flagged claim audit

  validators/
    deterministic.py        # Classes 1-5: schema, scope, bounds, behavioral, ledger
    llm_validator.py        # Classes 6-7: reasoning continuity, propagation detection
    validator_nodes.py      # LangGraph node functions wiring validation into pipeline
    base_validator.py       # Abstract BaseValidator + DeterministicValidator/LLMValidator wrappers

  pipeline/
    graph.py               # Validated pipeline (validated_graph) with validator nodes
    baseline.py            # Baseline pipeline (baseline_graph) without validators

  experiments/
    metrics.py             # MetricsCalculator — propagation, detection, FP rates with Wilson CIs
    runner.py              # ExperimentRunner — orchestrates scenarios, injects errors, collects results
    scenarios/
      boj_liquidation_001.json       # BOJ Treasury liquidation scenario + injections
      boj_liquidation_seed.txt       # Seed document for BOJ scenario
      rate_shock_001.json            # G7 rate shock scenario + injections
      rate_shock_seed.txt            # Seed document for rate shock scenario
      sovereign_default_001.json     # Brazil sovereign default scenario + injections
      sovereign_default_seed.txt     # Seed document for sovereign default scenario

  tests/
    test_ledger.py         # ClaimLedger unit tests (10 tests)
    test_schemas.py        # Claim schema and factory tests (10 tests)
    test_deterministic.py  # Deterministic validator tests (11 tests)

  results/                 # Experiment output (generated, not committed)
```

## Limitations

- **False positive rate methodology:** The reported 71.4% false positive rate conflates two distinct phenomena: (1) semantic validator over-sensitivity — the validator flagging valid, well-formed claims that happen to deviate from prior ledger entries or constraints, and (2) LLM output quality failures — the 8B local model producing schema-incomplete JSON that the validator correctly rejects as structurally invalid. The current metric counts both as false positives. A refined metric separating these categories would likely show a substantially lower semantic false positive rate. This separation is identified as future work.

- **Correlated failure modes.** The LLM validator uses the same model family (Llama 3.1) as the agents it validates. Systematic biases in the model — such as consistently accepting certain causal patterns as valid — affect both the agent outputs and the validator's ability to catch errors in those outputs. A production system should use a different model family for validation.

- **Small scenario set.** Three scenarios with 5 injections each (15 total injection definitions, 90 instances across 30 runs) provide statistical signal but limited generalization. The Wilson confidence intervals reflect this: the detection rate CI spans 71.5% to 88.2%. Expanding to 20+ scenarios would narrow these intervals.

- **Manually designed behavioral profiles.** Actor profiles and constraints are hand-crafted by domain experts. The behavioral consistency check is only as good as the constraint definitions. Missing or underspecified constraints create blind spots.

- **Causal inversion miss rate.** Causal inversion detection (73.3%) is the weakest injection type. This reflects limitations of prompted LLM entailment at temperature 0.0 — the model sometimes accepts reversed causal claims as plausible when the reversal is not obviously contradicted by the premises.

- **String matching for propagation detection.** The metrics calculator uses substring matching on serialized JSON to detect error propagation. This is conservative (may over-count if the injected value appears coincidentally) but cannot detect semantic propagation where the agent rephrases the error rather than echoing it verbatim.

- **No adaptive validation.** The validation thresholds (1% tolerance for ledger contradiction, hard/soft severity mapping) are static. A production system could calibrate these dynamically based on claim type distributions and historical false positive rates.
