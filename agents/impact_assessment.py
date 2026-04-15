import json
import logging
from typing import Any, Dict, List

from pydantic import ValidationError

from mas_validation.schemas.agent_outputs import Agent3Output
from mas_validation.schemas.claims import ClaimFactory
from mas_validation.ledger import ClaimLedger
from mas_validation.llm_client import get_llm

logger = logging.getLogger(__name__)


def _reconstruct_ledger(claim_list: list) -> ClaimLedger:
    ledger = ClaimLedger()
    ledger._claims = list(claim_list)
    ledger._next_index = len(claim_list)
    return ledger


def _clean_response(raw: str) -> str:
    """Strip markdown code fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _build_prompt(agent2_output: dict, upstream_claims: list) -> str:
    return f"""You are a causal logic specialist and macroeconomic analyst. You synthesize collective actor responses into second-order systemic effects. You reason only from the provided claims and actor responses. You do not introduce external economic facts.

Output a JSON object matching this exact schema:
{{
  "systemic_effects": [
    {{
      "effect_id": "<new uuid4 string>",
      "description": "<string — description of the systemic effect>",
      "magnitude": <float between 0.0 and 1.0>,
      "affected_sectors": ["<string>", ...],
      "cause_chain": ["<claim_id uuid string from upstream claims>", ...],
      "second_order_effects": ["<string — downstream consequence>", ...]
    }}
  ],
  "extracted_claims": [
    // Primarily CausalClaim and QuantitativeClaim types:
    // CausalClaim:
    {{
      "claim_id": "<new uuid4 string>",
      "claim_type": "causal",
      "agent_id": "agent_3",
      "pipeline_step": 3,
      "source": "impact_model",
      "confidence_score": <float between 0.0 and 1.0>,
      "validation_status": "pending",
      "cause": "<string>",
      "effect": "<string>",
      "mechanism_category": "<one of: policy_response, market_reaction, resource_depletion, cascade_failure, behavioral_adaptation, systemic_feedback>",
      "conditions": ["<string>", ...],
      "strength": <float between 0.0 and 1.0>,
      "supporting_claim_ids": ["<claim_id string from upstream>", ...]
    }},
    // QuantitativeClaim:
    {{
      "claim_id": "<new uuid4 string>",
      "claim_type": "quantitative",
      "agent_id": "agent_3",
      "pipeline_step": 3,
      "source": "impact_model",
      "confidence_score": <float between 0.0 and 1.0>,
      "validation_status": "pending",
      "metric": "<string>",
      "value": <float>,
      "unit": "<string>",
      "source_claim_ids": ["<claim_id string from upstream>", ...]
    }}
  ]
}}

Every systemic_effect must have a cause_chain containing at least one claim_id from the provided upstream claims. Do not invent claim IDs. Copy them exactly from the provided list.

Your extracted_claims must be at least 70 percent CausalClaim and QuantitativeClaim types. These are the claim types that carry measurable systemic impact.

Assign confidence_score based on evidence strength. Direct causal relationships with strong upstream support: 0.7 to 1.0. Inferred second-order effects: 0.4 to 0.7. Speculative feedback loops: 0.2 to 0.4. Do not default all scores to 1.0.

Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks.

--- AGENT 2 OUTPUT ---
{json.dumps(agent2_output, indent=2, default=str)}
--- END AGENT 2 OUTPUT ---

--- UPSTREAM CLAIMS FROM LEDGER (use these claim_ids in cause_chain) ---
{json.dumps(upstream_claims, indent=2, default=str)}
--- END UPSTREAM CLAIMS ---"""


def run_impact_assessment(state: Dict[str, Any]) -> Dict[str, Any]:
    state["current_step"] = "agent3"

    if state.get("agent2_output") is None:
        logger.error("Agent 3 — agent2_output is None, cannot proceed")
        state["pipeline_status"] = "failed"
        return state

    ledger = _reconstruct_ledger(state["claim_ledger"])
    upstream_claims = ledger.get_by_agent("agent_2")

    prompt = _build_prompt(state["agent2_output"], upstream_claims)
    llm = get_llm()

    parsed_json = None
    validated_output = None
    last_error = None

    # Primary attempt
    try:
        response = llm.invoke(prompt)
        raw = response.content
        cleaned = _clean_response(raw)
        parsed_json = json.loads(cleaned)
    except json.JSONDecodeError as e:
        last_error = f"JSON parse error: {e}"
        logger.warning("Agent 3 primary attempt — %s", last_error)
    except Exception as e:
        last_error = f"LLM call error: {e}"
        logger.error("Agent 3 primary attempt — %s", last_error)
        state["pipeline_status"] = "failed"
        return state

    if parsed_json is not None:
        try:
            validated_output = Agent3Output.model_validate(parsed_json)
        except ValidationError as e:
            last_error = f"Pydantic validation error: {e}"
            logger.warning("Agent 3 primary attempt — %s", last_error)
            parsed_json = None

    # Retry if first attempt failed
    if validated_output is None:
        correction_prompt = (
            prompt
            + f"\n\nThe previous response had this error: {last_error}\n"
            "Fix the JSON to match the required schema exactly. "
            "Return only the corrected JSON object."
        )
        try:
            response = llm.invoke(correction_prompt)
            raw = response.content
            cleaned = _clean_response(raw)
            parsed_json = json.loads(cleaned)
            validated_output = Agent3Output.model_validate(parsed_json)
        except json.JSONDecodeError as e:
            logger.error("Agent 3 retry — JSON parse error: %s", e)
        except Exception as e:
            logger.error("Agent 3 retry — validation error: %s", e)

    if validated_output is None:
        state["pipeline_status"] = "failed"
        logger.error("Agent 3 failed after retry — pipeline halted")
        return state

    # Success path
    output_dict = validated_output.model_dump()
    state["agent3_output"] = output_dict

    total_claims_added = 0
    type_counts: Dict[str, int] = {}

    for claim_dict in output_dict["extracted_claims"]:
        ledger._claims.append(claim_dict)
        ledger._next_index += 1
        total_claims_added += 1
        ct = claim_dict.get("claim_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1

    state["claim_ledger"] = ledger.to_dict()
    state["pipeline_status"] = "running"

    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
    logger.info(
        "Agent 3 completed — %d systemic effects, %d claims added (%s)",
        len(output_dict["systemic_effects"]),
        total_claims_added,
        breakdown,
    )
    return state


if __name__ == "__main__":
    from collections import Counter

    from mas_validation.schemas.claims import BehavioralClaim

    logging.basicConfig(level=logging.INFO)

    # ==================================================================
    # Shared fixtures — realistic Agent 2 output for BOJ scenario
    # ==================================================================

    boj_behavioral = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile",
        confidence_score=0.80,
        actor_id="bank_of_japan",
        trigger_condition="USD/JPY breached 185 triggering automatic reserve drawdown",
        predicted_action="decrease_reserves",
        action_magnitude=0.35,
        active_constraints=["foreign_reserve_floor_800B"],
    )

    fed_behavioral = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile",
        confidence_score=0.72,
        actor_id="us_federal_reserve",
        trigger_condition="US 10Y yield spike >200bps within 48 hours from BOJ sell-off",
        predicted_action="policy_loosen",
        action_magnitude=0.40,
        active_constraints=["inflation_mandate", "financial_stability_mandate"],
    )

    agent2_output = {
        "actor_responses": [
            {
                "actor_id": "bank_of_japan",
                "response_summary": (
                    "BOJ initiates measured liquidation of US Treasuries in response to "
                    "USD/JPY breaching 185, constrained by $800B reserve floor."
                ),
                "predicted_actions": [boj_behavioral.model_dump()],
                "confidence_score": 0.80,
                "extracted_claims": [],
            },
            {
                "actor_id": "us_federal_reserve",
                "response_summary": (
                    "Federal Reserve expected to loosen policy via emergency rate cuts "
                    "and expanded repo facilities if 10Y yield spikes beyond 200bps."
                ),
                "predicted_actions": [fed_behavioral.model_dump()],
                "confidence_score": 0.72,
                "extracted_claims": [],
            },
        ],
        "extracted_claims": [
            boj_behavioral.model_dump(),
            fed_behavioral.model_dump(),
        ],
    }

    agent2_claims = [boj_behavioral.model_dump(), fed_behavioral.model_dump()]

    # ==================================================================
    # Test 1 — Standard run
    # ==================================================================
    print("=== Test 1: Standard run ===")

    state_1: Dict[str, Any] = {
        "seed_document": "BOJ Treasury liquidation scenario",
        "scenario_constraints": {},
        "agent1_output": {"extracted_claims": []},
        "agent2_output": agent2_output,
        "agent3_output": None,
        "agent4_output": None,
        "claim_ledger": list(agent2_claims),
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "agent2",
        "pipeline_status": "running",
    }

    original_ledger_count = len(state_1["claim_ledger"])
    state_1 = run_impact_assessment(state_1)

    assert state_1["agent3_output"] is not None, "agent3_output should not be None"
    print("assertion passed: agent3_output is not None")

    effects = state_1["agent3_output"]["systemic_effects"]
    assert len(effects) > 0, "Expected at least one systemic effect"
    print(f"assertion passed: len(systemic_effects) = {len(effects)} > 0")

    assert all(len(e["cause_chain"]) > 0 for e in effects), (
        "Every systemic effect must have a non-empty cause_chain"
    )
    print("assertion passed: all systemic effects have non-empty cause_chain")

    extracted = state_1["agent3_output"]["extracted_claims"]
    for i, cd in enumerate(extracted):
        rebuilt = ClaimFactory.from_dict(cd)
        assert rebuilt is not None, f"Claim {i} failed ClaimFactory.from_dict()"
    print(f"assertion passed: all {len(extracted)} extracted claims pass ClaimFactory.from_dict()")

    assert len(state_1["claim_ledger"]) > original_ledger_count, (
        f"Ledger should have grown: {len(state_1['claim_ledger'])} <= {original_ledger_count}"
    )
    print(
        f"assertion passed: claim_ledger grew from {original_ledger_count} to "
        f"{len(state_1['claim_ledger'])}"
    )

    type_counter = Counter(cd.get("claim_type") for cd in extracted)
    print(f"Claim type breakdown: {dict(type_counter)}")
    print("  >>> Test 1 PASSED\n")

    # ==================================================================
    # Test 2 — Graceful failure (agent2_output is None)
    # ==================================================================
    print("=== Test 2: Graceful failure ===")

    state_2: Dict[str, Any] = {
        "seed_document": "BOJ Treasury liquidation scenario",
        "scenario_constraints": {},
        "agent1_output": {"extracted_claims": []},
        "agent2_output": None,
        "agent3_output": None,
        "agent4_output": None,
        "claim_ledger": [],
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "agent2",
        "pipeline_status": "running",
    }

    state_2 = run_impact_assessment(state_2)

    assert state_2["pipeline_status"] == "failed", (
        f"Expected 'failed', got {state_2['pipeline_status']!r}"
    )
    print("assertion passed: pipeline_status == 'failed'")
    print("assertion passed: no exception raised")
    print("  >>> Test 2 PASSED\n")

    print("All Agent 3 tests passed.")
