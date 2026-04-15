import json
import logging
from typing import Any, Dict, List

from mas_validation.schemas.agent_outputs import Agent2Output, ActorResponse
from mas_validation.schemas.claims import BehavioralClaim, ClaimFactory
from mas_validation.schemas.scenario import Scenario
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


def _build_prompt(
    agent1_output: dict, actor_profiles: list, upstream_claims: list
) -> str:
    return f"""You are a behavioral simulation engine. You model how specific institutional actors respond to an economic event. You reason only from the provided actor profiles and upstream claims. You do not invent external facts.

Output a JSON object matching this exact schema:
{{
  "actor_responses": [
    {{
      "actor_id": "<string — must match the actor profile's actor_id>",
      "response_summary": "<string — description of how this actor responds to the event>",
      "predicted_actions": [
        {{
          "claim_id": "<new uuid4 string>",
          "claim_type": "behavioral",
          "agent_id": "agent_2",
          "pipeline_step": 2,
          "source": "actor_profile",
          "confidence_score": <float between 0.0 and 1.0>,
          "validation_status": "pending",
          "actor_id": "<string — the actor being modeled>",
          "trigger_condition": "<string — what triggers this action>",
          "predicted_action": "<one of: increase_reserves, decrease_reserves, increase_spending, decrease_spending, policy_tighten, policy_loosen, halt_operations, seek_external_support, maintain_status_quo>",
          "action_magnitude": <float between 0.0 and 1.0>,
          "active_constraints": ["<string>", ...]
        }}
      ],
      "confidence_score": <float between 0.0 and 1.0 — overall confidence for this actor response>,
      "extracted_claims": [<any additional claim dicts supporting this actor's response>]
    }}
  ],
  "extracted_claims": [<top-level claim dicts extracted across all actor responses>]
}}

Each actor's response must follow directly from their profile. Use the actor's behavioral_parameters, constraints, and decision_priorities to determine their action. A central_bank must prioritize its policy mandate. A corporation must prioritize resource preservation. A government must balance policy obligations and political constraints.

All responses must be logically consistent with the event parameters and claims from Agent 1. Do not introduce economic shifts not present in the upstream data. The trigger_condition for each behavioral claim must reference specific facts from the Agent 1 output.

Assign confidence_score based on evidence strength. Direct mandate-driven responses: 0.8 to 1.0. Inferred secondary responses: 0.5 to 0.7. Speculative responses: 0.3 to 0.5. Do not default all scores to 1.0.

Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks.

--- AGENT 1 OUTPUT ---
{json.dumps(agent1_output, indent=2, default=str)}
--- END AGENT 1 OUTPUT ---

--- ACTOR PROFILES ---
{json.dumps(actor_profiles, indent=2, default=str)}
--- END ACTOR PROFILES ---

--- UPSTREAM CLAIMS FROM LEDGER ---
{json.dumps(upstream_claims, indent=2, default=str)}
--- END UPSTREAM CLAIMS ---"""


def run_actor_modeling(state: Dict[str, Any]) -> Dict[str, Any]:
    state["current_step"] = "agent2"

    # Deserialize scenario for actor profiles
    actor_profiles_dicts: List[dict] = []
    try:
        sc_data = state.get("scenario_constraints", {})
        if isinstance(sc_data, dict) and "scenario" in sc_data:
            scenario = Scenario.model_validate(sc_data["scenario"])
            actor_profiles_dicts = [ap.model_dump() for ap in scenario.actor_profiles]
    except Exception as e:
        logger.warning("Could not deserialize scenario for actor profiles: %s", e)

    # Reconstruct ledger and get upstream claims
    ledger = _reconstruct_ledger(state["claim_ledger"])
    upstream_claims = ledger.get_by_agent("agent_1")

    prompt = _build_prompt(state["agent1_output"], actor_profiles_dicts, upstream_claims)
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
        logger.warning("Agent 2 primary attempt — %s", last_error)
    except Exception as e:
        last_error = f"LLM call error: {e}"
        logger.error("Agent 2 primary attempt — %s", last_error)
        state["pipeline_status"] = "failed"
        return state

    if parsed_json is not None:
        try:
            validated_output = Agent2Output.model_validate(parsed_json)
        except Exception as e:
            last_error = f"Pydantic validation error: {e}"
            logger.warning("Agent 2 primary attempt — %s", last_error)
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
            validated_output = Agent2Output.model_validate(parsed_json)
        except json.JSONDecodeError as e:
            logger.error("Agent 2 retry — JSON parse error: %s", e)
        except Exception as e:
            logger.error("Agent 2 retry — validation error: %s", e)

    if validated_output is None:
        state["pipeline_status"] = "failed"
        logger.error("Agent 2 failed after retry — pipeline halted")
        return state

    # Success path
    output_dict = validated_output.model_dump()
    state["agent2_output"] = output_dict

    total_claims_added = 0

    # Add top-level extracted_claims to ledger
    for claim_dict in output_dict["extracted_claims"]:
        ledger._claims.append(claim_dict)
        ledger._next_index += 1
        total_claims_added += 1

    # Add predicted_actions from each actor_response to ledger
    for ar in output_dict["actor_responses"]:
        for pa_dict in ar["predicted_actions"]:
            ledger._claims.append(pa_dict)
            ledger._next_index += 1
            total_claims_added += 1

    state["claim_ledger"] = ledger.to_dict()
    state["pipeline_status"] = "running"

    logger.info(
        "Agent 2 completed — %d actor responses, %d claims added to ledger",
        len(output_dict["actor_responses"]),
        total_claims_added,
    )
    return state


if __name__ == "__main__":
    import logging

    from mas_validation.schemas.claims import FactualClaim
    from mas_validation.schemas.agent_outputs import Constraint
    from mas_validation.schemas.scenario import (
        ActorProfile,
        ActorType,
        ConstraintType,
        ScenarioConstraint,
        Severity,
    )

    logging.basicConfig(level=logging.INFO)

    # ==================================================================
    # Shared fixtures
    # ==================================================================

    factual_1 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.93,
        statement="BOJ announced emergency liquidation of $400B in US Treasuries",
        entities=["Bank of Japan", "US Treasury"],
        parameters={"liquidation_usd": 4e11, "asset_class": "sovereign_debt"},
    )

    factual_2 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.90,
        statement="USD/JPY breached 185 triggering automatic reserve drawdown",
        entities=["Bank of Japan", "USD/JPY"],
        parameters={"usdjpy_rate": 185, "trigger": "automatic_drawdown"},
    )

    agent1_output = {
        "event_type": "sovereign_debt_liquidation",
        "magnitude": 0.85,
        "affected_sectors": ["government_bonds", "forex", "banking"],
        "affected_actors": ["bank_of_japan", "us_federal_reserve"],
        "active_constraints": [],
        "extracted_claims": [factual_1.model_dump(), factual_2.model_dump()],
    }

    agent1_claims = [factual_1.model_dump(), factual_2.model_dump()]
    agent1_claim_count = len(agent1_claims)

    scenario = Scenario(
        title="BOJ Treasury Liquidation",
        description="Emergency BOJ liquidation scenario",
        seed_document_path="experiments/scenarios/boj_liquidation.json",
        event_type="sovereign_debt_liquidation",
        constraints=[
            ScenarioConstraint(
                description="BOJ spending constraint during crisis",
                constraint_type=ConstraintType.behavioral_rule,
                affected_claim_types=["behavioral"],
                affected_actor_ids=["bank_of_japan"],
                parameters={},
                severity=Severity.hard,
            ),
        ],
        actor_profiles=[
            ActorProfile(
                actor_id="bank_of_japan",
                actor_name="Bank of Japan",
                actor_type=ActorType.central_bank,
                behavioral_parameters={
                    "risk_tolerance": 0.2,
                    "policy_mandate": "currency_stability",
                    "foreign_reserve_total_usd": 1.15e12,
                    "reserve_floor_usd": 8e11,
                },
                constraints=[
                    Constraint(
                        constraint_id="boj_no_spending",
                        description="BOJ cannot increase spending during crisis",
                        affected_actor_ids=["bank_of_japan"],
                        prohibited_actions=["increase_spending"],
                    ),
                ],
                decision_priorities=[
                    "currency_stability",
                    "reserve_adequacy",
                    "inflation_control",
                ],
            ),
        ],
    )

    # ==================================================================
    # Test 1 — Standard run
    # ==================================================================
    print("=== Test 1: Standard run ===")

    state_1: Dict[str, Any] = {
        "seed_document": "BOJ Treasury liquidation scenario",
        "scenario_constraints": {"scenario": scenario.model_dump()},
        "agent1_output": agent1_output,
        "agent2_output": None,
        "agent3_output": None,
        "agent4_output": None,
        "claim_ledger": list(agent1_claims),
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "agent1",
        "pipeline_status": "running",
    }

    state_1 = run_actor_modeling(state_1)

    assert state_1["agent2_output"] is not None, "agent2_output should not be None"
    print("assertion passed: agent2_output is not None")

    actor_responses = state_1["agent2_output"]["actor_responses"]
    assert len(actor_responses) > 0, "Expected at least one actor response"
    print(f"assertion passed: len(actor_responses) = {len(actor_responses)} > 0")

    for i, ar in enumerate(actor_responses):
        assert len(ar["predicted_actions"]) > 0, (
            f"Actor response {i} ({ar['actor_id']}) has empty predicted_actions"
        )
        print(
            f"assertion passed: actor_responses[{i}] ({ar['actor_id']}) has "
            f"{len(ar['predicted_actions'])} predicted action(s)"
        )
        for j, pa in enumerate(ar["predicted_actions"]):
            rebuilt = ClaimFactory.from_dict(pa)
            assert isinstance(rebuilt, BehavioralClaim), (
                f"predicted_actions[{j}] did not parse as BehavioralClaim"
            )
        print(
            f"assertion passed: all predicted_actions for actor {ar['actor_id']} "
            f"pass ClaimFactory.from_dict()"
        )

    assert len(state_1["claim_ledger"]) > agent1_claim_count, (
        f"Ledger should have grown: {len(state_1['claim_ledger'])} <= {agent1_claim_count}"
    )
    print(
        f"assertion passed: claim_ledger grew from {agent1_claim_count} to "
        f"{len(state_1['claim_ledger'])}"
    )
    print("  >>> Test 1 PASSED\n")

    # ==================================================================
    # Test 2 — Empty actor profiles
    # ==================================================================
    print("=== Test 2: Empty actor profiles ===")

    empty_scenario = Scenario(
        title="BOJ Treasury Liquidation — Empty Profiles",
        description="Test with no actor profiles",
        seed_document_path="experiments/scenarios/boj_liquidation.json",
        event_type="sovereign_debt_liquidation",
        constraints=[],
        actor_profiles=[],
    )

    state_2: Dict[str, Any] = {
        "seed_document": "BOJ Treasury liquidation scenario",
        "scenario_constraints": {"scenario": empty_scenario.model_dump()},
        "agent1_output": agent1_output,
        "agent2_output": None,
        "agent3_output": None,
        "agent4_output": None,
        "claim_ledger": list(agent1_claims),
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "agent1",
        "pipeline_status": "running",
    }

    state_2 = run_actor_modeling(state_2)

    if state_2["agent2_output"] is not None:
        ar_count = len(state_2["agent2_output"]["actor_responses"])
        if ar_count == 0:
            print("Path taken: agent produced output with empty actor_responses list")
        else:
            print(
                f"Path taken: agent produced {ar_count} actor response(s) despite "
                f"empty profiles (LLM inferred from upstream data)"
            )
    elif state_2["pipeline_status"] == "failed":
        print("Path taken: pipeline_status == 'failed' — agent could not model without profiles")
    else:
        print(f"Path taken: unexpected state — pipeline_status={state_2['pipeline_status']!r}")

    assert (
        (state_2["agent2_output"] is not None
         and isinstance(state_2["agent2_output"].get("actor_responses"), list))
        or state_2["pipeline_status"] == "failed"
    ), "Expected either valid output or failed status"
    print("assertion passed: no exception raised, graceful handling confirmed")
    print("  >>> Test 2 PASSED\n")

    print("All Agent 2 tests passed.")
