import json
import logging
from typing import Any, Dict

from mas_validation.schemas.agent_outputs import Agent1Output, Constraint
from mas_validation.schemas.claims import ClaimFactory
from mas_validation.ledger import ClaimLedger
from mas_validation.llm_client import get_llm

logger = logging.getLogger(__name__)


def _reconstruct_ledger(claim_list: list) -> ClaimLedger:
    ledger = ClaimLedger()
    ledger._claims = list(claim_list)
    ledger._next_index = len(claim_list)
    return ledger


def _build_prompt(seed_document: str) -> str:
    return f"""You are a structured document parser. You have no external world knowledge. You extract information only from the provided document. You never infer or assume facts not present in the text.

Output a JSON object matching this exact schema:
{{
  "event_type": "<string — type of event described>",
  "magnitude": <float between 0.0 and 1.0 — severity of the event>,
  "affected_sectors": ["<string>", ...],
  "affected_actors": ["<string>", ...],
  "active_constraints": [
    {{
      "constraint_id": "<string — unique identifier>",
      "description": "<string — what the constraint limits>",
      "affected_actor_ids": ["<string>", ...],
      "prohibited_actions": ["<string — one of: increase_reserves, decrease_reserves, increase_spending, decrease_spending, policy_tighten, policy_loosen, halt_operations, seek_external_support, maintain_status_quo>", ...]
    }}
  ],
  "extracted_claims": [
    {{
      "claim_id": "<new uuid4 string>",
      "claim_type": "<one of: factual, quantitative, causal, behavioral>",
      "agent_id": "agent_1",
      "pipeline_step": 1,
      "source": "seed_document",
      "confidence_score": <float between 0.0 and 1.0>,
      "validation_status": "pending",
      // For factual claims include: "statement" (string), "entities" (list of strings), "parameters" (dict)
      // For quantitative claims include: "metric" (string), "value" (float), "unit" (string), "source_claim_ids" (list of strings)
      // For causal claims include: "cause" (string), "effect" (string), "mechanism_category" (one of: policy_response, market_reaction, resource_depletion, cascade_failure, behavioral_adaptation, systemic_feedback), "conditions" (list of strings), "strength" (float), "supporting_claim_ids" (list of strings)
      // For behavioral claims include: "actor_id" (string), "trigger_condition" (string), "predicted_action" (one of: increase_reserves, decrease_reserves, increase_spending, decrease_spending, policy_tighten, policy_loosen, halt_operations, seek_external_support, maintain_status_quo), "action_magnitude" (float or null), "active_constraints" (list of strings)
    }}
  ]
}}

Extract information ONLY from the provided document. Do not use outside facts. If information is not present in the document, omit it or mark it with low confidence.

Assign confidence_score based on textual evidence. Direct quotes or explicit figures: 0.9 to 1.0. Logical inferences from stated facts: 0.6 to 0.8. Vague implications or indirect signals: 0.3 to 0.5. Do not default all scores to 1.0.

Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks. The JSON must match the schema above exactly.

--- DOCUMENT ---
{seed_document}
--- END DOCUMENT ---"""


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


def run_scenario_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    state["current_step"] = "agent1"

    prompt = _build_prompt(state["seed_document"])
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
        logger.warning("Agent 1 primary attempt — %s", last_error)
    except Exception as e:
        last_error = f"LLM call error: {e}"
        logger.error("Agent 1 primary attempt — %s", last_error)
        state["pipeline_status"] = "failed"
        return state

    if parsed_json is not None:
        try:
            validated_output = Agent1Output.model_validate(parsed_json)
        except Exception as e:
            last_error = f"Pydantic validation error: {e}"
            logger.warning("Agent 1 primary attempt — %s", last_error)
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
            validated_output = Agent1Output.model_validate(parsed_json)
        except json.JSONDecodeError as e:
            logger.error("Agent 1 retry — JSON parse error: %s", e)
        except Exception as e:
            logger.error("Agent 1 retry — validation error: %s", e)

    if validated_output is None:
        state["pipeline_status"] = "failed"
        logger.error("Agent 1 failed after retry — pipeline halted")
        return state

    # Success path
    output_dict = validated_output.model_dump()
    state["agent1_output"] = output_dict

    ledger = _reconstruct_ledger(state["claim_ledger"])
    for claim_dict in output_dict["extracted_claims"]:
        ledger._claims.append(claim_dict)
        ledger._next_index += 1
    state["claim_ledger"] = ledger.to_dict()

    state["pipeline_status"] = "running"
    logger.info(
        "Agent 1 completed — extracted %d claims",
        len(output_dict["extracted_claims"]),
    )
    return state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ==================================================================
    # Test 1 — Standard run
    # ==================================================================
    print("=== Test 1: Standard run ===")

    state_1: Dict[str, Any] = {
        "seed_document": (
            "The Bank of Japan announced an emergency liquidation of USD 400 billion "
            "in US Treasury securities following a currency crisis. The USD/JPY exchange "
            "rate breached 185, triggering automatic reserve drawdown protocols. Affected "
            "sectors include US bond markets, Japanese export industries, and Southeast "
            "Asian currency reserves. The liquidation is expected to increase US 10-year "
            "yields by 150 to 200 basis points within 30 days."
        ),
        "claim_ledger": [],
        "current_step": None,
        "pipeline_status": "pending",
        "scenario_constraints": {},
        "agent1_output": None,
        "agent2_output": None,
        "agent3_output": None,
        "agent4_output": None,
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
    }

    state_1 = run_scenario_analysis(state_1)

    assert state_1["agent1_output"] is not None, "agent1_output should not be None"
    print("assertion passed: agent1_output is not None")

    assert state_1["pipeline_status"] == "running", (
        f"Expected 'running', got {state_1['pipeline_status']!r}"
    )
    print("assertion passed: pipeline_status == 'running'")

    assert state_1["current_step"] == "agent1", (
        f"Expected 'agent1', got {state_1['current_step']!r}"
    )
    print("assertion passed: current_step == 'agent1'")

    assert len(state_1["claim_ledger"]) > 0, "claim_ledger should not be empty"
    print(f"assertion passed: len(claim_ledger) = {len(state_1['claim_ledger'])} > 0")

    assert len(state_1["claim_ledger"]) == len(state_1["agent1_output"]["extracted_claims"]), (
        f"Ledger count {len(state_1['claim_ledger'])} != "
        f"extracted_claims count {len(state_1['agent1_output']['extracted_claims'])}"
    )
    print(
        f"assertion passed: len(claim_ledger) == len(extracted_claims) == "
        f"{len(state_1['claim_ledger'])}"
    )

    claim_types = [c.get("claim_type", "unknown") for c in state_1["claim_ledger"]]
    print(f"Extracted {len(claim_types)} claims with types: {claim_types}")
    print("  >>> Test 1 PASSED\n")

    # ==================================================================
    # Test 2 — Graceful failure / low confidence
    # ==================================================================
    print("=== Test 2: Graceful failure ===")

    state_2: Dict[str, Any] = {
        "seed_document": "aaaaaaa bbbbb ccccc 12345 nonsense text with no financial content whatsoever.",
        "claim_ledger": [],
        "current_step": None,
        "pipeline_status": "pending",
        "scenario_constraints": {},
        "agent1_output": None,
        "agent2_output": None,
        "agent3_output": None,
        "agent4_output": None,
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
    }

    # Should not raise
    state_2 = run_scenario_analysis(state_2)

    if state_2["agent1_output"] is not None:
        scores = [
            c.get("confidence_score", 1.0)
            for c in state_2["agent1_output"]["extracted_claims"]
            if isinstance(c.get("confidence_score"), (int, float))
        ]
        avg_confidence = sum(scores) / len(scores) if scores else 0.0
        if avg_confidence < 0.7:
            print(
                f"Path taken: agent produced output with low avg confidence "
                f"({avg_confidence:.2f} < 0.7)"
            )
        else:
            print(
                f"Path taken: agent produced output with avg confidence "
                f"{avg_confidence:.2f} (higher than expected but no crash)"
            )
    elif state_2["pipeline_status"] == "failed":
        print("Path taken: pipeline_status == 'failed' — agent could not parse nonsense input")
    else:
        print(f"Path taken: unexpected state — pipeline_status={state_2['pipeline_status']!r}")

    assert state_2["agent1_output"] is not None or state_2["pipeline_status"] == "failed", (
        "Expected either valid output or failed status"
    )
    print("assertion passed: no exception raised, graceful handling confirmed")
    print("  >>> Test 2 PASSED\n")

    print("All Agent 1 tests passed.")
