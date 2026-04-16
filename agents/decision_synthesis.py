import json
import logging
import statistics
from typing import Any, Dict, List

from pydantic import ValidationError

from mas_validation.schemas.agent_outputs import Agent4Output
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


def _build_prompt(
    agent3_output: dict, upstream_claims: list, flagged_claims: list
) -> str:
    flagged_section = (
        f"--- FLAGGED CLAIMS REQUIRING EXPLICIT MENTION IN flagged_uncertainties ---\n"
        f"{json.dumps(flagged_claims, indent=2, default=str)}\n"
        f"--- END FLAGGED CLAIMS ---"
        if flagged_claims
        else "FLAGGED CLAIMS: None"
    )

    return f"""You are the Chief Risk Officer and final synthesis lead for a multi-agent economic simulation. You produce the final prediction report. You are the last audit point in the pipeline. You reason only from the provided evidence and flagged claims.

Output a JSON object matching this exact schema:
{{
  "executive_summary": "<string — high level synthesis of the simulation findings>",
  "key_findings": [
    {{
      "finding_id": "<new uuid4 string>",
      "description": "<string — description of the finding>",
      "confidence_score": <float between 0.0 and 1.0>,
      "supporting_claim_ids": ["<claim_id uuid string from upstream claims>", ...]
    }}
  ],
  "flagged_uncertainties": [
    "<string — each string describes a claim that was flagged by the validation layer>"
  ],
  "overall_confidence": 0.0,
  "extracted_claims": [<any additional synthesis claims>]
}}

You MUST review the FLAGGED CLAIMS section below. Every flagged claim must appear as a separate entry in flagged_uncertainties. Do not summarize multiple flagged claims into one entry. Do not omit any flagged claim. This is a mandatory audit requirement.

Every KeyFinding must have at least one supporting_claim_id referencing a claim_id from the provided upstream claims. Do not invent claim IDs. Copy them exactly from the provided list.

Assign confidence_score based on evidence strength. Direct causal relationships with strong upstream support: 0.7 to 1.0. Inferred second-order effects: 0.4 to 0.7. Speculative feedback loops: 0.2 to 0.4. Do not default all scores to 1.0.

Respond ONLY with a valid JSON object. No explanation, no markdown, no code blocks.

--- AGENT 3 OUTPUT ---
{json.dumps(agent3_output, indent=2, default=str)}
--- END AGENT 3 OUTPUT ---

--- UPSTREAM CLAIMS FROM LEDGER (use these claim_ids in supporting_claim_ids) ---
{json.dumps(upstream_claims, indent=2, default=str)}
--- END UPSTREAM CLAIMS ---

{flagged_section}"""


def run_decision_synthesis(state: Dict[str, Any]) -> Dict[str, Any]:
    state["current_step"] = "agent4"

    if state.get("agent3_output") is None:
        logger.error("Agent 4 — agent3_output is None, cannot proceed")
        state["pipeline_status"] = "failed"
        return state

    ledger = _reconstruct_ledger(state["claim_ledger"])
    upstream_claims = ledger.get_by_agent("agent_3")
    flagged_claims = ledger.get_by_validation_status("flagged")
    logger.info("Agent 4 — %d flagged claims passed for audit", len(flagged_claims))

    prompt = _build_prompt(state["agent3_output"], upstream_claims, flagged_claims)
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
        logger.warning("Agent 4 primary attempt — %s", last_error)
    except Exception as e:
        last_error = f"LLM call error: {e}"
        logger.error("Agent 4 primary attempt — %s", last_error)
        state["pipeline_status"] = "failed"
        return state

    if parsed_json is not None:
        try:
            validated_output = Agent4Output.model_validate(parsed_json)
        except ValidationError as e:
            last_error = f"Pydantic validation error: {e}"
            logger.warning("Agent 4 primary attempt — %s", last_error)
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
            validated_output = Agent4Output.model_validate(parsed_json)
        except json.JSONDecodeError as e:
            logger.error("Agent 4 retry — JSON parse error: %s", e)
        except Exception as e:
            logger.error("Agent 4 retry — validation error: %s", e)

    if validated_output is None:
        state["pipeline_status"] = "failed"
        logger.error("Agent 4 failed after retry — pipeline halted")
        return state

    # Success path
    output_dict = validated_output.model_dump()

    # Compute overall_confidence as mean of key_findings confidence scores
    findings = output_dict["key_findings"]
    if findings:
        computed_confidence = statistics.mean(
            f["confidence_score"] for f in findings
        )
    else:
        computed_confidence = 0.0
    output_dict["overall_confidence"] = computed_confidence

    state["agent4_output"] = output_dict

    total_claims_added = 0
    for claim_dict in output_dict["extracted_claims"]:
        ledger._claims.append(claim_dict)
        ledger._next_index += 1
        total_claims_added += 1

    state["claim_ledger"] = ledger.to_dict()
    state["pipeline_status"] = "completed"

    logger.info(
        "Agent 4 completed — %d key findings, overall_confidence=%.3f, "
        "%d flagged uncertainties surfaced, %d synthesis claims added",
        len(findings),
        computed_confidence,
        len(output_dict["flagged_uncertainties"]),
        total_claims_added,
    )
    return state


if __name__ == "__main__":
    from mas_validation.schemas.claims import CausalClaim, FactualClaim, QuantitativeClaim

    logging.basicConfig(level=logging.INFO)

    # ==================================================================
    # Shared fixtures — realistic Agent 3 output for BOJ scenario
    # ==================================================================

    causal_1 = CausalClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="impact_model",
        confidence_score=0.71,
        cause="Rapid BOJ liquidation of US Treasuries",
        effect="US 10Y yield spike of 150-250bps within 72 hours",
        mechanism_category="market_reaction",
        conditions=[
            "liquidation volume exceeds $200B in 30 days",
            "no coordinated central bank intervention",
        ],
        strength=0.74,
        supporting_claim_ids=[],
    )

    quantitative_1 = QuantitativeClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="impact_model",
        confidence_score=0.58,
        metric="us_10y_yield_spike",
        value=2.0,
        unit="percentage_points",
        source_claim_ids=[causal_1.claim_id],
    )

    flagged_factual = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.45,
        statement="BOJ holdings_usd value inconsistent with prior ledger entry",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 4e11},
    )

    agent3_output = {
        "systemic_effects": [
            {
                "effect_id": "eff-001",
                "description": "US Treasury market dislocation with 150-250bps yield spike",
                "magnitude": 0.85,
                "affected_sectors": ["government_bonds", "forex", "money_markets"],
                "cause_chain": [causal_1.claim_id],
                "second_order_effects": [
                    "Global risk-off sentiment drives flight from EM debt",
                ],
            },
            {
                "effect_id": "eff-002",
                "description": "US housing market contraction of 8-15% over 6 months",
                "magnitude": 0.65,
                "affected_sectors": ["housing", "banking"],
                "cause_chain": [causal_1.claim_id, quantitative_1.claim_id],
                "second_order_effects": [
                    "Regional bank failures accelerate due to MBS exposure",
                ],
            },
        ],
        "extracted_claims": [causal_1.model_dump(), quantitative_1.model_dump()],
    }

    # Build ledger: two passed (agent_3), one flagged
    causal_1_dict = causal_1.model_dump()
    causal_1_dict["validation_status"] = "passed"
    quantitative_1_dict = quantitative_1.model_dump()
    quantitative_1_dict["validation_status"] = "passed"
    flagged_dict = flagged_factual.model_dump()
    flagged_dict["validation_status"] = "flagged"

    claim_ledger = [causal_1_dict, quantitative_1_dict, flagged_dict]

    # ==================================================================
    # Test 1 — Standard run with flagged claims
    # ==================================================================
    print("=== Test 1: Standard run with flagged claims ===")

    state_1: Dict[str, Any] = {
        "seed_document": "BOJ Treasury liquidation scenario",
        "scenario_constraints": {},
        "agent1_output": {"extracted_claims": []},
        "agent2_output": {"actor_responses": [], "extracted_claims": []},
        "agent3_output": agent3_output,
        "agent4_output": None,
        "claim_ledger": list(claim_ledger),
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "agent3",
        "pipeline_status": "running",
    }

    state_1 = run_decision_synthesis(state_1)

    assert state_1["agent4_output"] is not None, "agent4_output should not be None"
    print("assertion passed: agent4_output is not None")

    assert len(state_1["agent4_output"]["executive_summary"]) > 0, (
        "executive_summary should be non-empty"
    )
    print("assertion passed: executive_summary is non-empty")

    kf = state_1["agent4_output"]["key_findings"]
    assert all(len(f["supporting_claim_ids"]) > 0 for f in kf), (
        "Every key finding must have at least one supporting_claim_id"
    )
    print(f"assertion passed: all {len(kf)} key findings have supporting_claim_ids")

    if kf:
        expected_mean = statistics.mean(f["confidence_score"] for f in kf)
    else:
        expected_mean = 0.0
    actual_confidence = state_1["agent4_output"]["overall_confidence"]
    assert abs(actual_confidence - expected_mean) < 0.001, (
        f"overall_confidence {actual_confidence} != mean {expected_mean}"
    )
    print(
        f"assertion passed: overall_confidence={actual_confidence:.4f} matches "
        f"computed mean={expected_mean:.4f}"
    )

    fu = state_1["agent4_output"]["flagged_uncertainties"]
    assert len(fu) > 0, "flagged_uncertainties should not be empty"
    print(f"assertion passed: len(flagged_uncertainties) = {len(fu)} > 0")

    print(f"\nFinal overall_confidence: {actual_confidence:.4f}")
    print("Flagged uncertainties:")
    for i, u in enumerate(fu):
        print(f"  [{i}] {u}")
    print("  >>> Test 1 PASSED\n")

    # ==================================================================
    # Test 2 — None guard
    # ==================================================================
    print("=== Test 2: None guard ===")

    state_2: Dict[str, Any] = {
        "seed_document": "BOJ Treasury liquidation scenario",
        "scenario_constraints": {},
        "agent1_output": {"extracted_claims": []},
        "agent2_output": {"actor_responses": [], "extracted_claims": []},
        "agent3_output": None,
        "agent4_output": None,
        "claim_ledger": [],
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "agent3",
        "pipeline_status": "running",
    }

    state_2 = run_decision_synthesis(state_2)

    assert state_2["pipeline_status"] == "failed", (
        f"Expected 'failed', got {state_2['pipeline_status']!r}"
    )
    print("assertion passed: pipeline_status == 'failed'")
    print("assertion passed: no exception raised")
    print("  >>> Test 2 PASSED\n")

    print("All Agent 4 tests passed.")
