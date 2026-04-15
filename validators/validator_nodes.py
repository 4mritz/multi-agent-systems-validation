import logging
from typing import Any, Dict, List

from mas_validation.state import PipelineState
from mas_validation.ledger import ClaimLedger
from mas_validation.validators.deterministic import validate_claim, ValidationCheckResult
from mas_validation.validators.llm_validator import detect_discontinuity, detect_propagation
from mas_validation.schemas.claims import BaseClaim, ClaimFactory
from mas_validation.schemas.agent_outputs import Agent1Output, Agent2Output, Agent3Output
from mas_validation.schemas.scenario import Scenario

logger = logging.getLogger(__name__)


def _reconstruct_ledger(state: PipelineState) -> ClaimLedger:
    ledger = ClaimLedger()
    ledger._claims = list(state["claim_ledger"])
    ledger._next_index = len(state["claim_ledger"])
    return ledger


def _worst_status(statuses: List[str]) -> str:
    if "rejected" in statuses:
        return "rejected"
    if "flagged" in statuses:
        return "flagged"
    return "passed"


def _execute_validation_handoff(
    state: PipelineState,
    handoff_key: str,
    agent_output_dict: dict,
    claims: List[dict],
    reasoning: str,
    scenario: Scenario,
) -> PipelineState:
    # Step 1 — reconstruct ledger
    ledger = _reconstruct_ledger(state)

    # Step 2 — deterministic validation
    all_results: List[ValidationCheckResult] = []
    for claim_dict in claims:
        det_results = validate_claim(claim_dict, scenario, ledger)
        all_results.extend(det_results)
        for r in det_results:
            if r.status in ("rejected", "flagged"):
                claim_id = claim_dict.get("claim_id", "unknown")
                logger.warning(
                    "Deterministic %s: claim %s check %d [%s] — %s",
                    r.status, claim_id, r.check_number, r.check_name, r.message,
                )

    # Step 3 — LLM validation
    parsed_claims: List[BaseClaim] = []
    for claim_dict in claims:
        try:
            parsed_claims.append(ClaimFactory.from_dict(claim_dict))
        except Exception:
            pass

    if parsed_claims:
        try:
            disc_result = detect_discontinuity(parsed_claims, reasoning)
        except Exception as e:
            logger.warning("LLM discontinuity check unavailable: %s", e)
            disc_result = ValidationCheckResult(
                check_number=4,
                check_name="reasoning_chain_continuity",
                status="flagged",
                compared_values={"error": str(e)},
                message=f"LLM unavailable — skipped: {e}",
            )
        all_results.append(disc_result)
        if disc_result.status in ("rejected", "flagged"):
            logger.warning(
                "LLM discontinuity %s: %s", disc_result.status, disc_result.message,
            )

        try:
            prop_result = detect_propagation(parsed_claims, reasoning, ledger)
        except Exception as e:
            logger.warning("LLM propagation check unavailable: %s", e)
            prop_result = ValidationCheckResult(
                check_number=5,
                check_name="hallucination_propagation",
                status="flagged",
                compared_values={"error": str(e)},
                message=f"LLM unavailable — skipped: {e}",
            )
        all_results.append(prop_result)
        if prop_result.status in ("rejected", "flagged"):
            logger.warning(
                "LLM propagation %s: %s", prop_result.status, prop_result.message,
            )

    # Step 4 — update ledger claim statuses
    for claim_dict in claims:
        cid = claim_dict.get("claim_id")
        if not cid:
            continue
        # Collect statuses from deterministic checks that ran for this claim
        claim_statuses = []
        det_results_for_claim = validate_claim(claim_dict, scenario, ledger)
        for r in det_results_for_claim:
            claim_statuses.append(r.status)
        # Also include LLM check statuses (apply to all claims)
        for r in all_results:
            if r.check_number in (4, 5):
                claim_statuses.append(r.status)
        if claim_statuses:
            worst = _worst_status(claim_statuses)
            try:
                ledger.update_validation_status(cid, worst)
            except ValueError:
                # Claim may not be in ledger yet — add it first
                ledger.add_claim(claim_dict)
                ledger.update_validation_status(cid, worst)

    state["claim_ledger"] = ledger.to_dict()

    # Step 5 — compute overall handoff status
    overall = _worst_status([r.status for r in all_results])

    # Step 6 — write back to state
    state["validation_results"][handoff_key].extend(
        [r.model_dump() for r in all_results]
    )

    if overall == "rejected":
        state["pipeline_status"] = "failed"
        state["fallback_flags"][handoff_key] = True
        rejection_messages = [
            r.message for r in all_results if r.status == "rejected"
        ]
        state["fallback_reasons"][handoff_key] = "; ".join(rejection_messages)
        logger.error(
            "Handoff %s REJECTED: %s", handoff_key, state["fallback_reasons"][handoff_key],
        )

    return state


def validate_1_to_2(state: PipelineState) -> Dict[str, Any]:
    agent1_output = state["agent1_output"]
    claims = agent1_output["extracted_claims"]
    reasoning = " ".join(
        c.get("statement", "") for c in claims
    )

    sc_data = state["scenario_constraints"]
    if isinstance(sc_data, dict) and "scenario" in sc_data:
        scenario = Scenario(**sc_data["scenario"])
    else:
        scenario = Scenario(
            title="empty",
            description="empty",
            seed_document_path="",
            event_type="",
            constraints=[],
            actor_profiles=[],
        )

    return _execute_validation_handoff(
        state, "1_to_2", agent1_output, claims, reasoning, scenario,
    )


def validate_2_to_3(state: PipelineState) -> Dict[str, Any]:
    agent2_output = state["agent2_output"]
    claims = agent2_output["extracted_claims"]
    reasoning = " ".join(
        r.get("response_summary", "") for r in agent2_output.get("actor_responses", [])
    )

    sc_data = state["scenario_constraints"]
    if isinstance(sc_data, dict) and "scenario" in sc_data:
        scenario = Scenario(**sc_data["scenario"])
    else:
        scenario = Scenario(
            title="empty",
            description="empty",
            seed_document_path="",
            event_type="",
            constraints=[],
            actor_profiles=[],
        )

    return _execute_validation_handoff(
        state, "2_to_3", agent2_output, claims, reasoning, scenario,
    )


def validate_3_to_4(state: PipelineState) -> Dict[str, Any]:
    agent3_output = state["agent3_output"]
    claims = agent3_output["extracted_claims"]
    reasoning = " ".join(
        e.get("description", "") for e in agent3_output.get("systemic_effects", [])
    )

    sc_data = state["scenario_constraints"]
    if isinstance(sc_data, dict) and "scenario" in sc_data:
        scenario = Scenario(**sc_data["scenario"])
    else:
        scenario = Scenario(
            title="empty",
            description="empty",
            seed_document_path="",
            event_type="",
            constraints=[],
            actor_profiles=[],
        )

    return _execute_validation_handoff(
        state, "3_to_4", agent3_output, claims, reasoning, scenario,
    )


if __name__ == "__main__":
    import json

    from mas_validation.schemas.claims import FactualClaim
    from mas_validation.schemas.agent_outputs import Constraint
    from mas_validation.schemas.scenario import (
        ConstraintType,
        ScenarioConstraint,
        Severity,
    )

    logging.basicConfig(level=logging.WARNING)

    # --- Build scenario ---
    scenario = Scenario(
        title="BOJ Treasury Liquidation — Validator Node Test",
        description="Test scenario for validator node handoff",
        seed_document_path="experiments/scenarios/boj_liquidation.json",
        event_type="sovereign_debt_liquidation",
        constraints=[
            ScenarioConstraint(
                description="Minimum liquidity ratio for all financial actors",
                constraint_type=ConstraintType.quantitative_bound,
                affected_claim_types=["factual"],
                affected_actor_ids=["bank_of_japan"],
                parameters={"min_liquidity_ratio": 0.15},
                severity=Severity.hard,
            ),
        ],
        actor_profiles=[],
    )

    # --- Build claims ---
    poisoned_claim = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.90,
        statement="BOJ liquidity ratio dropped to 0.08 under sell-off pressure",
        entities=["Bank of Japan"],
        parameters={"liquidity_ratio": 0.08},
    )

    valid_claim = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.92,
        statement="Japan foreign reserves stand at $1.15T",
        entities=["Japan", "Ministry of Finance"],
        parameters={"liquidity_ratio": 0.20},
    )

    # --- Build agent1_output ---
    agent1_output = {
        "event_type": "sovereign_debt_liquidation",
        "magnitude": 0.85,
        "affected_sectors": ["government_bonds", "forex"],
        "affected_actors": ["bank_of_japan"],
        "active_constraints": [],
        "extracted_claims": [poisoned_claim.model_dump(), valid_claim.model_dump()],
    }

    # --- Build state ---
    state: PipelineState = {
        "seed_document": "BOJ Treasury liquidation scenario seed document",
        "scenario_constraints": {"scenario": scenario.model_dump()},
        "agent1_output": agent1_output,
        "agent2_output": None,
        "agent3_output": None,
        "agent4_output": None,
        "claim_ledger": [],
        "validation_results": {"1_to_2": [], "2_to_3": [], "3_to_4": []},
        "fallback_flags": {"1_to_2": False, "2_to_3": False, "3_to_4": False},
        "fallback_reasons": {"1_to_2": None, "2_to_3": None, "3_to_4": None},
        "current_step": "validator_1_to_2",
        "pipeline_status": "running",
    }

    # --- Execute ---
    result = validate_1_to_2(state)

    # --- Assertions ---
    assert result["pipeline_status"] == "failed", (
        f"Expected 'failed', got {result['pipeline_status']!r}"
    )
    print("assertion passed: pipeline_status == 'failed'")

    assert result["fallback_flags"]["1_to_2"] is True, (
        f"Expected True, got {result['fallback_flags']['1_to_2']!r}"
    )
    print("assertion passed: fallback_flags['1_to_2'] == True")

    assert result["fallback_reasons"]["1_to_2"] is not None, (
        "Expected a rejection reason string"
    )
    assert len(result["fallback_reasons"]["1_to_2"]) > 0, (
        "Expected non-empty rejection reason"
    )
    print(f"assertion passed: fallback_reasons['1_to_2'] = {result['fallback_reasons']['1_to_2']!r}")

    poisoned_in_ledger = None
    for c in result["claim_ledger"]:
        if c["claim_id"] == poisoned_claim.claim_id:
            poisoned_in_ledger = c
            break
    assert poisoned_in_ledger is not None, "Poisoned claim not found in ledger"
    assert poisoned_in_ledger["validation_status"] == "rejected", (
        f"Expected 'rejected', got {poisoned_in_ledger['validation_status']!r}"
    )
    print(f"assertion passed: poisoned claim {poisoned_claim.claim_id} validation_status == 'rejected'")

    # --- Print full validation results ---
    print("\n=== Validation results for 1_to_2 ===")
    for entry in result["validation_results"]["1_to_2"]:
        print(json.dumps(entry, indent=2, default=str))
        print()

    print("All validator node assertions passed.")
