import json
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from mas_validation.schemas.claims import BaseClaim, BehavioralClaim
from mas_validation.validators.deterministic import ValidationCheckResult
from mas_validation.ledger import ClaimLedger
from mas_validation.llm_client import get_llm


CLAIM_ID_PATTERN = re.compile(r"claim_id:\s*[0-9a-f\-]{36}")


def _parse_and_validate_llm_response(
    raw_response: str, valid_classifications: list
) -> dict:
    try:
        parsed = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("response is not valid JSON")

    for key in ("classification", "justification"):
        if key not in parsed:
            raise ValueError(f"response missing required key: {key}")

    if parsed["classification"] not in valid_classifications:
        raise ValueError(
            f"classification {parsed['classification']!r} not in "
            f"{valid_classifications}"
        )

    if not CLAIM_ID_PATTERN.search(parsed["justification"]):
        raise ValueError("justification missing claim ID reference")

    return parsed


# ---------------------------------------------------------------------------
# Check 4: Reasoning chain continuity
# ---------------------------------------------------------------------------


def detect_discontinuity(
    premises: List[BaseClaim], reasoning: str
) -> ValidationCheckResult:
    system_prompt = (
        "You are a logic checker with no world knowledge. You only determine "
        "whether a conclusion follows from the provided premises. You have no "
        "access to external facts. Respond ONLY with a JSON object matching "
        "this exact schema: {classification: ENTAILED or PARTIAL or "
        "DISCONTINUOUS, unsupported_premises: list of strings, "
        "contradicted_claim_ids: list of claim_id uuid strings, justification: "
        "string that MUST contain at least one claim ID reference in the format "
        "claim_id: <uuid>}"
    )

    premise_lines = []
    for p in premises:
        fields = p.model_dump()
        premise_lines.append(
            f"claim_id: {p.claim_id} | type: {p.claim_type} | content: {json.dumps(fields, default=str)}"
        )

    user_text = "\n".join(premise_lines) + f"\n\nConclusion to evaluate: {reasoning}"

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    raw = response.content

    try:
        parsed = _parse_and_validate_llm_response(
            raw, ["ENTAILED", "PARTIAL", "DISCONTINUOUS"]
        )
    except ValueError as e:
        return ValidationCheckResult(
            check_number=4,
            check_name="reasoning_chain_continuity",
            status="rejected",
            compared_values={"error": str(e)},
            message=str(e),
        )

    status_map = {
        "ENTAILED": "passed",
        "PARTIAL": "flagged",
        "DISCONTINUOUS": "rejected",
    }
    status = status_map[parsed["classification"]]

    return ValidationCheckResult(
        check_number=4,
        check_name="reasoning_chain_continuity",
        status=status,
        compared_values=parsed,
        message=parsed["justification"],
    )


# ---------------------------------------------------------------------------
# Check 5: Hallucination propagation
# ---------------------------------------------------------------------------


def detect_propagation(
    upstream_claims: List[BaseClaim],
    downstream_reasoning: str,
    ledger: ClaimLedger,
) -> ValidationCheckResult:
    flagged_claims = ledger.get_by_validation_status("flagged")
    flagged_ids = {c["claim_id"] for c in flagged_claims}

    if not flagged_claims:
        return ValidationCheckResult(
            check_number=5,
            check_name="hallucination_propagation",
            status="passed",
            compared_values={},
            message="no flagged upstream claims to check",
        )

    system_prompt = (
        "You are a logic checker. I will provide premises and agent reasoning. "
        "Some premises are marked UNVERIFIED because they failed earlier "
        "validation. Determine if the agent treats an UNVERIFIED premise as "
        "settled fact without independent evaluation or acknowledgment of "
        "uncertainty. Respond ONLY with a JSON object matching this exact "
        "schema: {classification: PROPAGATED or UNCERTAIN_ACKNOWLEDGED or "
        "CLEAN, propagated_claim_ids: list of claim_id uuid strings that were "
        "propagated, justification: string that MUST contain at least one "
        "claim ID reference in the format claim_id: <uuid>}"
    )

    premise_lines = []
    for c in upstream_claims:
        tag = "[UNVERIFIED]" if c.claim_id in flagged_ids else "[VERIFIED]"
        fields = c.model_dump()
        premise_lines.append(
            f"{tag} claim_id: {c.claim_id} | type: {c.claim_type} | "
            f"content: {json.dumps(fields, default=str)}"
        )

    user_text = (
        "\n".join(premise_lines)
        + f"\n\nDownstream reasoning to evaluate: {downstream_reasoning}"
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ])
    raw = response.content

    try:
        parsed = _parse_and_validate_llm_response(
            raw, ["PROPAGATED", "UNCERTAIN_ACKNOWLEDGED", "CLEAN"]
        )
    except ValueError as e:
        return ValidationCheckResult(
            check_number=5,
            check_name="hallucination_propagation",
            status="rejected",
            compared_values={"error": str(e)},
            message=str(e),
        )

    status_map = {
        "PROPAGATED": "rejected",
        "UNCERTAIN_ACKNOWLEDGED": "flagged",
        "CLEAN": "passed",
    }
    status = status_map[parsed["classification"]]

    compared = dict(parsed)
    compared["flagged_claim_ids_checked"] = sorted(flagged_ids)

    return ValidationCheckResult(
        check_number=5,
        check_name="hallucination_propagation",
        status=status,
        compared_values=compared,
        message=parsed["justification"],
    )


if __name__ == "__main__":
    from mas_validation.schemas.claims import FactualClaim

    # ==================================================================
    # Test 1 — Entailment success
    # ==================================================================
    print("=== Test 1: Entailment success ===")

    premise_1 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.93,
        statement="BOJ holds $1.27T in US Treasury securities as of Q3 2024",
        entities=["Bank of Japan", "US Treasury"],
        parameters={"holdings_usd": 1.27e12, "quarter": "Q3 2024"},
    )
    premise_2 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.88,
        statement="Japan foreign reserves stand at $1.15T with a policy floor of $800B",
        entities=["Japan", "Ministry of Finance"],
        parameters={"reserves_usd": 1.15e12, "floor_usd": 8e11},
    )

    reasoning_entailed = (
        "Given that BOJ holds $1.27T in US Treasuries and Japan's total "
        "foreign reserves are $1.15T with an $800B floor, BOJ's Treasury "
        "holdings represent the majority of Japan's reserve assets, and any "
        "large-scale liquidation would quickly approach the $800B floor."
    )

    result_1 = detect_discontinuity([premise_1, premise_2], reasoning_entailed)
    print(f"  Status: {result_1.status}")
    print(f"  Message: {result_1.message}")
    print(f"  Compared: {json.dumps(result_1.compared_values, default=str, indent=2)}")
    assert result_1.status == "passed", f"Expected passed, got {result_1.status}"
    print("  >>> Test 1 PASSED\n")

    # ==================================================================
    # Test 2 — Discontinuity detection
    # ==================================================================
    print("=== Test 2: Discontinuity detection ===")

    reasoning_discontinuous = (
        "Capital reserves are fully available and unconstrained. BOJ can "
        "liquidate its entire portfolio without any limits or policy floors."
    )

    result_2 = detect_discontinuity([premise_1, premise_2], reasoning_discontinuous)
    print(f"  Status: {result_2.status}")
    print(f"  Message: {result_2.message}")
    print(f"  Compared: {json.dumps(result_2.compared_values, default=str, indent=2)}")
    assert result_2.status == "rejected", f"Expected rejected, got {result_2.status}"
    justification_2 = result_2.compared_values.get("justification", "")
    assert CLAIM_ID_PATTERN.search(justification_2), (
        "Expected claim ID reference in justification"
    )
    print("  >>> Test 2 PASSED\n")

    # ==================================================================
    # Test 3 — Propagation catch
    # ==================================================================
    print("=== Test 3: Propagation catch ===")

    ledger = ClaimLedger()
    flagged_fact = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="unverified_source",
        confidence_score=0.45,
        statement="BOJ has secretly reduced holdings to $400B",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 4e11},
    )
    ledger.add_claim(flagged_fact)
    ledger.update_validation_status(flagged_fact.claim_id, "flagged")

    behavioral_upstream = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:boj",
        confidence_score=0.70,
        actor_id="bank_of_japan",
        trigger_condition="USD/JPY breaches 170",
        predicted_action="decrease_reserves",
        action_magnitude=0.35,
        active_constraints=["foreign_reserve_floor_800B"],
    )
    ledger.add_claim(behavioral_upstream)

    downstream = (
        "Since BOJ holdings are confirmed at $400B — well below the $800B "
        "reserve floor — the liquidation has already breached the critical "
        "threshold. This is an established fact that requires immediate "
        "policy response."
    )

    result_3 = detect_propagation(
        [flagged_fact, behavioral_upstream], downstream, ledger
    )
    print(f"  Status: {result_3.status}")
    print(f"  Message: {result_3.message}")
    print(f"  Compared: {json.dumps(result_3.compared_values, default=str, indent=2)}")
    assert result_3.status == "rejected", f"Expected rejected, got {result_3.status}"
    print("  >>> Test 3 PASSED\n")

    # ==================================================================
    # Test 4 — Guardrail verification
    # ==================================================================
    print("=== Test 4: Guardrail verification ===")

    try:
        _parse_and_validate_llm_response(
            '{"classification": "ENTAILED", "justification": "the reasoning is sound"}',
            ["ENTAILED", "PARTIAL", "DISCONTINUOUS"],
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "justification missing claim ID reference" in str(e), (
            f"Unexpected error message: {e}"
        )
        print(f"  Caught expected ValueError: {e}")
    print("  >>> Test 4 PASSED\n")

    # ==================================================================
    print("All 4 LLM validator tests passed.")
