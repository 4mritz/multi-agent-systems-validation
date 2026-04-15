from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from mas_validation.schemas.claims import BaseClaim, BehavioralClaim, ClaimFactory
from mas_validation.schemas.scenario import Scenario
from mas_validation.ledger import ClaimLedger


TOLERANCE = 0.01

_VALID_STATUSES = {"passed", "flagged", "rejected"}


class ValidationCheckResult(BaseModel):
    check_number: int
    check_name: str
    status: str
    compared_values: Dict[str, Any] = Field(default_factory=dict)
    message: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in _VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(_VALID_STATUSES)}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Private check functions
# ---------------------------------------------------------------------------


def _check_schema(claim_data: dict) -> ValidationCheckResult:
    """Check 1: Schema validation via ClaimFactory."""
    try:
        parsed = ClaimFactory.from_dict(claim_data)
        return ValidationCheckResult(
            check_number=1,
            check_name="schema_validation",
            status="passed",
            compared_values={"claim_type": parsed.claim_type},
            message=f"Claim parsed successfully as {type(parsed).__name__}",
        )
    except Exception as e:
        return ValidationCheckResult(
            check_number=1,
            check_name="schema_validation",
            status="rejected",
            compared_values={"error": str(e)},
            message=f"Schema validation failed: {e}",
        )


def _check_scope(
    claim: BaseClaim, scenario: Scenario
) -> tuple[ValidationCheckResult, list]:
    """Check 2: Find matching ScenarioConstraints by claim_type."""
    matching = [
        sc for sc in scenario.constraints
        if claim.claim_type in sc.affected_claim_types
    ]
    if not matching:
        return (
            ValidationCheckResult(
                check_number=2,
                check_name="scope_check",
                status="passed",
                compared_values={"matching_constraints": 0},
                message="No scenario constraints apply to this claim type",
            ),
            [],
        )
    return (
        ValidationCheckResult(
            check_number=2,
            check_name="scope_check",
            status="passed",
            compared_values={
                "matching_constraints": len(matching),
                "constraint_ids": [sc.constraint_id for sc in matching],
            },
            message=f"{len(matching)} scenario constraint(s) match claim type '{claim.claim_type}'",
        ),
        matching,
    )


def _check_quantitative_bounds(
    claim: BaseClaim, matching_constraints: list
) -> ValidationCheckResult:
    """Check 3: Quantitative bound enforcement."""
    claim_params = getattr(claim, "parameters", {}) or {}
    worst_status = "passed"
    compared: Dict[str, Any] = {}
    violation_msg = ""

    for sc in matching_constraints:
        if sc.constraint_type.value != "quantitative_bound":
            continue

        for param_key, param_val in sc.parameters.items():
            if param_key.startswith("min_"):
                claim_key = param_key[4:]  # strip "min_"
                if claim_key in claim_params:
                    claim_val = claim_params[claim_key]
                    if isinstance(claim_val, (int, float)) and isinstance(param_val, (int, float)):
                        compared[f"constraint_{param_key}"] = param_val
                        compared[f"claim_{claim_key}"] = claim_val
                        if claim_val < param_val:
                            severity = sc.severity.value
                            new_status = "rejected" if severity == "hard" else "flagged"
                            violation_msg = (
                                f"{claim_key}={claim_val} below min bound {param_val} "
                                f"(severity={severity})"
                            )
                            if _status_rank(new_status) > _status_rank(worst_status):
                                worst_status = new_status

            elif param_key.startswith("max_"):
                claim_key = param_key[4:]  # strip "max_"
                if claim_key in claim_params:
                    claim_val = claim_params[claim_key]
                    if isinstance(claim_val, (int, float)) and isinstance(param_val, (int, float)):
                        compared[f"constraint_{param_key}"] = param_val
                        compared[f"claim_{claim_key}"] = claim_val
                        if claim_val > param_val:
                            severity = sc.severity.value
                            new_status = "rejected" if severity == "hard" else "flagged"
                            violation_msg = (
                                f"{claim_key}={claim_val} above max bound {param_val} "
                                f"(severity={severity})"
                            )
                            if _status_rank(new_status) > _status_rank(worst_status):
                                worst_status = new_status

    if worst_status == "passed":
        return ValidationCheckResult(
            check_number=3,
            check_name="quantitative_bounds",
            status="passed",
            compared_values=compared,
            message="All quantitative bounds satisfied",
        )
    return ValidationCheckResult(
        check_number=3,
        check_name="quantitative_bounds",
        status=worst_status,
        compared_values=compared,
        message=f"Quantitative bound violation: {violation_msg}",
    )


def _check_behavioral_consistency(
    claim: BaseClaim, scenario: Scenario
) -> ValidationCheckResult:
    """Check 4: Behavioral claim vs actor profile prohibited actions."""
    if not isinstance(claim, BehavioralClaim):
        return ValidationCheckResult(
            check_number=4,
            check_name="behavioral_consistency",
            status="passed",
            compared_values={},
            message="Not a behavioral claim — skipped",
        )

    actor_profile = None
    for ap in scenario.actor_profiles:
        if ap.actor_id == claim.actor_id:
            actor_profile = ap
            break

    if actor_profile is None:
        return ValidationCheckResult(
            check_number=4,
            check_name="behavioral_consistency",
            status="flagged",
            compared_values={"actor_id": claim.actor_id},
            message=f"Actor profile not found for actor_id '{claim.actor_id}'",
        )

    # Build a map from actor constraint_id to scenario constraint severity
    scenario_constraint_map: Dict[str, str] = {
        sc.constraint_id: sc.severity.value for sc in scenario.constraints
    }

    for actor_constraint in actor_profile.constraints:
        if claim.predicted_action in actor_constraint.prohibited_actions:
            severity = scenario_constraint_map.get(actor_constraint.constraint_id, "soft")
            status = "rejected" if severity == "hard" else "flagged"
            return ValidationCheckResult(
                check_number=4,
                check_name="behavioral_consistency",
                status=status,
                compared_values={
                    "predicted_action": claim.predicted_action,
                    "violated_constraint_id": actor_constraint.constraint_id,
                },
                message=(
                    f"Predicted action '{claim.predicted_action}' is prohibited by "
                    f"constraint '{actor_constraint.constraint_id}' (severity={severity})"
                ),
            )

    return ValidationCheckResult(
        check_number=4,
        check_name="behavioral_consistency",
        status="passed",
        compared_values={"predicted_action": claim.predicted_action},
        message="Behavioral claim consistent with actor profile constraints",
    )


def _check_ledger_contradiction(
    claim: BaseClaim, ledger: ClaimLedger
) -> ValidationCheckResult:
    """Check 5: Contradiction detection against existing ledger claims."""
    entities = getattr(claim, "entities", [])
    existing = ledger.get_contradictable_claims(claim.claim_type, entities)

    claim_params = getattr(claim, "parameters", {}) or {}
    compared: Dict[str, Any] = {}

    for existing_claim in existing:
        existing_params = existing_claim.get("parameters", {}) or {}
        for key in set(claim_params) & set(existing_params):
            new_val = claim_params[key]
            old_val = existing_params[key]
            if isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
                if old_val != 0:
                    rel_diff = abs(new_val - old_val) / abs(old_val)
                    if rel_diff > TOLERANCE:
                        compared[f"existing_{key}"] = old_val
                        compared[f"new_{key}"] = new_val
                        compared["relative_difference"] = round(rel_diff, 6)
                        return ValidationCheckResult(
                            check_number=5,
                            check_name="ledger_contradiction",
                            status="flagged",
                            compared_values=compared,
                            message=(
                                f"Parameter '{key}' differs by {rel_diff:.2%} from "
                                f"existing ledger claim (tolerance={TOLERANCE:.2%})"
                            ),
                        )

    return ValidationCheckResult(
        check_number=5,
        check_name="ledger_contradiction",
        status="passed",
        compared_values=compared,
        message="No contradictions found in ledger",
    )


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------


def validate_claim(
    claim_data: dict, scenario: Scenario, ledger: ClaimLedger
) -> List[ValidationCheckResult]:
    """Run all five deterministic checks and return results list."""
    results: List[ValidationCheckResult] = []

    # Check 1: Schema
    schema_result = _check_schema(claim_data)
    results.append(schema_result)
    if schema_result.status == "rejected":
        return results

    claim = ClaimFactory.from_dict(claim_data)

    # Check 2: Scope
    scope_result, matching_constraints = _check_scope(claim, scenario)
    results.append(scope_result)

    # Check 3: Quantitative bounds
    bounds_result = _check_quantitative_bounds(claim, matching_constraints)
    results.append(bounds_result)

    # Check 4: Behavioral consistency
    behavioral_result = _check_behavioral_consistency(claim, scenario)
    results.append(behavioral_result)

    # Check 5: Ledger contradiction (always runs)
    ledger_result = _check_ledger_contradiction(claim, ledger)
    results.append(ledger_result)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_RANK = {"passed": 0, "flagged": 1, "rejected": 2}


def _status_rank(status: str) -> int:
    return _STATUS_RANK.get(status, -1)


if __name__ == "__main__":
    import json

    from mas_validation.schemas.claims import FactualClaim
    from mas_validation.schemas.agent_outputs import Constraint
    from mas_validation.schemas.scenario import (
        ActorProfile,
        ActorType,
        ConstraintType,
        ScenarioConstraint,
        Severity,
    )

    def print_results(results: List[ValidationCheckResult]) -> None:
        for r in results:
            print(f"  Check {r.check_number} [{r.check_name}]: {r.status}")
            if r.compared_values:
                print(f"    compared_values: {json.dumps(r.compared_values, default=str)}")
            print(f"    message: {r.message}")

    # ==================================================================
    # Test 1 — Quantitative reject
    # ==================================================================
    print("=== Test 1: Quantitative reject ===")

    scenario_1 = Scenario(
        title="BOJ Treasury Liquidation — Quantitative Test",
        description="Test scenario for quantitative bound rejection",
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

    claim_1 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.90,
        statement="BOJ liquidity ratio dropped to 0.08 under sell-off pressure",
        entities=["Bank of Japan"],
        parameters={"liquidity_ratio": 0.08},
    )

    ledger_1 = ClaimLedger()
    results_1 = validate_claim(claim_1.model_dump(), scenario_1, ledger_1)
    print_results(results_1)
    assert any(r.status == "rejected" for r in results_1), "Expected at least one rejected"
    print("  >>> Test 1 PASSED\n")

    # ==================================================================
    # Test 2 — Behavioral reject
    # ==================================================================
    print("=== Test 2: Behavioral reject ===")

    halt_constraint_id = "boj_no_halt_operations"

    scenario_2 = Scenario(
        title="BOJ Treasury Liquidation — Behavioral Test",
        description="Test scenario for behavioral consistency rejection",
        seed_document_path="experiments/scenarios/boj_liquidation.json",
        event_type="sovereign_debt_liquidation",
        constraints=[
            ScenarioConstraint(
                constraint_id=halt_constraint_id,
                description="BOJ cannot halt operations under any scenario",
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
                    "risk_tolerance": 0.3,
                    "reserve_floor_usd": 8e11,
                },
                constraints=[
                    Constraint(
                        constraint_id=halt_constraint_id,
                        description="BOJ cannot halt operations under any scenario",
                        affected_actor_ids=["bank_of_japan"],
                        prohibited_actions=["halt_operations"],
                    ),
                ],
                decision_priorities=["yen_stability", "reserve_adequacy"],
            ),
        ],
    )

    claim_2 = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:boj",
        confidence_score=0.60,
        actor_id="bank_of_japan",
        trigger_condition="Extreme yen depreciation beyond 180 USD/JPY",
        predicted_action="halt_operations",
        action_magnitude=0.90,
        active_constraints=[halt_constraint_id],
    )

    ledger_2 = ClaimLedger()
    results_2 = validate_claim(claim_2.model_dump(), scenario_2, ledger_2)
    print_results(results_2)
    assert any(r.status == "rejected" for r in results_2), "Expected at least one rejected"
    print("  >>> Test 2 PASSED\n")

    # ==================================================================
    # Test 3 — Ledger flag
    # ==================================================================
    print("=== Test 3: Ledger flag ===")

    scenario_3 = Scenario(
        title="BOJ Treasury Liquidation — Ledger Test",
        description="Test scenario for ledger contradiction flagging",
        seed_document_path="experiments/scenarios/boj_liquidation.json",
        event_type="sovereign_debt_liquidation",
        constraints=[],
        actor_profiles=[],
    )

    existing_claim = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.92,
        statement="BOJ holds $1.27T in US Treasury securities",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 1.27e12},
    )

    ledger_3 = ClaimLedger()
    ledger_3.add_claim(existing_claim)

    new_claim = FactualClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="impact_model",
        confidence_score=0.75,
        statement="BOJ holdings reduced to $0.90T after liquidation",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 0.90e12},
    )

    results_3 = validate_claim(new_claim.model_dump(), scenario_3, ledger_3)
    print_results(results_3)
    assert any(r.status == "flagged" for r in results_3), "Expected at least one flagged"
    print("  >>> Test 3 PASSED\n")

    # ==================================================================
    print("All 3 deterministic validator tests passed.")
