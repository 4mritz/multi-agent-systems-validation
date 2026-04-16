import pytest

from mas_validation.ledger import ClaimLedger
from mas_validation.schemas.claims import BehavioralClaim, FactualClaim
from mas_validation.schemas.agent_outputs import Constraint
from mas_validation.schemas.scenario import (
    ActorProfile,
    ActorType,
    ConstraintType,
    Scenario,
    ScenarioConstraint,
    Severity,
)
from mas_validation.validators.deterministic import validate_claim


def _empty_scenario(**overrides) -> Scenario:
    defaults = dict(
        title="Test",
        description="Test scenario",
        seed_document_path="test.json",
        event_type="test",
        constraints=[],
        actor_profiles=[],
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def _valid_factual_dict(**overrides) -> dict:
    claim = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.9,
        statement="Test statement",
        entities=["Entity"],
        parameters={"value": 100.0},
    )
    d = claim.model_dump()
    d.update(overrides)
    return d


def test_schema_check_passes_valid_claim():
    results = validate_claim(_valid_factual_dict(), _empty_scenario(), ClaimLedger())
    assert results[0].check_name == "schema_validation"
    assert results[0].status == "passed"


def test_schema_check_rejects_invalid_claim():
    results = validate_claim(
        {"garbage": True},
        _empty_scenario(),
        ClaimLedger(),
    )
    assert results[0].check_name == "schema_validation"
    assert results[0].status == "rejected"


def test_schema_rejection_returns_early():
    results = validate_claim(
        {"garbage": True},
        _empty_scenario(),
        ClaimLedger(),
    )
    assert len(results) == 1


def test_quantitative_bound_hard_reject():
    scenario = _empty_scenario(
        constraints=[
            ScenarioConstraint(
                description="Min value",
                constraint_type=ConstraintType.quantitative_bound,
                affected_claim_types=["factual"],
                affected_actor_ids=[],
                parameters={"min_value": 200.0},
                severity=Severity.hard,
            ),
        ]
    )
    claim_dict = _valid_factual_dict()  # parameters={"value": 100.0}
    results = validate_claim(claim_dict, scenario, ClaimLedger())
    statuses = [r.status for r in results]
    assert "rejected" in statuses


def test_quantitative_bound_soft_flag():
    scenario = _empty_scenario(
        constraints=[
            ScenarioConstraint(
                description="Min value",
                constraint_type=ConstraintType.quantitative_bound,
                affected_claim_types=["factual"],
                affected_actor_ids=[],
                parameters={"min_value": 200.0},
                severity=Severity.soft,
            ),
        ]
    )
    claim_dict = _valid_factual_dict()
    results = validate_claim(claim_dict, scenario, ClaimLedger())
    statuses = [r.status for r in results]
    assert "flagged" in statuses


def test_quantitative_bound_passes_within_bounds():
    scenario = _empty_scenario(
        constraints=[
            ScenarioConstraint(
                description="Min value",
                constraint_type=ConstraintType.quantitative_bound,
                affected_claim_types=["factual"],
                affected_actor_ids=[],
                parameters={"min_value": 50.0},
                severity=Severity.hard,
            ),
        ]
    )
    claim_dict = _valid_factual_dict()  # value=100.0, above min=50.0
    results = validate_claim(claim_dict, scenario, ClaimLedger())
    statuses = [r.status for r in results]
    assert all(s == "passed" for s in statuses)


def test_behavioral_consistency_rejects_prohibited_action():
    constraint_id = "no_halt"
    scenario = _empty_scenario(
        constraints=[
            ScenarioConstraint(
                constraint_id=constraint_id,
                description="No halting",
                constraint_type=ConstraintType.behavioral_rule,
                affected_claim_types=["behavioral"],
                affected_actor_ids=["boj"],
                parameters={},
                severity=Severity.hard,
            ),
        ],
        actor_profiles=[
            ActorProfile(
                actor_id="boj",
                actor_name="BOJ",
                actor_type=ActorType.central_bank,
                behavioral_parameters={},
                constraints=[
                    Constraint(
                        constraint_id=constraint_id,
                        description="No halting",
                        affected_actor_ids=["boj"],
                        prohibited_actions=["halt_operations"],
                    ),
                ],
                decision_priorities=[],
            ),
        ],
    )
    claim = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="profile",
        confidence_score=0.7,
        actor_id="boj",
        trigger_condition="trigger",
        predicted_action="halt_operations",
        active_constraints=[constraint_id],
    )
    results = validate_claim(claim.model_dump(), scenario, ClaimLedger())
    statuses = [r.status for r in results]
    assert "rejected" in statuses


def test_behavioral_consistency_flags_unknown_actor():
    scenario = _empty_scenario(
        constraints=[
            ScenarioConstraint(
                description="Rule",
                constraint_type=ConstraintType.behavioral_rule,
                affected_claim_types=["behavioral"],
                affected_actor_ids=[],
                parameters={},
                severity=Severity.hard,
            ),
        ],
    )
    claim = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="profile",
        confidence_score=0.7,
        actor_id="unknown_actor",
        trigger_condition="trigger",
        predicted_action="maintain_status_quo",
        active_constraints=[],
    )
    results = validate_claim(claim.model_dump(), scenario, ClaimLedger())
    behavioral_check = [r for r in results if r.check_name == "behavioral_consistency"]
    assert len(behavioral_check) == 1
    assert behavioral_check[0].status == "flagged"


def test_ledger_contradiction_flags_numeric_mismatch():
    ledger = ClaimLedger()
    existing = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed",
        confidence_score=0.9,
        statement="BOJ holds $1.27T",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 1.27e12},
    )
    ledger.add_claim(existing)

    new_claim = FactualClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="model",
        confidence_score=0.75,
        statement="BOJ holdings at $0.9T",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 0.9e12},
    )
    results = validate_claim(new_claim.model_dump(), _empty_scenario(), ledger)
    ledger_checks = [r for r in results if r.check_name == "ledger_contradiction"]
    assert len(ledger_checks) == 1
    assert ledger_checks[0].status == "flagged"


def test_ledger_contradiction_passes_within_tolerance():
    ledger = ClaimLedger()
    existing = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed",
        confidence_score=0.9,
        statement="Holdings",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 1.0e12},
    )
    ledger.add_claim(existing)

    # 0.5% difference = within 1% tolerance
    new_claim = FactualClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="model",
        confidence_score=0.85,
        statement="Holdings",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 1.005e12},
    )
    results = validate_claim(new_claim.model_dump(), _empty_scenario(), ledger)
    ledger_checks = [r for r in results if r.check_name == "ledger_contradiction"]
    assert len(ledger_checks) == 1
    assert ledger_checks[0].status == "passed"


def test_all_checks_run_for_valid_claim():
    results = validate_claim(_valid_factual_dict(), _empty_scenario(), ClaimLedger())
    assert len(results) == 5
    check_numbers = [r.check_number for r in results]
    assert check_numbers == [1, 2, 3, 4, 5]
