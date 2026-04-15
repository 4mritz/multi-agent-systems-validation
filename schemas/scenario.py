from datetime import datetime
from enum import Enum
from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from mas_validation.schemas.agent_outputs import Constraint


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActorType(str, Enum):
    central_bank = "central_bank"
    government = "government"
    corporation = "corporation"
    financial_institution = "financial_institution"
    household_sector = "household_sector"
    international_org = "international_org"


class ConstraintType(str, Enum):
    quantitative_bound = "quantitative_bound"
    behavioral_rule = "behavioral_rule"
    causal_invariant = "causal_invariant"


class Severity(str, Enum):
    hard = "hard"
    soft = "soft"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ActorProfile(BaseModel):
    """Defines the identity and behavioral parameters of a simulation actor.
    Actor constraints are checked by the deterministic validator against
    BehavioralClaim outputs from Agent 2."""

    actor_id: str = Field(default_factory=lambda: str(uuid4()))
    actor_name: str
    actor_type: ActorType
    behavioral_parameters: Dict[str, Any]
    constraints: List[Constraint]
    decision_priorities: List[str]


class ScenarioConstraint(BaseModel):
    """Defines a ground truth constraint that must hold across the pipeline.
    Hard constraints trigger reject on violation. Soft constraints trigger flag.
    Used by the deterministic validator to check Agent outputs against scenario
    ground truth."""

    constraint_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    constraint_type: ConstraintType
    affected_claim_types: List[str]
    affected_actor_ids: List[str]
    parameters: Dict[str, Any]
    severity: Severity


class Scenario(BaseModel):
    """Complete scenario definition including ground truth constraints and actor
    profiles. The injected_errors list is empty for standard runs and populated
    for adversarial test runs."""

    scenario_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    seed_document_path: str
    event_type: str
    constraints: List[ScenarioConstraint]
    actor_profiles: List[ActorProfile]
    injected_errors: List[dict] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Instantiate a complete BOJ Treasury Liquidation scenario
    # ------------------------------------------------------------------

    boj_profile = ActorProfile(
        actor_id="bank_of_japan",
        actor_name="Bank of Japan",
        actor_type=ActorType.central_bank,
        behavioral_parameters={
            "foreign_reserve_total_usd": 1.15e12,
            "reserve_floor_usd": 8e11,
            "us_treasury_holdings_usd": 1.27e12,
            "risk_tolerance": 0.3,
            "liquidation_rate_cap_monthly_usd": 2e11,
            "policy_mandate": "price_stability_and_yen_defense",
            "intervention_threshold_usdjpy": 170,
        },
        constraints=[
            Constraint(
                constraint_id="foreign_reserve_floor_800B",
                description="BOJ cannot liquidate below $800B foreign reserve floor",
                affected_actor_ids=["bank_of_japan"],
            ),
            Constraint(
                constraint_id="usjp_alliance_obligations",
                description="US-Japan security alliance constrains aggressive sell-off",
                affected_actor_ids=["bank_of_japan", "us_federal_reserve"],
            ),
        ],
        decision_priorities=[
            "yen_stability",
            "reserve_adequacy",
            "alliance_preservation",
            "inflation_control",
        ],
    )

    nomura_profile = ActorProfile(
        actor_id="nomura_holdings",
        actor_name="Nomura Holdings",
        actor_type=ActorType.financial_institution,
        behavioral_parameters={
            "ust_portfolio_usd": 45e9,
            "var_limit_usd": 2e9,
            "risk_tolerance": 0.55,
            "leverage_ratio": 12.4,
            "hedging_coverage": 0.72,
            "liquidity_buffer_days": 5,
            "mandate": "profit_maximisation_within_risk_limits",
        },
        constraints=[
            Constraint(
                constraint_id="nomura_var_limit",
                description="Daily VaR must not exceed $2B under 99% confidence",
                affected_actor_ids=["nomura_holdings"],
            ),
        ],
        decision_priorities=[
            "portfolio_risk_reduction",
            "client_obligation_fulfillment",
            "regulatory_compliance",
        ],
    )

    sc_quantitative = ScenarioConstraint(
        description="BOJ foreign reserves must remain above $800B floor at all times",
        constraint_type=ConstraintType.quantitative_bound,
        affected_claim_types=["behavioral", "quantitative"],
        affected_actor_ids=["bank_of_japan"],
        parameters={
            "metric": "foreign_reserves_usd",
            "lower_bound": 8e11,
            "unit": "USD",
        },
        severity=Severity.hard,
    )

    sc_behavioral = ScenarioConstraint(
        description="BOJ liquidation rate cannot exceed $200B per month without "
        "triggering coordinated central bank response",
        constraint_type=ConstraintType.behavioral_rule,
        affected_claim_types=["behavioral"],
        affected_actor_ids=["bank_of_japan"],
        parameters={
            "max_monthly_liquidation_usd": 2e11,
            "escalation_trigger": "coordinated_cb_response",
        },
        severity=Severity.soft,
    )

    sc_causal = ScenarioConstraint(
        description="US 10Y yield change must not exceed 300bps in any 72-hour window "
        "absent direct Fed intervention",
        constraint_type=ConstraintType.causal_invariant,
        affected_claim_types=["causal", "quantitative"],
        affected_actor_ids=["bank_of_japan", "us_federal_reserve"],
        parameters={
            "metric": "us_10y_yield_change_bps",
            "upper_bound": 300,
            "window_hours": 72,
            "exception_condition": "fed_emergency_intervention",
        },
        severity=Severity.hard,
    )

    scenario = Scenario(
        title="BOJ Treasury Liquidation Crisis",
        description=(
            "Simulates a rapid Bank of Japan liquidation of US Treasury holdings "
            "triggered by a sustained USD/JPY breach above 170. Models cascading "
            "effects across sovereign debt markets, housing, and banking sectors."
        ),
        seed_document_path="experiments/scenarios/boj_treasury_liquidation.json",
        event_type="sovereign_debt_liquidation",
        constraints=[sc_quantitative, sc_behavioral, sc_causal],
        actor_profiles=[boj_profile, nomura_profile],
        injected_errors=[
            {
                "injection_id": "err_boj_reserve_underflow",
                "target_agent": "agent_2",
                "target_claim_type": "behavioral",
                "injected_value": {
                    "predicted_action": "decrease_reserves",
                    "action_magnitude": 0.95,
                },
                "correct_value": {
                    "predicted_action": "decrease_reserves",
                    "action_magnitude": 0.35,
                },
                "description": (
                    "Inflates BOJ liquidation magnitude to 95% of reserves, "
                    "violating the $800B reserve floor hard constraint. "
                    "Validator should reject this claim."
                ),
            }
        ],
    )

    # ------------------------------------------------------------------
    # 2. Print and verify nested structures
    # ------------------------------------------------------------------

    print(scenario.model_dump_json(indent=2))
    print()

    assert len(scenario.actor_profiles) == 2
    print("assertion passed: len(actor_profiles) == 2")

    assert len(scenario.constraints) == 3
    print("assertion passed: len(constraints) == 3")

    assert len(scenario.injected_errors) == 1
    print("assertion passed: len(injected_errors) == 1")

    # ------------------------------------------------------------------
    # 3. Verify enum rejection
    # ------------------------------------------------------------------

    print()
    try:
        ActorProfile(
            actor_name="Bad Actor",
            actor_type="hedge_fund",  # type: ignore[arg-type]
            behavioral_parameters={},
            constraints=[],
            decision_priorities=[],
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("enum rejection passed: actor_type='hedge_fund' -> ValidationError")

    try:
        ScenarioConstraint(
            description="x",
            constraint_type=ConstraintType.quantitative_bound,
            affected_claim_types=[],
            affected_actor_ids=[],
            parameters={},
            severity="critical",  # type: ignore[arg-type]
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("enum rejection passed: severity='critical' -> ValidationError")

    try:
        ScenarioConstraint(
            description="x",
            constraint_type="unknown",  # type: ignore[arg-type]
            affected_claim_types=[],
            affected_actor_ids=[],
            parameters={},
            severity=Severity.hard,
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        print("enum rejection passed: constraint_type='unknown' -> ValidationError")

    print("\nAll scenario schema checks passed.")
