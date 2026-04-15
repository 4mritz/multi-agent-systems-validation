from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from mas_validation.schemas.claims import (
    BaseClaim,
    BehavioralClaim,
    CausalClaim,
    ClaimFactory,
    FactualClaim,
    QuantitativeClaim,
)


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


VALID_PREDICTED_ACTIONS = {
    "increase_reserves",
    "decrease_reserves",
    "increase_spending",
    "decrease_spending",
    "policy_tighten",
    "policy_loosen",
    "halt_operations",
    "seek_external_support",
    "maintain_status_quo",
}


class Constraint(BaseModel):
    constraint_id: str
    description: str
    affected_actor_ids: list[str]
    prohibited_actions: list[str] = Field(default_factory=list)

    @field_validator("prohibited_actions")
    @classmethod
    def check_prohibited_actions(cls, v: list[str]) -> list[str]:
        invalid = [a for a in v if a not in VALID_PREDICTED_ACTIONS]
        if invalid:
            raise ValueError(
                f"Invalid prohibited_actions: {invalid}. "
                f"Must be one of {sorted(VALID_PREDICTED_ACTIONS)}"
            )
        return v


class ActorResponse(BaseModel):
    actor_id: str
    response_summary: str
    predicted_actions: list[dict]  # BehavioralClaim.model_dump() dicts
    confidence_score: float = Field(ge=0.0, le=1.0)
    extracted_claims: list[dict]  # BaseClaim.model_dump() dicts


class SystemicEffect(BaseModel):
    effect_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    magnitude: float = Field(ge=0.0, le=1.0)
    affected_sectors: list[str]
    cause_chain: list[str]  # claim IDs referencing ledger claims
    second_order_effects: list[str]


class KeyFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    supporting_claim_ids: list[str]


# ---------------------------------------------------------------------------
# Agent output models
# ---------------------------------------------------------------------------


class Agent1Output(BaseModel):
    event_type: str
    magnitude: float
    affected_sectors: list[str]
    affected_actors: list[str]
    active_constraints: list[Constraint]
    extracted_claims: list[dict]  # claim dicts from BaseClaim.model_dump()


class Agent2Output(BaseModel):
    actor_responses: list[ActorResponse]
    extracted_claims: list[dict]  # top-level claims across all actor responses


class Agent3Output(BaseModel):
    systemic_effects: list[SystemicEffect]
    extracted_claims: list[dict]  # causal/quantitative claims from impact assessment


class Agent4Output(BaseModel):
    executive_summary: str
    key_findings: list[KeyFinding]
    flagged_uncertainties: list[str]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    extracted_claims: list[dict]  # claims supporting final synthesis


if __name__ == "__main__":
    import json

    from pydantic import ValidationError

    # ------------------------------------------------------------------
    # 1. Instantiate all four outputs with BOJ Treasury liquidation data
    # ------------------------------------------------------------------

    # Shared claims used across agents
    factual_1 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.93,
        statement="BOJ holds $1.27T in US Treasuries as of Q3 2024",
        entities=["Bank of Japan", "US Treasury"],
        parameters={"holdings_usd": 1.27e12, "quarter": "Q3 2024"},
    )

    factual_2 = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.88,
        statement="Japan foreign reserves stand at $1.15T with a policy floor of $800B",
        entities=["Japan", "Ministry of Finance"],
        parameters={"reserves_usd": 1.15e12, "floor_usd": 8e11},
    )

    behavioral_boj = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:boj",
        confidence_score=0.76,
        actor_id="bank_of_japan",
        trigger_condition="USD/JPY breaches 170 with sustained capital outflows >$50B/month",
        predicted_action="decrease_reserves",
        action_magnitude=0.35,
        active_constraints=["foreign_reserve_floor_800B", "usjp_alliance_obligations"],
    )

    behavioral_fed = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:fed",
        confidence_score=0.70,
        actor_id="us_federal_reserve",
        trigger_condition="10Y yield spikes >200bps within 48 hours",
        predicted_action="policy_loosen",
        action_magnitude=0.40,
        active_constraints=["inflation_mandate", "financial_stability_mandate"],
    )

    behavioral_pboc = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:pboc",
        confidence_score=0.62,
        actor_id="peoples_bank_of_china",
        trigger_condition="US Treasury sell-off triggers contagion fears in Asian sovereign debt",
        predicted_action="decrease_reserves",
        action_magnitude=0.20,
        active_constraints=["capital_control_framework", "belt_and_road_commitments"],
    )

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
        supporting_claim_ids=[factual_1.claim_id, behavioral_boj.claim_id],
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

    causal_2 = CausalClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="impact_model",
        confidence_score=0.64,
        cause="US yield spike triggers mortgage rate surge",
        effect="US housing market contraction of 8-15% over 6 months",
        mechanism_category="cascade_failure",
        conditions=["mortgage rates exceed 9%", "no fiscal stimulus package"],
        strength=0.60,
        supporting_claim_ids=[causal_1.claim_id, quantitative_1.claim_id],
    )

    factual_synthesis = FactualClaim(
        agent_id="agent_4",
        pipeline_step=4,
        source="synthesis",
        confidence_score=0.82,
        statement="Combined sell-off probability from BOJ and PBOC exceeds 40% under crisis scenario",
        entities=["Bank of Japan", "PBOC", "US Treasury"],
        parameters={"combined_probability": 0.42},
    )

    # --- Agent 1 Output ---
    agent1_out = Agent1Output(
        event_type="sovereign_debt_liquidation",
        magnitude=0.85,
        affected_sectors=["government_bonds", "forex", "banking", "housing"],
        affected_actors=["bank_of_japan", "us_federal_reserve", "peoples_bank_of_china"],
        active_constraints=[
            Constraint(
                constraint_id="foreign_reserve_floor_800B",
                description="BOJ cannot liquidate below $800B foreign reserve floor",
                affected_actor_ids=["bank_of_japan"],
                prohibited_actions=["halt_operations", "increase_spending"],
            ),
            Constraint(
                constraint_id="usjp_alliance_obligations",
                description="US-Japan security alliance constrains aggressive sell-off",
                affected_actor_ids=["bank_of_japan", "us_federal_reserve"],
                prohibited_actions=["decrease_reserves"],
            ),
        ],
        extracted_claims=[factual_1.model_dump(), factual_2.model_dump()],
    )

    # --- Agent 2 Output ---
    boj_actor_claims = [factual_1.model_dump()]
    fed_actor_claims = [
        FactualClaim(
            agent_id="agent_2",
            pipeline_step=2,
            source="actor_profile:fed",
            confidence_score=0.85,
            statement="Fed emergency lending facilities can deploy up to $500B within 24 hours",
            entities=["Federal Reserve", "US Treasury"],
            parameters={"facility_capacity_usd": 5e11},
        ).model_dump()
    ]

    agent2_out = Agent2Output(
        actor_responses=[
            ActorResponse(
                actor_id="bank_of_japan",
                response_summary="BOJ likely to begin measured liquidation of US Treasuries "
                "if USD/JPY breaches 170, constrained by reserve floor and alliance ties.",
                predicted_actions=[behavioral_boj.model_dump()],
                confidence_score=0.76,
                extracted_claims=boj_actor_claims,
            ),
            ActorResponse(
                actor_id="us_federal_reserve",
                response_summary="Fed expected to loosen policy via emergency rate cuts and "
                "expanded repo facilities if 10Y yield spikes >200bps.",
                predicted_actions=[behavioral_fed.model_dump()],
                confidence_score=0.70,
                extracted_claims=fed_actor_claims,
            ),
            ActorResponse(
                actor_id="peoples_bank_of_china",
                response_summary="PBOC may reduce US Treasury holdings defensively if "
                "Asian sovereign debt contagion materialises.",
                predicted_actions=[behavioral_pboc.model_dump()],
                confidence_score=0.62,
                extracted_claims=[],
            ),
        ],
        extracted_claims=[
            behavioral_boj.model_dump(),
            behavioral_fed.model_dump(),
            behavioral_pboc.model_dump(),
        ],
    )

    # --- Agent 3 Output ---
    agent3_out = Agent3Output(
        systemic_effects=[
            SystemicEffect(
                description="US Treasury market dislocation with 150-250bps yield spike",
                magnitude=0.85,
                affected_sectors=["government_bonds", "forex", "money_markets"],
                cause_chain=[factual_1.claim_id, behavioral_boj.claim_id, causal_1.claim_id],
                second_order_effects=[
                    "Global risk-off sentiment drives flight from EM debt",
                    "USD volatility triggers FX hedging cost surge for Asian corporates",
                ],
            ),
            SystemicEffect(
                description="US housing market contraction of 8-15% over 6 months",
                magnitude=0.65,
                affected_sectors=["housing", "banking", "consumer_credit"],
                cause_chain=[causal_1.claim_id, quantitative_1.claim_id, causal_2.claim_id],
                second_order_effects=[
                    "Regional bank failures accelerate due to MBS exposure",
                ],
            ),
        ],
        extracted_claims=[
            causal_1.model_dump(),
            quantitative_1.model_dump(),
            causal_2.model_dump(),
        ],
    )

    # --- Agent 4 Output ---
    agent4_out = Agent4Output(
        executive_summary=(
            "A rapid BOJ liquidation of US Treasury holdings would trigger a severe "
            "multi-sector crisis. The most probable scenario involves a 150-250bps "
            "spike in US 10Y yields within 72 hours, cascading into housing market "
            "contraction, regional banking stress, and potential contagion across "
            "Asian sovereign debt markets. Fed emergency facilities and the US-Japan "
            "alliance provide partial buffers, but the system is fragile if PBOC "
            "joins the sell-off."
        ),
        key_findings=[
            KeyFinding(
                description="BOJ liquidation above $200B/month triggers 150-250bps yield spike "
                "with 74% mechanism strength",
                confidence_score=0.71,
                supporting_claim_ids=[
                    factual_1.claim_id,
                    behavioral_boj.claim_id,
                    causal_1.claim_id,
                ],
            ),
            KeyFinding(
                description="Housing market faces 8-15% contraction as mortgage rates exceed 9%",
                confidence_score=0.64,
                supporting_claim_ids=[causal_1.claim_id, causal_2.claim_id],
            ),
            KeyFinding(
                description="PBOC joining sell-off raises combined liquidation probability above 40%",
                confidence_score=0.62,
                supporting_claim_ids=[
                    behavioral_pboc.claim_id,
                    factual_synthesis.claim_id,
                ],
            ),
        ],
        flagged_uncertainties=[
            "PBOC response is highly contingent on US-China diplomatic state",
            "Fed emergency facility deployment speed is untested at this scale",
            "BOJ reserve floor may be revised downward under extreme yen pressure",
        ],
        overall_confidence=0.68,
        extracted_claims=[factual_synthesis.model_dump()],
    )

    # Print all outputs
    outputs = [
        ("Agent1Output", agent1_out),
        ("Agent2Output", agent2_out),
        ("Agent3Output", agent3_out),
        ("Agent4Output", agent4_out),
    ]
    for name, model in outputs:
        print(f"=== {name} ===")
        print(json.dumps(model.model_dump(), indent=2, default=str))
        print()

    # ------------------------------------------------------------------
    # 2. Round-trip verification of extracted_claims via ClaimFactory
    # ------------------------------------------------------------------
    print("=== ClaimFactory round-trip verification ===")
    all_claim_dicts: list[tuple[str, dict]] = []
    for name, model in outputs:
        for cd in model.extracted_claims:
            all_claim_dicts.append((name, cd))
        # Also check nested ActorResponse claims for Agent2
        if isinstance(model, Agent2Output):
            for ar in model.actor_responses:
                for cd in ar.extracted_claims:
                    all_claim_dicts.append((f"{name}::{ar.actor_id}", cd))
                for cd in ar.predicted_actions:
                    all_claim_dicts.append((f"{name}::{ar.actor_id}::predicted", cd))

    expected_types = {
        "factual": FactualClaim,
        "behavioral": BehavioralClaim,
        "causal": CausalClaim,
        "quantitative": QuantitativeClaim,
    }

    for source_label, cd in all_claim_dicts:
        rebuilt = ClaimFactory.from_dict(cd)
        expected_cls = expected_types[cd["claim_type"]]
        assert isinstance(rebuilt, expected_cls), (
            f"Expected {expected_cls.__name__}, got {type(rebuilt).__name__} "
            f"from {source_label}"
        )
        print(f"  {source_label:45s} -> {type(rebuilt).__name__:20s} OK")

    print(f"\n  All {len(all_claim_dicts)} claim dicts round-tripped successfully.\n")

    # ------------------------------------------------------------------
    # 3. Validation error checks
    # ------------------------------------------------------------------
    print("=== Validation error checks ===")
    error_count = 0

    # ActorResponse: confidence_score > 1.0
    try:
        ActorResponse(
            actor_id="x",
            response_summary="x",
            predicted_actions=[],
            confidence_score=1.5,
            extracted_claims=[],
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  ActorResponse  confidence_score=1.5   -> ValidationError OK")

    # ActorResponse: missing required field
    try:
        ActorResponse(actor_id="x", predicted_actions=[], confidence_score=0.5)  # type: ignore[call-arg]
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  ActorResponse  missing response_summary -> ValidationError OK")

    # SystemicEffect: magnitude < 0.0
    try:
        SystemicEffect(
            description="x",
            magnitude=-0.1,
            affected_sectors=[],
            cause_chain=[],
            second_order_effects=[],
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  SystemicEffect magnitude=-0.1          -> ValidationError OK")

    # SystemicEffect: missing required field
    try:
        SystemicEffect(magnitude=0.5, affected_sectors=[])  # type: ignore[call-arg]
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  SystemicEffect missing description      -> ValidationError OK")

    # KeyFinding: confidence_score > 1.0
    try:
        KeyFinding(
            description="x",
            confidence_score=2.0,
            supporting_claim_ids=[],
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  KeyFinding     confidence_score=2.0     -> ValidationError OK")

    # KeyFinding: missing required field
    try:
        KeyFinding(confidence_score=0.5)  # type: ignore[call-arg]
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  KeyFinding     missing description      -> ValidationError OK")

    # Constraint: missing required field
    try:
        Constraint(constraint_id="x")  # type: ignore[call-arg]
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  Constraint     missing description      -> ValidationError OK")

    # Constraint: invalid prohibited_actions value
    try:
        Constraint(
            constraint_id="x",
            description="x",
            affected_actor_ids=[],
            prohibited_actions=["decrease_reserves", "nuke_economy"],
        )
        assert False, "Should have raised ValidationError"
    except ValidationError:
        error_count += 1
        print("  Constraint     prohibited_actions='nuke_economy' -> ValidationError OK")

    print(f"\n  All {error_count} validation error checks passed.")
