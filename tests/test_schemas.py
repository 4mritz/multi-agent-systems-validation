import uuid

import pytest
from pydantic import ValidationError

from mas_validation.schemas.claims import (
    BaseClaim,
    BehavioralClaim,
    CausalClaim,
    ClaimFactory,
    FactualClaim,
    QuantitativeClaim,
)
from mas_validation.schemas.agent_outputs import Agent1Output, Constraint
from mas_validation.schemas.scenario import ActorProfile, ActorType


def test_factual_claim_default_fields():
    claim = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.9,
        statement="test",
        entities=["A"],
        parameters={},
    )
    # claim_id should be a valid UUID
    uuid.UUID(claim.claim_id)
    # timestamp should be an ISO format string
    assert "T" in claim.timestamp
    assert claim.validation_status == "pending"


def test_factual_claim_confidence_out_of_range():
    with pytest.raises(ValidationError):
        FactualClaim(
            agent_id="agent_1",
            pipeline_step=1,
            source="seed_document",
            confidence_score=1.5,
            statement="test",
            entities=["A"],
            parameters={},
        )


def test_behavioral_claim_invalid_predicted_action():
    with pytest.raises(ValidationError):
        BehavioralClaim(
            agent_id="agent_2",
            pipeline_step=2,
            source="profile",
            confidence_score=0.7,
            actor_id="boj",
            trigger_condition="trigger",
            predicted_action="fire_missiles",
            active_constraints=[],
        )


def test_causal_claim_invalid_mechanism():
    with pytest.raises(ValidationError):
        CausalClaim(
            agent_id="agent_3",
            pipeline_step=3,
            source="model",
            confidence_score=0.6,
            cause="A",
            effect="B",
            mechanism_category="magic",
            conditions=[],
            strength=0.5,
            supporting_claim_ids=[],
        )


def test_claim_factory_dispatches_all_types():
    types_and_classes = [
        (
            {
                "claim_type": "factual",
                "agent_id": "a1",
                "pipeline_step": 1,
                "source": "s",
                "confidence_score": 0.5,
                "statement": "x",
                "entities": [],
                "parameters": {},
            },
            FactualClaim,
        ),
        (
            {
                "claim_type": "behavioral",
                "agent_id": "a2",
                "pipeline_step": 2,
                "source": "s",
                "confidence_score": 0.5,
                "actor_id": "boj",
                "trigger_condition": "t",
                "predicted_action": "maintain_status_quo",
                "active_constraints": [],
            },
            BehavioralClaim,
        ),
        (
            {
                "claim_type": "causal",
                "agent_id": "a3",
                "pipeline_step": 3,
                "source": "s",
                "confidence_score": 0.5,
                "cause": "A",
                "effect": "B",
                "mechanism_category": "market_reaction",
                "conditions": [],
                "strength": 0.5,
                "supporting_claim_ids": [],
            },
            CausalClaim,
        ),
        (
            {
                "claim_type": "quantitative",
                "agent_id": "a3",
                "pipeline_step": 3,
                "source": "s",
                "confidence_score": 0.5,
                "metric": "m",
                "value": 1.0,
                "unit": "u",
                "source_claim_ids": [],
            },
            QuantitativeClaim,
        ),
    ]
    for d, expected_cls in types_and_classes:
        result = ClaimFactory.from_dict(d)
        assert isinstance(result, expected_cls)


def test_claim_factory_unknown_type_raises():
    with pytest.raises(ValueError):
        ClaimFactory.from_dict({"claim_type": "unknown", "agent_id": "x"})


def test_round_trip_all_types():
    claims = [
        FactualClaim(
            agent_id="a1", pipeline_step=1, source="s", confidence_score=0.5,
            statement="x", entities=["e"], parameters={"k": 1},
        ),
        BehavioralClaim(
            agent_id="a2", pipeline_step=2, source="s", confidence_score=0.5,
            actor_id="boj", trigger_condition="t",
            predicted_action="maintain_status_quo", active_constraints=[],
        ),
        CausalClaim(
            agent_id="a3", pipeline_step=3, source="s", confidence_score=0.5,
            cause="A", effect="B", mechanism_category="market_reaction",
            conditions=[], strength=0.5, supporting_claim_ids=[],
        ),
        QuantitativeClaim(
            agent_id="a3", pipeline_step=3, source="s", confidence_score=0.5,
            metric="m", value=1.0, unit="u", source_claim_ids=[],
        ),
    ]
    for claim in claims:
        dumped = claim.model_dump()
        rebuilt = ClaimFactory.from_dict(dumped)
        assert type(rebuilt) is type(claim)
        assert rebuilt.model_dump() == dumped


def test_agent1_output_validates():
    data = {
        "event_type": "crisis",
        "magnitude": 0.8,
        "affected_sectors": ["bonds"],
        "affected_actors": ["boj"],
        "active_constraints": [],
        "extracted_claims": [],
    }
    out = Agent1Output.model_validate(data)
    assert out.event_type == "crisis"


def test_scenario_enum_rejection():
    with pytest.raises(ValidationError):
        ActorProfile(
            actor_name="Bad Actor",
            actor_type="hedge_fund",
            behavioral_parameters={},
            constraints=[],
            decision_priorities=[],
        )


def test_constraint_prohibited_actions_validated():
    with pytest.raises(ValidationError):
        Constraint(
            constraint_id="x",
            description="x",
            affected_actor_ids=[],
            prohibited_actions=["nuke_economy"],
        )
