from datetime import datetime
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseClaim(BaseModel):
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    claim_type: str
    agent_id: str
    pipeline_step: int
    source: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    validation_status: Literal["pending", "passed", "flagged", "rejected"] = "pending"


class FactualClaim(BaseClaim):
    claim_type: Literal["factual"] = "factual"
    statement: str
    entities: list[str]
    parameters: dict


class BehavioralClaim(BaseClaim):
    claim_type: Literal["behavioral"] = "behavioral"
    actor_id: str
    trigger_condition: str
    predicted_action: Literal[
        "increase_reserves",
        "decrease_reserves",
        "increase_spending",
        "decrease_spending",
        "policy_tighten",
        "policy_loosen",
        "halt_operations",
        "seek_external_support",
        "maintain_status_quo",
    ]
    action_magnitude: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    active_constraints: list[str]


class CausalClaim(BaseClaim):
    claim_type: Literal["causal"] = "causal"
    cause: str
    effect: str
    mechanism_category: Literal[
        "policy_response",
        "market_reaction",
        "resource_depletion",
        "cascade_failure",
        "behavioral_adaptation",
        "systemic_feedback",
    ]
    conditions: list[str]
    strength: float = Field(ge=0.0, le=1.0)
    supporting_claim_ids: list[str]


class QuantitativeClaim(BaseClaim):
    claim_type: Literal["quantitative"] = "quantitative"
    metric: str
    value: float
    unit: str
    source_claim_ids: list[str]


CLAIM_TYPE_MAP = {
    "factual": FactualClaim,
    "behavioral": BehavioralClaim,
    "causal": CausalClaim,
    "quantitative": QuantitativeClaim,
}


class ClaimFactory:
    @staticmethod
    def from_dict(data: dict) -> BaseClaim:
        claim_type = data.get("claim_type")
        cls = CLAIM_TYPE_MAP.get(claim_type)
        if cls is None:
            raise ValueError(f"Unrecognized claim_type: {claim_type!r}")
        return cls(**data)


if __name__ == "__main__":
    import json

    factual = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.92,
        statement="The Bank of Japan held $1.27 trillion in US Treasury securities as of Q3 2024",
        entities=["Bank of Japan", "US Treasury", "Japan"],
        parameters={"holdings_usd": 1.27e12, "quarter": "Q3 2024", "asset_class": "sovereign_debt"},
    )

    behavioral = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:boj",
        confidence_score=0.78,
        actor_id="bank_of_japan",
        trigger_condition="USD/JPY breaches 170 with sustained capital outflows exceeding $50B/month",
        predicted_action="decrease_reserves",
        action_magnitude=0.35,
        active_constraints=["foreign_reserve_floor_800B", "usjp_alliance_obligations"],
    )

    causal = CausalClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="impact_model",
        confidence_score=0.65,
        cause="Rapid BOJ liquidation of US Treasuries",
        effect="US 10-year yield spike of 150-250 basis points within 72 hours",
        mechanism_category="market_reaction",
        conditions=[
            "liquidation volume exceeds $200B in 30 days",
            "no coordinated central bank intervention",
            "US debt-to-GDP above 120%",
        ],
        strength=0.72,
        supporting_claim_ids=[factual.claim_id, behavioral.claim_id],
    )

    quantitative = QuantitativeClaim(
        agent_id="agent_3",
        pipeline_step=3,
        source="impact_model",
        confidence_score=0.58,
        metric="us_10y_yield_spike",
        value=2.0,
        unit="percentage_points",
        source_claim_ids=[causal.claim_id],
    )

    claims = [factual, behavioral, causal, quantitative]

    for claim in claims:
        print(f"--- {claim.claim_type.upper()} CLAIM ---")
        print(json.dumps(claim.model_dump(), indent=2, default=str))
        print()

    # Round-trip verification via ClaimFactory
    print("=== ClaimFactory round-trip verification ===")
    for claim in claims:
        rebuilt = ClaimFactory.from_dict(claim.model_dump())
        assert type(rebuilt) is type(claim), f"Type mismatch: {type(rebuilt)} != {type(claim)}"
        assert rebuilt.model_dump() == claim.model_dump(), f"Data mismatch for {claim.claim_type}"
        print(f"  {claim.claim_type:14s} -> {type(rebuilt).__name__:20s} OK")

    print("\nAll claims validated successfully.")
