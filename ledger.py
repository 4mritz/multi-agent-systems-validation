from mas_validation.schemas.claims import BaseClaim, ClaimFactory

VALID_STATUSES = {"pending", "passed", "flagged", "rejected"}


class ClaimLedger:
    def __init__(self):
        self._claims: list[dict] = []
        self._next_index: int = 0

    def add_claim(self, claim) -> dict:
        if isinstance(claim, dict):
            try:
                validated = ClaimFactory.from_dict(claim)
            except (ValueError, Exception) as e:
                raise ValueError(f"Invalid claim dict: {e}") from e
            claim_dict = validated.model_dump()
        elif isinstance(claim, BaseClaim):
            claim_dict = claim.model_dump()
        else:
            raise TypeError(f"Expected BaseClaim or dict, got {type(claim).__name__}")

        claim_dict = dict(claim_dict)
        claim_dict["ledger_index"] = self._next_index
        self._next_index += 1
        self._claims.append(claim_dict)
        return claim_dict

    def get_all(self) -> list[dict]:
        return list(self._claims)

    def get_by_agent(self, agent_id: str) -> list[dict]:
        return [c for c in self._claims if c.get("agent_id") == agent_id]

    def get_by_step(self, pipeline_step: int) -> list[dict]:
        return [c for c in self._claims if c.get("pipeline_step") == pipeline_step]

    def get_by_type(self, claim_type: str) -> list[dict]:
        return [c for c in self._claims if c.get("claim_type") == claim_type]

    def get_by_validation_status(self, status: str) -> list[dict]:
        return [c for c in self._claims if c.get("validation_status") == status]

    def get_contradictable_claims(self, claim_type: str, entities: list[str]) -> list[dict]:
        entity_set = set(entities)
        results = []
        for c in self._claims:
            if c.get("claim_type") != claim_type:
                continue
            claim_entities = c.get("entities")
            if not isinstance(claim_entities, list):
                continue
            if entity_set.intersection(claim_entities):
                results.append(c)
        return results

    def update_validation_status(self, claim_id: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {status!r}. Must be one of {VALID_STATUSES}")
        for c in self._claims:
            if c.get("claim_id") == claim_id:
                c["validation_status"] = status
                return
        raise ValueError(f"No claim found with claim_id {claim_id!r}")

    def to_dict(self) -> list[dict]:
        return list(self._claims)

    def __len__(self) -> int:
        return len(self._claims)

    def __repr__(self) -> str:
        counts = {s: 0 for s in ("pending", "passed", "flagged", "rejected")}
        for c in self._claims:
            s = c.get("validation_status", "pending")
            if s in counts:
                counts[s] += 1
        return (
            f"ClaimLedger(n={len(self)}, "
            f"pending={counts['pending']}, "
            f"passed={counts['passed']}, "
            f"flagged={counts['flagged']}, "
            f"rejected={counts['rejected']})"
        )


if __name__ == "__main__":
    from mas_validation.schemas.claims import (
        BehavioralClaim,
        CausalClaim,
        FactualClaim,
        QuantitativeClaim,
    )

    ledger = ClaimLedger()

    # --- Add claims as BaseClaim objects ---
    factual = FactualClaim(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.92,
        statement="The Bank of Japan held $1.27 trillion in US Treasury securities as of Q3 2024",
        entities=["Bank of Japan", "US Treasury", "Japan"],
        parameters={"holdings_usd": 1.27e12, "quarter": "Q3 2024"},
    )
    ledger.add_claim(factual)

    behavioral = BehavioralClaim(
        agent_id="agent_2",
        pipeline_step=2,
        source="actor_profile:boj",
        confidence_score=0.78,
        actor_id="bank_of_japan",
        trigger_condition="USD/JPY breaches 170",
        predicted_action="decrease_reserves",
        action_magnitude=0.35,
        active_constraints=["foreign_reserve_floor_800B"],
    )
    ledger.add_claim(behavioral)

    # --- Add claims as dicts ---
    causal_dict = {
        "claim_type": "causal",
        "agent_id": "agent_3",
        "pipeline_step": 3,
        "source": "impact_model",
        "confidence_score": 0.65,
        "cause": "Rapid BOJ liquidation of US Treasuries",
        "effect": "US 10-year yield spike of 150-250 bps within 72 hours",
        "mechanism_category": "market_reaction",
        "conditions": ["liquidation > $200B in 30 days", "no central bank intervention"],
        "strength": 0.72,
        "supporting_claim_ids": [factual.claim_id, behavioral.claim_id],
    }
    ledger.add_claim(causal_dict)

    quant_dict = {
        "claim_type": "quantitative",
        "agent_id": "agent_3",
        "pipeline_step": 3,
        "source": "impact_model",
        "confidence_score": 0.58,
        "metric": "us_10y_yield_spike",
        "value": 2.0,
        "unit": "percentage_points",
        "source_claim_ids": [],
    }
    ledger.add_claim(quant_dict)

    print(f"Ledger after adding 4 claims: {ledger!r}")
    print(f"Total claims: {len(ledger)}")

    # --- get_contradictable_claims ---
    print("\n=== get_contradictable_claims ===")
    overlapping = ledger.get_contradictable_claims("factual", ["Bank of Japan", "Federal Reserve"])
    print(f"Factual claims mentioning 'Bank of Japan' or 'Federal Reserve': {len(overlapping)}")
    assert len(overlapping) == 1
    assert overlapping[0]["claim_id"] == factual.claim_id
    print(f"  Matched: {overlapping[0]['statement']}")

    no_match = ledger.get_contradictable_claims("factual", ["European Central Bank"])
    assert len(no_match) == 0
    print("  No match for unrelated entities: OK")

    behavioral_match = ledger.get_contradictable_claims("behavioral", ["anything"])
    assert len(behavioral_match) == 0
    print("  Behavioral claims have no entities field, returns empty: OK")

    # --- update_validation_status ---
    print(f"\n=== update_validation_status ===")
    print(f"Before: {ledger!r}")
    ledger.update_validation_status(factual.claim_id, "passed")
    print(f"After marking factual claim as 'passed': {ledger!r}")

    updated = [c for c in ledger.get_all() if c["claim_id"] == factual.claim_id][0]
    assert updated["validation_status"] == "passed"
    print("  Status update verified: OK")

    # --- Verify invalid status raises ValueError ---
    try:
        ledger.update_validation_status(factual.claim_id, "invalid_status")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid status rejected: {e}")

    # --- Verify missing claim_id raises ValueError ---
    try:
        ledger.update_validation_status("nonexistent-id", "passed")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Missing claim_id rejected: {e}")

    # --- Verify invalid dict raises ValueError ---
    print("\n=== Invalid dict handling ===")
    try:
        ledger.add_claim({"claim_type": "bogus", "agent_id": "x"})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid dict rejected: {e}")

    try:
        ledger.add_claim({"claim_type": "factual", "confidence_score": 5.0})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Out-of-range confidence rejected: {e}")

    print(f"\nFinal ledger: {ledger!r}")
    print("All ledger tests passed.")
