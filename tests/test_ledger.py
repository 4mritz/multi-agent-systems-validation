import pytest

from mas_validation.ledger import ClaimLedger
from mas_validation.schemas.claims import FactualClaim, BehavioralClaim


def _make_factual(**overrides) -> FactualClaim:
    defaults = dict(
        agent_id="agent_1",
        pipeline_step=1,
        source="seed_document",
        confidence_score=0.90,
        statement="BOJ holds $1.27T in US Treasuries",
        entities=["Bank of Japan"],
        parameters={"holdings_usd": 1.27e12},
    )
    defaults.update(overrides)
    return FactualClaim(**defaults)


def test_add_claim_as_baseclaim_object():
    ledger = ClaimLedger()
    claim = _make_factual()
    result = ledger.add_claim(claim)
    assert result["claim_id"] == claim.claim_id
    assert len(ledger) == 1


def test_add_claim_as_dict():
    ledger = ClaimLedger()
    claim_dict = {
        "claim_type": "factual",
        "agent_id": "agent_1",
        "pipeline_step": 1,
        "source": "seed_document",
        "confidence_score": 0.85,
        "statement": "Japan reserves at $1.15T",
        "entities": ["Japan"],
        "parameters": {"reserves_usd": 1.15e12},
    }
    result = ledger.add_claim(claim_dict)
    assert result["claim_type"] == "factual"
    assert "claim_id" in result


def test_add_invalid_claim_raises_valueerror():
    ledger = ClaimLedger()
    with pytest.raises(ValueError):
        ledger.add_claim({"claim_type": "unknown_type", "agent_id": "x"})


def test_get_by_agent():
    ledger = ClaimLedger()
    ledger.add_claim(_make_factual(agent_id="agent_1"))
    ledger.add_claim(_make_factual(agent_id="agent_2"))
    ledger.add_claim(_make_factual(agent_id="agent_1"))

    agent1_claims = ledger.get_by_agent("agent_1")
    assert len(agent1_claims) == 2
    agent2_claims = ledger.get_by_agent("agent_2")
    assert len(agent2_claims) == 1


def test_get_by_validation_status():
    ledger = ClaimLedger()
    claim = _make_factual()
    ledger.add_claim(claim)
    ledger.update_validation_status(claim.claim_id, "passed")

    passed = ledger.get_by_validation_status("passed")
    assert len(passed) == 1
    assert passed[0]["claim_id"] == claim.claim_id


def test_update_validation_status_invalid_raises():
    ledger = ClaimLedger()
    claim = _make_factual()
    ledger.add_claim(claim)
    with pytest.raises(ValueError):
        ledger.update_validation_status(claim.claim_id, "invalid_status")


def test_update_validation_status_missing_id_raises():
    ledger = ClaimLedger()
    with pytest.raises(ValueError):
        ledger.update_validation_status("nonexistent-id", "passed")


def test_ledger_contradiction():
    ledger = ClaimLedger()
    claim1 = _make_factual(parameters={"holdings_usd": 1.27e12})
    ledger.add_claim(claim1)

    results = ledger.get_contradictable_claims("factual", ["Bank of Japan"])
    assert len(results) == 1
    assert results[0]["parameters"]["holdings_usd"] == 1.27e12


def test_to_dict_returns_copy():
    ledger = ClaimLedger()
    ledger.add_claim(_make_factual())
    snapshot = ledger.to_dict()
    snapshot.clear()
    assert len(ledger) == 1


def test_repr_counts():
    ledger = ClaimLedger()

    # 2 pending
    ledger.add_claim(_make_factual())
    ledger.add_claim(_make_factual())

    # 1 passed
    c_pass = _make_factual()
    ledger.add_claim(c_pass)
    ledger.update_validation_status(c_pass.claim_id, "passed")

    # 1 flagged
    c_flag = _make_factual()
    ledger.add_claim(c_flag)
    ledger.update_validation_status(c_flag.claim_id, "flagged")

    # 1 rejected
    c_rej = _make_factual()
    ledger.add_claim(c_rej)
    ledger.update_validation_status(c_rej.claim_id, "rejected")

    r = repr(ledger)
    assert "n=5" in r
    assert "pending=2" in r
    assert "passed=1" in r
    assert "flagged=1" in r
    assert "rejected=1" in r
