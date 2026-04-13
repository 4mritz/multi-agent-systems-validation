class ClaimLedger:
    def __init__(self):
        self._claims = []
        self._next_index = 0

    def add_claim(self, claim: dict) -> dict:
        claim["ledger_index"] = self._next_index
        self._next_index += 1
        self._claims.append(claim)
        return claim

    def get_all(self) -> list:
        return list(self._claims)

    def get_by_agent(self, agent_id: str) -> list:
        return [c for c in self._claims if c.get("agent_id") == agent_id]

    def get_by_step(self, pipeline_step: int) -> list:
        return [c for c in self._claims if c.get("pipeline_step") == pipeline_step]

    def get_by_type(self, claim_type: str) -> list:
        return [c for c in self._claims if c.get("claim_type") == claim_type]

    def get_by_validation_status(self, status: str) -> list:
        return [c for c in self._claims if c.get("validation_status") == status]

    def to_dict(self) -> list:
        return list(self._claims)
