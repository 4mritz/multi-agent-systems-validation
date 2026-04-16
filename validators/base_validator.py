from abc import ABC, abstractmethod

from mas_validation.validators.deterministic import ValidationCheckResult


class BaseValidator(ABC):
    """Abstract base for all validator implementations.

    Exists to document the interface contract and enable future
    validator implementations without coupling to LangGraph node structure.
    """

    @abstractmethod
    def validate(self, claims: list[dict], context: dict) -> list[ValidationCheckResult]:
        """Validate a list of claim dicts against a context.

        Args:
            claims: List of claim dicts from the current agent's output.
            context: Dict containing at minimum: scenario (Scenario), ledger (ClaimLedger).

        Returns:
            List of ValidationCheckResult, one per check performed.
            Checks are ordered by check_number ascending.
        """
        ...

    @staticmethod
    def worst_status(results: list[ValidationCheckResult]) -> str:
        """Return the worst status across all results."""
        if any(r.status == "rejected" for r in results):
            return "rejected"
        if any(r.status == "flagged" for r in results):
            return "flagged"
        return "passed"


class DeterministicValidator(BaseValidator):
    """Thin wrapper around the existing validate_claim function."""

    def validate(self, claims: list[dict], context: dict) -> list[ValidationCheckResult]:
        from mas_validation.validators.deterministic import validate_claim

        scenario = context["scenario"]
        ledger = context["ledger"]

        all_results: list[ValidationCheckResult] = []
        for claim_dict in claims:
            results = validate_claim(claim_dict, scenario, ledger)
            all_results.extend(results)

        return sorted(all_results, key=lambda r: r.check_number)


class LLMValidator(BaseValidator):
    """Thin wrapper around detect_discontinuity + detect_propagation."""

    def validate(self, claims: list[dict], context: dict) -> list[ValidationCheckResult]:
        from mas_validation.validators.llm_validator import (
            detect_discontinuity,
            detect_propagation,
        )
        from mas_validation.schemas.claims import ClaimFactory

        ledger = context["ledger"]
        reasoning = context.get("reasoning", "")

        parsed_claims = []
        for cd in claims:
            try:
                parsed_claims.append(ClaimFactory.from_dict(cd))
            except Exception:
                pass

        results: list[ValidationCheckResult] = []
        if not parsed_claims:
            return results

        disc_result = detect_discontinuity(parsed_claims, reasoning)
        results.append(disc_result)

        prop_result = detect_propagation(parsed_claims, reasoning, ledger)
        results.append(prop_result)

        return sorted(results, key=lambda r: r.check_number)
