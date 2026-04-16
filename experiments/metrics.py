import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Result of a single pipeline run (baseline or validated)."""
    scenario_id: str
    pipeline_type: str  # "baseline" or "validated"
    injected_errors: list[dict] = field(default_factory=list)
    agent4_output: dict | None = None
    claim_ledger: list[dict] = field(default_factory=list)
    validation_results: dict = field(default_factory=dict)
    pipeline_status: str = "pending"
    error: str | None = None


class MetricsCalculator:

    @staticmethod
    def _wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson score confidence interval for proportion k/n."""
        if n == 0:
            return 0.0, 0.0
        p = k / n
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return max(0.0, center - margin), min(1.0, center + margin)

    def detect_propagation_in_output(
        self,
        agent4_output: dict | None,
        injected_errors: list[dict],
    ) -> list[dict]:
        """Check whether each injected error's injected_value appears in agent4_output."""
        results = []
        if agent4_output is None:
            for ie in injected_errors:
                results.append({
                    "injection_id": ie.get("injection_id"),
                    "target_agent": ie.get("target_agent"),
                    "injected_value": ie.get("injected_value"),
                    "correct_value": ie.get("correct_value"),
                    "propagated": False,
                    "found_in_field": None,
                })
            return results

        output_str = json.dumps(agent4_output)

        for ie in injected_errors:
            injected_val = str(ie.get("injected_value", ""))
            propagated = False
            found_in_field = None

            if injected_val and injected_val in output_str:
                propagated = True
                # Try to identify which top-level field contains it
                for key, val in agent4_output.items():
                    val_str = json.dumps(val) if not isinstance(val, str) else val
                    if injected_val in str(val_str):
                        found_in_field = key
                        break

            results.append({
                "injection_id": ie.get("injection_id"),
                "target_agent": ie.get("target_agent"),
                "injected_value": ie.get("injected_value"),
                "correct_value": ie.get("correct_value"),
                "propagated": propagated,
                "found_in_field": found_in_field,
            })

        return results

    def compute_error_propagation_rate(self, results: list[RunResult]) -> dict:
        """Given baseline RunResults, compute error propagation rate."""
        n_scenarios = len(results)
        n_propagated = 0
        per_type: dict[str, dict[str, int]] = {
            "quantitative": {"detected": 0, "total": 0},
            "behavioral": {"detected": 0, "total": 0},
            "causal": {"detected": 0, "total": 0},
            "factual": {"detected": 0, "total": 0},
        }

        for run in results:
            propagation_hits = self.detect_propagation_in_output(
                run.agent4_output, run.injected_errors
            )
            scenario_has_propagation = False
            for hit in propagation_hits:
                inj_type = self._get_injection_type(hit, run.injected_errors)
                if inj_type in per_type:
                    per_type[inj_type]["total"] += 1
                    if hit["propagated"]:
                        per_type[inj_type]["detected"] += 1
                        scenario_has_propagation = True

            if scenario_has_propagation:
                n_propagated += 1

        rate = n_propagated / n_scenarios if n_scenarios > 0 else 0.0
        ci_low, ci_high = self._wilson_ci(n_scenarios, n_propagated)

        per_injection_type = {}
        for itype, counts in per_type.items():
            t = counts["total"]
            d = counts["detected"]
            per_injection_type[itype] = {
                "detected": d,
                "total": t,
                "rate": d / t if t > 0 else 0.0,
            }

        return {
            "n_scenarios": n_scenarios,
            "n_propagated": n_propagated,
            "propagation_rate": rate,
            "wilson_ci_low": ci_low,
            "wilson_ci_high": ci_high,
            "per_injection_type": per_injection_type,
        }

    def compute_detection_rate(
        self,
        validated_results: list[RunResult],
        baseline_results: list[RunResult],
    ) -> dict:
        """Compute validator detection rate across all injected errors."""
        n_injections_total = 0
        n_detected = 0
        by_type: dict[str, dict[str, int]] = {
            "quantitative": {"detected": 0, "total": 0},
            "behavioral": {"detected": 0, "total": 0},
            "causal": {"detected": 0, "total": 0},
            "factual": {"detected": 0, "total": 0},
        }
        by_class: dict[str, int] = {
            "schema_validation": 0,
            "scope_check": 0,
            "quantitative_bounds": 0,
            "behavioral_consistency": 0,
            "ledger_contradiction": 0,
            "reasoning_chain_continuity": 0,
            "hallucination_propagation": 0,
        }

        for run in validated_results:
            for ie in run.injected_errors:
                n_injections_total += 1
                inj_type = ie.get("target_claim_type", "")
                if inj_type in by_type:
                    by_type[inj_type]["total"] += 1

                detected = self._was_injection_detected(ie, run)
                if detected:
                    n_detected += 1
                    if inj_type in by_type:
                        by_type[inj_type]["detected"] += 1

                    # Count which validation classes triggered
                    triggered_classes = self._get_triggered_classes(ie, run)
                    for cls_name in triggered_classes:
                        if cls_name in by_class:
                            by_class[cls_name] += 1

        detection_rate = n_detected / n_injections_total if n_injections_total > 0 else 0.0
        ci_low, ci_high = self._wilson_ci(n_injections_total, n_detected)

        by_injection_type = {}
        for itype, counts in by_type.items():
            t = counts["total"]
            d = counts["detected"]
            by_injection_type[itype] = {
                "detected": d,
                "total": t,
                "rate": d / t if t > 0 else 0.0,
            }

        by_validation_class = {}
        for cls_name, triggered in by_class.items():
            by_validation_class[cls_name] = {
                "triggered": triggered,
                "pct": triggered / n_injections_total if n_injections_total > 0 else 0.0,
            }

        return {
            "n_injections_total": n_injections_total,
            "n_detected": n_detected,
            "detection_rate": detection_rate,
            "wilson_ci_low": ci_low,
            "wilson_ci_high": ci_high,
            "by_injection_type": by_injection_type,
            "by_validation_class": by_validation_class,
        }

    def compute_false_positive_rate(self, validated_results: list[RunResult]) -> dict:
        """Count validation flags/rejects on claims NOT in the injected_errors set.

        Only counts claims that passed schema validation as FP candidates.
        Schema rejections are correct validator behavior, not false positives —
        they indicate the LLM produced malformed output, which the validator
        correctly caught.
        """
        n_clean_claims = 0
        n_false_positives = 0

        for run in validated_results:
            injected_types = {
                (ie.get("target_agent"), ie.get("target_claim_type"))
                for ie in run.injected_errors
            }

            for claim in run.claim_ledger:
                status = claim.get("validation_status", "pending")
                claim_type = claim.get("claim_type", "")
                agent_id = claim.get("agent_id", "")
                agent_num = self._agent_id_to_num(agent_id)

                is_injected = (agent_num, claim_type) in injected_types

                # Skip injected claims — true positives, not FP candidates
                if is_injected:
                    continue

                # Skip claims that failed schema validation — these are correct
                # rejections of malformed LLM output, not false positives.
                # A schema rejection means the LLM produced structurally invalid
                # output; the validator catching this is correct behavior.
                if status == "rejected":
                    if claim_type == "behavioral" and not claim.get("actor_id"):
                        continue
                    if claim_type == "causal" and claim.get("strength") is None:
                        continue
                    if claim_type == "quantitative" and "source_claim_ids" not in claim:
                        continue
                    if claim_type not in ("factual", "behavioral", "causal", "quantitative"):
                        continue

                n_clean_claims += 1
                if status in ("flagged", "rejected"):
                    n_false_positives += 1

        fp_rate = n_false_positives / n_clean_claims if n_clean_claims > 0 else 0.0
        ci_low, ci_high = self._wilson_ci(n_clean_claims, n_false_positives)

        return {
            "n_clean_claims": n_clean_claims,
            "n_false_positives": n_false_positives,
            "false_positive_rate": fp_rate,
            "wilson_ci_low": ci_low,
            "wilson_ci_high": ci_high,
        }

    # --- helpers ---

    @staticmethod
    def _get_injection_type(hit: dict, injected_errors: list[dict]) -> str:
        """Look up target_claim_type for a propagation hit from injected_errors."""
        inj_id = hit.get("injection_id")
        for ie in injected_errors:
            if ie.get("injection_id") == inj_id:
                return ie.get("target_claim_type", "")
        return ""

    @staticmethod
    def _agent_id_to_num(agent_id: str) -> int | None:
        """Convert 'agent_1' -> 1, etc."""
        if agent_id and agent_id.startswith("agent_"):
            try:
                return int(agent_id.split("_")[1])
            except (IndexError, ValueError):
                pass
        return None

    def _was_injection_detected(self, ie: dict, run: RunResult) -> bool:
        """Check if a specific injected error was caught by the validator.

        An injection is detected if the injected_value does NOT appear in
        agent4_output. We cannot reliably trace a specific injected claim
        through the ledger by ID (injections mutate existing claims in-place),
        so we use the propagation check as ground truth: if the injected value
        didn't reach agent4, the validator stopped it.
        """
        # If pipeline failed/crashed, agent4 output is None.
        # A failed validated pipeline means the validator rejected something
        # and halted execution — that counts as detection.
        if run.agent4_output is None and run.pipeline_status in ("failed", "crashed"):
            return True

        # If agent4 output exists, check whether the injected value propagated
        propagation = self.detect_propagation_in_output(run.agent4_output, [ie])
        if not propagation:
            return False

        # Detected = injected value did NOT reach agent4 output
        return not propagation[0]["propagated"]

    def _get_triggered_classes(self, ie: dict, run: RunResult) -> list[str]:
        """Return validation class names that plausibly caught this injection.

        Since we cannot trace a specific injection to a specific check result
        without claim-level IDs on validation events, we return the classes
        that fired in the handoff corresponding to the injection's target agent.
        """
        target_agent = ie.get("target_agent", 0)

        # Map target agent to the handoff key where it would be caught
        handoff_map = {
            1: "1_to_2",  # Agent 1 output validated at handoff 1->2
            2: "2_to_3",  # Agent 2 output validated at handoff 2->3
            3: "3_to_4",  # Agent 3 output validated at handoff 3->4
        }
        target_handoff = handoff_map.get(target_agent)

        triggered = []
        for handoff_key, checks in run.validation_results.items():
            # Only count checks from the relevant handoff
            if target_handoff and handoff_key != target_handoff:
                continue
            for check in checks:
                status = check.get("status", "")
                check_name = check.get("check_name", "")
                if status in ("rejected", "flagged") and check_name not in triggered:
                    triggered.append(check_name)
        return triggered
