import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mas_validation.state import PipelineState
from mas_validation.experiments.metrics import MetricsCalculator, RunResult
from mas_validation.agents.scenario_analysis import run_scenario_analysis
from mas_validation.agents.actor_modeling import run_actor_modeling
from mas_validation.agents.impact_assessment import run_impact_assessment
from mas_validation.agents.decision_synthesis import run_decision_synthesis
from mas_validation.validators.validator_nodes import (
    validate_1_to_2,
    validate_2_to_3,
    validate_3_to_4,
)

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent


class ExperimentRunner:

    def __init__(self):
        self.metrics = MetricsCalculator()

    @staticmethod
    def _load_scenario(scenario_path: str) -> dict:
        """Load a scenario JSON file."""
        with open(scenario_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_seed_document(seed_path: str) -> str:
        """Load the seed document text file, resolved relative to experiments/."""
        resolved = EXPERIMENTS_DIR / seed_path.replace("experiments/", "", 1)
        if not resolved.exists():
            # Try as-is
            resolved = Path(seed_path)
        if not resolved.exists():
            # Try relative to experiments dir
            resolved = EXPERIMENTS_DIR / seed_path
        with open(resolved, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _build_initial_state(scenario: dict, seed_document: str) -> PipelineState:
        """Build the initial PipelineState from a scenario dict and seed document."""
        return PipelineState(
            seed_document=seed_document,
            scenario_constraints={"scenario": scenario},
            agent1_output=None,
            agent2_output=None,
            agent3_output=None,
            agent4_output=None,
            claim_ledger=[],
            validation_results={"1_to_2": [], "2_to_3": [], "3_to_4": []},
            fallback_flags={"1_to_2": False, "2_to_3": False, "3_to_4": False},
            fallback_reasons={"1_to_2": None, "2_to_3": None, "3_to_4": None},
            current_step=None,
            pipeline_status="pending",
        )

    @staticmethod
    def _apply_injected_errors(
        state: PipelineState, injected_errors: list[dict]
    ) -> PipelineState:
        """Inject errors into agent1_output claims after Agent 1 has run."""
        agent1_output = state.get("agent1_output")
        if agent1_output is None:
            return state

        claims = agent1_output.get("extracted_claims", [])

        for ie in injected_errors:
            target_agent = ie.get("target_agent")
            if target_agent != 1:
                continue

            target_type = ie.get("target_claim_type", "")
            injected_value = ie.get("injected_value", "")

            for claim in claims:
                if claim.get("claim_type") != target_type:
                    continue

                # For quantitative claims: replace numeric value
                if target_type == "quantitative":
                    try:
                        claim["value"] = float(injected_value)
                    except (ValueError, TypeError):
                        pass
                    break

                # For factual claims: replace first numeric parameter
                elif target_type == "factual":
                    params = claim.get("parameters", {})
                    for key in list(params.keys()):
                        if isinstance(params[key], (int, float)):
                            try:
                                params[key] = float(injected_value)
                            except (ValueError, TypeError):
                                pass
                            break
                    break

                # For causal claims: inject into effect field
                elif target_type == "causal":
                    claim["effect"] = str(injected_value)
                    break

                # For behavioral claims: inject into predicted_action
                elif target_type == "behavioral":
                    claim["predicted_action"] = str(injected_value)
                    break

        # Also update the claim_ledger to reflect injected changes
        ledger = state.get("claim_ledger", [])
        for ie in injected_errors:
            if ie.get("target_agent") != 1:
                continue
            target_type = ie.get("target_claim_type", "")
            injected_value = ie.get("injected_value", "")
            for lc in ledger:
                if lc.get("claim_type") != target_type:
                    continue
                if target_type == "quantitative":
                    try:
                        lc["value"] = float(injected_value)
                    except (ValueError, TypeError):
                        pass
                    break
                elif target_type == "factual":
                    params = lc.get("parameters", {})
                    for key in list(params.keys()):
                        if isinstance(params[key], (int, float)):
                            try:
                                params[key] = float(injected_value)
                            except (ValueError, TypeError):
                                pass
                            break
                    break
                elif target_type == "causal":
                    lc["effect"] = str(injected_value)
                    break
                elif target_type == "behavioral":
                    lc["predicted_action"] = str(injected_value)
                    break

        # Store injection metadata for metrics
        state["scenario_constraints"]["injected_errors"] = injected_errors

        return state

    def run_single(
        self,
        scenario_path: str,
        pipeline_type: str = "validated",
        apply_injections: bool = True,
    ) -> RunResult:
        """Run a single scenario through the specified pipeline."""
        try:
            scenario = self._load_scenario(scenario_path)
            seed_path = scenario.get("seed_document_path", "")
            seed_document = self._load_seed_document(seed_path)
            injected_errors = scenario.get("injected_errors", [])

            # Step 1: Run Agent 1
            state = self._build_initial_state(scenario, seed_document)
            state = run_scenario_analysis(state)

            if state.get("pipeline_status") == "failed":
                return RunResult(
                    scenario_id=scenario.get("scenario_id", "unknown"),
                    pipeline_type=pipeline_type,
                    injected_errors=injected_errors if apply_injections else [],
                    agent4_output=None,
                    claim_ledger=state.get("claim_ledger", []),
                    validation_results=state.get("validation_results", {}),
                    pipeline_status="failed",
                    error="Agent 1 failed",
                )

            # Step 2: Inject errors after Agent 1
            if apply_injections and injected_errors:
                state = self._apply_injected_errors(state, injected_errors)

            # Step 3: Run the rest of the pipeline
            if pipeline_type == "baseline":
                state = self._run_baseline_remaining(state)
            elif pipeline_type == "validated":
                state = self._run_validated_remaining(state, injected_errors, apply_injections)
            else:
                raise ValueError(f"Unknown pipeline_type: {pipeline_type}")

            return RunResult(
                scenario_id=scenario.get("scenario_id", "unknown"),
                pipeline_type=pipeline_type,
                injected_errors=injected_errors if apply_injections else [],
                agent4_output=state.get("agent4_output"),
                claim_ledger=state.get("claim_ledger", []),
                validation_results=state.get("validation_results", {}),
                pipeline_status=state.get("pipeline_status", "unknown"),
                error=None,
            )

        except Exception as e:
            logger.exception("Pipeline crashed for %s", scenario_path)
            return RunResult(
                scenario_id=scenario.get("scenario_id", "unknown") if "scenario" in dir() else "unknown",
                pipeline_type=pipeline_type,
                injected_errors=[],
                agent4_output=None,
                claim_ledger=[],
                validation_results={},
                pipeline_status="crashed",
                error=str(e),
            )

    def _run_baseline_remaining(self, state: PipelineState) -> PipelineState:
        """Run agents 2, 3, 4 directly (no validators)."""
        try:
            state = run_actor_modeling(state)
        except Exception as e:
            logger.error("Agent 2 crashed: %s", e)
            state["pipeline_status"] = "failed"
            return state

        try:
            state = run_impact_assessment(state)
        except Exception as e:
            logger.error("Agent 3 crashed: %s", e)
            state["pipeline_status"] = "failed"
            return state

        try:
            state = run_decision_synthesis(state)
        except Exception as e:
            logger.error("Agent 4 crashed: %s", e)
            state["pipeline_status"] = "failed"
            return state

        return state

    def _run_validated_remaining(
        self,
        state: PipelineState,
        injected_errors: list[dict],
        apply_injections: bool,
    ) -> PipelineState:
        """Run validator_1_to_2 -> agent2 -> validator_2_to_3 -> agent3 -> validator_3_to_4 -> agent4.

        Halts pipeline if any validator rejects. This is the critical difference
        between baseline and validated pipelines — rejections stop propagation.
        """
        # Validator 1->2
        try:
            state = validate_1_to_2(state)
        except Exception as e:
            logger.error("Validator 1->2 crashed: %s", e)
            state["fallback_flags"]["1_to_2"] = True
            state["fallback_reasons"]["1_to_2"] = str(e)

        # HALT if validator rejected
        if state.get("pipeline_status") == "failed":
            logger.info("Pipeline halted after validator 1->2 rejection")
            return state

        # Agent 2
        try:
            state = run_actor_modeling(state)
        except Exception as e:
            logger.error("Agent 2 crashed: %s", e)
            state["pipeline_status"] = "failed"
            return state

        if state.get("pipeline_status") == "failed":
            return state

        # Apply agent-2-targeted injections after agent 2 runs
        if apply_injections and injected_errors:
            state = self._apply_agent2_injections(state, injected_errors)

        # Validator 2->3
        try:
            state = validate_2_to_3(state)
        except Exception as e:
            logger.error("Validator 2->3 crashed: %s", e)
            state["fallback_flags"]["2_to_3"] = True
            state["fallback_reasons"]["2_to_3"] = str(e)

        # HALT if validator rejected
        if state.get("pipeline_status") == "failed":
            logger.info("Pipeline halted after validator 2->3 rejection")
            return state

        # Agent 3
        try:
            state = run_impact_assessment(state)
        except Exception as e:
            logger.error("Agent 3 crashed: %s", e)
            state["pipeline_status"] = "failed"
            return state

        if state.get("pipeline_status") == "failed":
            return state

        # Apply agent-3-targeted injections after agent 3 runs
        if apply_injections and injected_errors:
            state = self._apply_agent3_injections(state, injected_errors)

        # Validator 3->4
        try:
            state = validate_3_to_4(state)
        except Exception as e:
            logger.error("Validator 3->4 crashed: %s", e)
            state["fallback_flags"]["3_to_4"] = True
            state["fallback_reasons"]["3_to_4"] = str(e)

        # HALT if validator rejected
        if state.get("pipeline_status") == "failed":
            logger.info("Pipeline halted after validator 3->4 rejection")
            return state

        # Agent 4
        try:
            state = run_decision_synthesis(state)
        except Exception as e:
            logger.error("Agent 4 crashed: %s", e)
            state["pipeline_status"] = "failed"
            return state

        return state

    @staticmethod
    def _apply_agent2_injections(
        state: PipelineState, injected_errors: list[dict]
    ) -> PipelineState:
        """Inject errors targeted at agent 2 into agent2_output."""
        agent2_output = state.get("agent2_output")
        if agent2_output is None:
            return state

        claims = agent2_output.get("extracted_claims", [])

        for ie in injected_errors:
            if ie.get("target_agent") != 2:
                continue
            target_type = ie.get("target_claim_type", "")
            injected_value = ie.get("injected_value", "")

            for claim in claims:
                if claim.get("claim_type") != target_type:
                    continue
                if target_type == "behavioral":
                    claim["predicted_action"] = str(injected_value)
                    break

        return state

    @staticmethod
    def _apply_agent3_injections(
        state: PipelineState, injected_errors: list[dict]
    ) -> PipelineState:
        """Inject errors targeted at agent 3 into agent3_output."""
        agent3_output = state.get("agent3_output")
        if agent3_output is None:
            return state

        claims = agent3_output.get("extracted_claims", [])

        for ie in injected_errors:
            if ie.get("target_agent") != 3:
                continue
            target_type = ie.get("target_claim_type", "")
            injected_value = ie.get("injected_value", "")

            for claim in claims:
                if claim.get("claim_type") != target_type:
                    continue
                if target_type == "causal":
                    claim["effect"] = str(injected_value)
                    break

        return state

    def run_experiment(
        self,
        scenario_dir: str = "experiments/scenarios",
        n_repeats: int = 1,
    ) -> dict:
        """Run the full experiment across all scenarios."""
        scenario_path = EXPERIMENTS_DIR / "scenarios"
        if not scenario_path.exists():
            scenario_path = Path(scenario_dir)

        scenario_files = sorted(scenario_path.glob("*_001.json"))
        logger.info("Found %d scenario files", len(scenario_files))

        baseline_results: list[RunResult] = []
        validated_results: list[RunResult] = []

        for repeat in range(n_repeats):
            for sf in scenario_files:
                sf_str = str(sf)
                logger.info(
                    "Running scenario %s (repeat %d/%d)",
                    sf.stem, repeat + 1, n_repeats,
                )

                # Baseline run
                br = self.run_single(sf_str, pipeline_type="baseline", apply_injections=True)
                baseline_results.append(br)
                logger.info(
                    "Baseline %s: status=%s", sf.stem, br.pipeline_status
                )

                # Validated run
                vr = self.run_single(sf_str, pipeline_type="validated", apply_injections=True)
                validated_results.append(vr)
                logger.info(
                    "Validated %s: status=%s", sf.stem, vr.pipeline_status
                )

        # Compute metrics
        baseline_propagation = self.metrics.compute_error_propagation_rate(baseline_results)
        validated_propagation = self.metrics.compute_error_propagation_rate(validated_results)
        detection = self.metrics.compute_detection_rate(validated_results, baseline_results)
        false_positive = self.metrics.compute_false_positive_rate(validated_results)

        n_injection_instances = sum(
            len(r.injected_errors) for r in baseline_results
        )

        comparison = {
            "propagation_rate_baseline": baseline_propagation["propagation_rate"],
            "propagation_rate_validated": validated_propagation["propagation_rate"],
            "reduction_percentage_points": (
                baseline_propagation["propagation_rate"]
                - validated_propagation["propagation_rate"]
            ),
            "detection_rate": detection["detection_rate"],
            "false_positive_rate": false_positive["false_positive_rate"],
        }

        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        results = {
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "n_scenarios": len(scenario_files) * n_repeats,
            "n_injection_instances": n_injection_instances,
            "baseline": {
                "error_propagation_rate": baseline_propagation,
                "run_results": [self._run_result_to_dict(r) for r in baseline_results],
            },
            "validated": {
                "detection_rate": detection,
                "false_positive_rate": false_positive,
                "error_propagation_rate": validated_propagation,
                "run_results": [self._run_result_to_dict(r) for r in validated_results],
            },
            "comparison": comparison,
        }

        # Save results
        results_dir = EXPERIMENTS_DIR.parent / "results"
        results_dir.mkdir(exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        output_file = results_dir / f"experiment_{ts_str}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", output_file)

        return results

    @staticmethod
    def _run_result_to_dict(r: RunResult) -> dict:
        return {
            "scenario_id": r.scenario_id,
            "pipeline_type": r.pipeline_type,
            "injected_errors": r.injected_errors,
            "agent4_output": r.agent4_output,
            "claim_ledger": r.claim_ledger,
            "validation_results": r.validation_results,
            "pipeline_status": r.pipeline_status,
            "error": r.error,
        }


def run_experiment(
    scenario_dir: str = "experiments/scenarios",
    n_repeats: int = 1,
) -> dict:
    runner = ExperimentRunner()
    return runner.run_experiment(scenario_dir=scenario_dir, n_repeats=n_repeats)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    results = run_experiment()
    print(json.dumps(results["comparison"], indent=2))
