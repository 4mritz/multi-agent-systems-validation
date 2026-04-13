from typing import Optional
from typing_extensions import TypedDict


class PipelineState(TypedDict):
    seed_document: str
    scenario_constraints: dict
    agent1_output: Optional[dict]
    agent2_output: Optional[dict]
    agent3_output: Optional[dict]
    agent4_output: Optional[dict]
    claim_ledger: list
    validation_results: dict  # {"1_to_2": [], "2_to_3": [], "3_to_4": []}
    fallback_flags: dict      # {"1_to_2": False, "2_to_3": False, "3_to_4": False}
    fallback_reasons: dict    # {"1_to_2": None, "2_to_3": None, "3_to_4": None}
    current_step: Optional[str]
    pipeline_status: str      # default: "pending"
