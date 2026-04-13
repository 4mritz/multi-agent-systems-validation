from langgraph.graph import StateGraph, START, END
from mas_validation.state import PipelineState
from mas_validation.agents.scenario_analysis import run_scenario_analysis
from mas_validation.agents.actor_modeling import run_actor_modeling
from mas_validation.agents.impact_assessment import run_impact_assessment
from mas_validation.agents.decision_synthesis import run_decision_synthesis
from mas_validation.validators.validator_nodes import (
    validate_1_to_2,
    validate_2_to_3,
    validate_3_to_4,
)

graph = StateGraph(PipelineState)

# Add agent nodes
graph.add_node("agent1", run_scenario_analysis)
graph.add_node("agent2", run_actor_modeling)
graph.add_node("agent3", run_impact_assessment)
graph.add_node("agent4", run_decision_synthesis)

# Add validator nodes
graph.add_node("validator_1_to_2", validate_1_to_2)
graph.add_node("validator_2_to_3", validate_2_to_3)
graph.add_node("validator_3_to_4", validate_3_to_4)

# Wire edges: START → agent1 → v1→2 → agent2 → v2→3 → agent3 → v3→4 → agent4 → END
graph.add_edge(START, "agent1")
graph.add_edge("agent1", "validator_1_to_2")
graph.add_edge("validator_1_to_2", "agent2")
graph.add_edge("agent2", "validator_2_to_3")
graph.add_edge("validator_2_to_3", "agent3")
graph.add_edge("agent3", "validator_3_to_4")
graph.add_edge("validator_3_to_4", "agent4")
graph.add_edge("agent4", END)

validated_graph = graph.compile()
