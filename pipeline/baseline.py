from langgraph.graph import StateGraph, START, END
from mas_validation.state import PipelineState
from mas_validation.agents.scenario_analysis import run_scenario_analysis
from mas_validation.agents.actor_modeling import run_actor_modeling
from mas_validation.agents.impact_assessment import run_impact_assessment
from mas_validation.agents.decision_synthesis import run_decision_synthesis

graph = StateGraph(PipelineState)

# Add agent nodes only (no validators)
graph.add_node("agent1", run_scenario_analysis)
graph.add_node("agent2", run_actor_modeling)
graph.add_node("agent3", run_impact_assessment)
graph.add_node("agent4", run_decision_synthesis)

# Wire edges: START → agent1 → agent2 → agent3 → agent4 → END
graph.add_edge(START, "agent1")
graph.add_edge("agent1", "agent2")
graph.add_edge("agent2", "agent3")
graph.add_edge("agent3", "agent4")
graph.add_edge("agent4", END)

baseline_graph = graph.compile()
