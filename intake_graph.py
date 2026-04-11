from langgraph.graph import StateGraph, START, END
 
from agents.agent_state import GraphState
from agents.edit_detector import detect_edit_node
from agents.intake_agent import intake_agent_node, decompose_claims_node

from agents.intake_agent    import intake_agent_node, decompose_claims_node
from agents.edit_detector   import detect_edit_node
from agents.claim_retriever import claim_retrieval_node
from agents.intake_synthesizer import intake_synthesize_node

from pipeline_agentic import _LANGFUSE_ENABLED

if _LANGFUSE_ENABLED:
    from observability import (
        observed_intake_agent_node      as intake_agent_node,
        observed_decompose_claims_node  as decompose_claims_node,
        observed_detect_edit_node       as detect_edit_node,
    )


def route_after_edit_detection(state: GraphState) -> str:
    """
    If an edit was detected and handled, skip the intake agent.
    The answer is already set by edit_detector.
    """
    if state.get("edit_detected"):
        print("[IntakeRouter] Edit detected — skipping intake agent")
        return END
 
    return "intake_agent_node"

def build_intake_graph():

    """
    Builds the intake conversation graph.
    Invoked on every user turn during intake mode.
 
    Nodes:
      edit_detector_node  - checks if message is a correction
      intake_agent_node   - extracts fields, asks next question
    """
    graph = StateGraph(GraphState)
 
    graph.add_node("edit_detector_node", detect_edit_node)
    graph.add_node("intake_agent_node",  intake_agent_node)
 
    graph.add_edge(START, "edit_detector_node")
    graph.add_conditional_edges("edit_detector_node", route_after_edit_detection)
    graph.add_edge("intake_agent_node", END)
 
    print("[IntakeGraph] Intake conversation graph compiled")
    return graph.compile()
 

def build_decomposition_graph():
    """
    Builds the one-shot decomposition graph.
    Invoked once when the user clicks "Get Legal Advice".
 
    Nodes:
      decompose_claims_node - decomposes the case log into legal claims
    """
    graph = StateGraph(GraphState)
 
    graph.add_node("decompose_claims_node", decompose_claims_node)
 
    graph.add_edge(START, "decompose_claims_node")
    graph.add_edge("decompose_claims_node", END)
 
    print("[IntakeGraph] Decomposition graph compiled")
    return graph.compile()