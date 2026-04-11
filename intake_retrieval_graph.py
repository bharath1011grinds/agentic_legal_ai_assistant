from langgraph.graph import StateGraph, START, END
 
from agents.agent_state import GraphState
from agents.claim_retriever import claim_retrieval_node
from agents.kb_miss_node import kb_miss_node

from agents.claim_retriever import claim_retrieval_node


from pipeline_agentic import _LANGFUSE_ENABLED

if _LANGFUSE_ENABLED:
    from observability import (
        observed_claim_retrieval_node   as claim_retrieval_node,
        observed_kb_miss_node as kb_miss_node
    )

#Routing after retrieval
def route_after_claim_retrieval(state: GraphState) -> str:
    grader = state.get("grader_output")
 
    if grader is None:
        print("[IntakeRetrievalRouter] No grader output — routing to kb_miss")
        return "kb_miss_node"
 
    decision = grader.rerouting_decision
    print(f"[IntakeRetrievalRouter] Decision: {decision} — {grader.confidence_reason}")
 
    if decision == "proceed":
        # Server picks up final_state and calls stream_intake_answer()
        return END
 
    return "kb_miss_node"

def build_intake_retrieval_graph():
    """
    Builds the retrieval graph for intake mode.
 
    Nodes:
      claim_retrieval_node — parallel retrieval + dedup + claim-aware grading
      kb_miss_node         — handles full KB miss gracefully
    """
    graph = StateGraph(GraphState)
 
    graph.add_node("claim_retrieval_node", claim_retrieval_node)
    graph.add_node("kb_miss_node",         kb_miss_node)
 
    graph.add_edge(START, "claim_retrieval_node")
    graph.add_conditional_edges("claim_retrieval_node", route_after_claim_retrieval)
    graph.add_edge("kb_miss_node", END)
 
    print("[IntakeRetrievalGraph] Compiled")
    return graph.compile()
