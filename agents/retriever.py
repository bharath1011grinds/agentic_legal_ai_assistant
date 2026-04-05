"""
agents/retriever.py
 
Retriever node — wraps the existing HybridRetriever from hybrid_retriever_phase2.py.
 
Moved out of pipeline_v3_agentic.py into the agents package so that:
  1. observability.py can import and wrap it cleanly (no circular imports)
  2. The Langfuse node swap in pipeline_v3_agentic.py covers ALL 5 nodes uniformly
 
State: reads  → rewritten_query, raw_query
       writes → chunks
"""
 
import sys
import os
from .agent_state import GraphState

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
 
from hybrid_retriever_phase2 import build_hybrid_retriever
 
# ---------------------------------------------------------------------------
# Singleton retriever — built once, reused across all queries
# ---------------------------------------------------------------------------
 
_retriever = None
 
 
def get_retriever(index_path: str = "vectorstore/knowyourrights/"):
    """
    Load the HybridRetriever singleton.
    Called once at pipeline startup, reused for every query.
    """
    global _retriever
    if _retriever is None:
        print("[Retriever] Loading hybrid retriever...")
        _retriever = build_hybrid_retriever(
            index_path       = index_path,
            k                = 5,
            bm25_candidates  = 20,
            faiss_candidates = 20,
        )
        print("[Retriever] Ready")
    return _retriever
 
 
# ---------------------------------------------------------------------------
# Node function — operates on PipelineState
# ---------------------------------------------------------------------------
 
def retrieve_chunks_node(state: GraphState) -> dict:
    """
    LangGraph node: retrieves relevant chunks using HybridRetriever.
 
    Uses rewritten_query (from analyzer or rewriter) as the search query.
    Falls back to raw_query if rewritten_query is empty.
 
    Reads  : state["rewritten_query"], state["raw_query"]
    Writes : state["chunks"]
    """
    query = state.get("rewritten_query") or state.get("raw_query", "")
 
    print(f"\n[Retriever] Query: '{query}'")
 
    retriever = get_retriever()
 
    try:
        # _get_relevant_documents is the BaseRetriever interface —
        # internally runs BM25 + MMR + cross-encoder reranking
        chunks = retriever._get_relevant_documents(query)
    except Exception as e:
        print(f"[Retriever] Error: {e}")
        chunks = []
 
    print(f"[Retriever] Retrieved {len(chunks)} chunks")
    #Returning the dict alone suffices, langgraph makes the changes to the state once the node is registered.
    return {"chunks": chunks}