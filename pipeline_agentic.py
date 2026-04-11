import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
 
from agents.agent_state import GraphState, initial_state
from agents.retriever import retrieve_chunks_node, get_retriever
from agents.relevance_grader import grade_chunks_node
from agents.synthesizer import synthesize_node, stream_answer
from agents.context_resolver import resolve_context_node
from agents.ambiguity_detector import detect_ambiguity_node
from agents.situation_classifier import classify_situation_node
from agents.scope_guard import scope_guard_node
from agents.kb_miss_node import kb_miss_node

from hybrid_retriever_phase2 import build_hybrid_retriever
 
load_dotenv()

# Built once at module load, reused across all queries
_retriever = None

_LANGFUSE_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

if _LANGFUSE_ENABLED:
    print("[Langfuse] Observability enabled — traced nodes active")
if _LANGFUSE_ENABLED:
    from observability import (
        observed_retrieve_chunks_node    as retrieve_chunks_node,
        observed_grade_chunks_node       as grade_chunks_node,
        observed_synthesize_node         as synthesize_node,
        observed_synthesize_stream_node  as stream_answer,
        observed_detect_ambiguity_node   as detect_ambiguity_node,    
        observed_classify_situation_node as classify_situation_node,    
        observed_resolve_context_node    as resolve_context_node,     
        observed_scope_guard_node        as scope_guard_node,           
        observed_kb_miss_node            as kb_miss_node,               
        observed_ask,
        flush as langfuse_flush,
        shutdown as langfuse_shutdown
    )
else:
    print("[Langfuse] Keys not found — running without observability")
    langfuse_flush = lambda: None   # no-op if Langfuse not configured



def clarification_end_node(state: GraphState) -> dict:
    """
    Packages the ambiguity clarifying questions as the 'answer' so the server
    can detect and forward them to the UI as a special event.
    """
    ambiguity = state.get("ambiguity")
    questions = ambiguity.clarifying_questions if ambiguity else []
    # Store questions in state so server.py can pick them up
    print("at clarification end node")
    return {
        "answer": "__CLARIFICATION__",
        "clarifying_questions": questions,
        "citations": [],
        "image_refs": [],
    }

def route_after_ambiguity(state: GraphState) -> str:
    ambiguity = state.get("ambiguity")
    if ambiguity and ambiguity.is_ambiguous:
        print(f"[Router] Ambiguous query — returning clarifying questions to UI")
        return "clarification_end_node" #Server intercepts and sends the questions
    
    return "classify_situation_node"

 
def route_after_situation(state: GraphState) -> str:
    if state.get("out_of_scope"):
        print(f"[Router] Out of scope — routing to ScopeGuard")
        return "scope_guard_node"
    if state.get("answer") == "__CLARIFICATION__":
        return "kb_miss_node" #need to write a new node for this that tells the user, "the question is within scope
                            # but we dont have enough data to support them at the moment"
    if not state.get("chunks"):
        return "retrieve_chunks_node" 
    
    return "kb_miss_node"#need to write a new node for this that tells the user, "the question is within scope
                            # but we dont have enough data to support them at the moment"



#NOTE: This node runs ONLY after the query grading is done.
def route_after_grading(state: GraphState) -> str:

    if not state.get('grader_output'):
        print("[Router] grader_output is None — defaulting to synthesize")
        return 'synthesizer_node'
    
    grader_output = state.get('grader_output')
    decision = grader_output.rerouting_decision
    print(f"\n[Router] Decision: {decision} — {grader_output.confidence_reason}")

    route_map = {
        "proceed":  "classify_situation_node",
        "detect":  "detect_ambiguity_node"
    }

    return route_map.get(decision, "synthesizer_node")


def _append_turn(history: list[dict], user_query: str, assistant_answer: str) -> list[dict]:
    """Append a completed turn to the history list and return the updated list."""
    return history + [
        {"role": "user",      "content": user_query},
        {"role": "assistant", "content": assistant_answer},
    ]


def build_graph():

    #retriever = get_retriever()

    graph = StateGraph(GraphState)

    #Register all the nodes to the graph... retrieve_chunks_node defined in this file.
    graph.add_node("context_resolver_node", resolve_context_node)
    graph.add_node("detect_ambiguity_node", detect_ambiguity_node)
    graph.add_node("classify_situation_node", classify_situation_node)
    graph.add_node("retrieve_chunks_node", retrieve_chunks_node)
    graph.add_node("clarification_end_node", clarification_end_node)
    graph.add_node("relevance_grader_node", grade_chunks_node)
    graph.add_node("scope_guard_node", scope_guard_node)
    graph.add_node("synthesizer_node", synthesize_node)
    graph.add_node("kb_miss_node", kb_miss_node)



    graph.add_edge(START, "context_resolver_node")

    #detect ambiguity after resolving the query with past context.
    graph.add_edge("context_resolver_node", "retrieve_chunks_node")
    graph.add_edge("retrieve_chunks_node", "relevance_grader_node")
    graph.add_conditional_edges("relevance_grader_node",route_after_grading)
    graph.add_conditional_edges("detect_ambiguity_node", route_after_ambiguity)
    graph.add_conditional_edges("classify_situation_node", route_after_situation)

    #Conditional edge - the output of the function in the 2nd arg should be one of the registered nodes(node names).
    #should add a routing map if the router returns something apart from the node_names.... not needed here.

    graph.add_edge("clarification_end_node", END)
    graph.add_edge("kb_miss_node", END)
    graph.add_edge("scope_guard_node", END)
    graph.add_edge("synthesizer_node", END)


    return graph.compile()


# mirrors route_after_grading but routes proceed to END instead of synthesizer_node
#Truncates the graph instead of proceeding to the synth_node.
def route_after_grading_no_synth(state: GraphState) -> str:
    if not state.get('grader_output'):
        return END
    grader_output = state.get('grader_output')
    decision = grader_output.rerouting_decision
    print(f"\n[Router] Decision: {decision} — {grader_output.confidence_reason}")
    route_map = {
        "proceed": END,       # server.py picks up state and pipes into stream_answer()
        "detect": "detect_ambiguity_node",
    }
    return route_map.get(decision, END)

#Below is a truncated graph that stops right before the synthesize node. 
#This is done cos, stream_answer does not return a dict and all langgraph nodes MUST return a dict. 
#Taking the synth node off the below graph allows the stream_answer to directly return a stream.
#The final_state from this graph is passed to stream_answer that directly send the stream to the server
def build_graph_no_synth():

    graph = StateGraph(GraphState)

    #Register all the nodes to the graph... retrieve_chunks_node defined in this file.
    graph.add_node("context_resolver_node", resolve_context_node)
    graph.add_node("detect_ambiguity_node", detect_ambiguity_node)
    graph.add_node("classify_situation_node", classify_situation_node)
    graph.add_node("retrieve_chunks_node", retrieve_chunks_node)
    graph.add_node("relevance_grader_node", grade_chunks_node)
    graph.add_node("scope_guard_node", scope_guard_node)
    graph.add_node("clarification_end_node", clarification_end_node)
    #graph.add_node("synthesizer_node", synthesize_node)
    graph.add_node("kb_miss_node", kb_miss_node)


    graph.add_edge(START, "context_resolver_node")

    #detect ambiguity after resolving the query with past context.
    graph.add_edge("context_resolver_node", "detect_ambiguity_node")
    graph.add_conditional_edges("detect_ambiguity_node", route_after_ambiguity)
    graph.add_conditional_edges("classify_situation_node", route_after_situation)
    graph.add_edge("retrieve_chunks_node", "relevance_grader_node")
    graph.add_conditional_edges("relevance_grader_node", route_after_grading_no_synth)


   
    #Conditional edge - the output of the function in the 2nd arg should be one of the registered nodes(node names).
    #should add a routing map if the router returns something apart from the node_names.... not needed here.
    #graph.add_conditional_edges("relevance_grader_node",route_after_grading)
    graph.add_edge("scope_guard_node", END)
    graph.add_edge("clarification_end_node", END)
    graph.add_edge("kb_miss_node", END)

    #graph.add_edge("synthesizer_node", END)

    return graph.compile()





def ask(graph, raw_query: str, conversation_history : list[dict]|None = None):



    if conversation_history is None:
        conversation_history = []
 
    # Cap history before passing in - keeps prompt size bounded.
    # 6 entries = 3 full exchanges (user + assistant each).
    capped_history = conversation_history[-6:]

    #Call the observed graph if observability is TRUE
    if _LANGFUSE_ENABLED:
        result, updated_history =  observed_ask(graph, raw_query, conversation_history=capped_history)
        
        #updates the history when called in observed mode too
        updated_history = _append_turn(capped_history, raw_query, result.get("answer", ""))
        return result, updated_history
    
    print(f"\n{'='*60}")
    print(f"Query: {raw_query}")
    print(f"{'='*60}")

    #THIS is where the state becomes aware of the history...State is not maintained across invoke() calls/traces...
    state = initial_state(raw_query=raw_query, history=capped_history)
    result = graph.invoke(state)

 # ------------------------------------------------------------------
    # Display answer
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"ANSWER:\n{result['answer']}")
    print(f"{'='*60}")
 
    # ------------------------------------------------------------------
    # Display citations
    # ------------------------------------------------------------------
    citations = result.get("citations", [])
    if citations:
        print(f"\nCITATIONS ({len(citations)}):")
        for c in citations:
            tag = "[WEB]  " if c["type"] == "web" else "[ARXIV]"
            print(f"  {tag} {c['title']}")
            if c.get("authors"):
                print(f"           {c['authors']}")
            if c.get("url"):
                print(f"           {c['url']}")
 
    # ------------------------------------------------------------------
    # Display image refs
    # ------------------------------------------------------------------
    image_refs = result.get("image_refs", [])
    if image_refs:
        print(f"\nFIGURES ({len(image_refs)}):")
        for ref in image_refs:
            print(f"  [{ref.get('figure_type', '?')}] {ref.get('caption', '')[:80]}")
            print(f"           {ref.get('image_path', '')}")
 
    # ------------------------------------------------------------------
    # Display pipeline stats
    # ------------------------------------------------------------------
    grader_output = result.get("grader_output")
    analysis      = result.get("analysis")
    if grader_output:
        print(f"\nPIPELINE STATS:")
        print(f"  Query type      : {analysis.query_type if analysis else 'N/A'}")
        print(f"  Modality        : {analysis.modality if analysis else 'N/A'}")
        print(f"  Retry count     : {result.get('retry_count', 0)}")
        print(f"  Routing decision: {grader_output.rerouting_decision}")
        print(f"  Tavily used     : {'yes' if result.get('web_results') else 'no'}")
        print(f"  Chunks graded   : {len(grader_output.chunk_grades)}")
        print(f"  Chunks passed   : {len(grader_output.passed_chunks)}")

    updated_history = _append_turn(capped_history, raw_query, result.get("answer", ""))
 
    return result, updated_history
 


def ask_voice(pipeline, audio_path: str, conversation_history : list[dict]|None = None):
    """
    Run the full agentic pipeline for a voice query.
 
    Transcribes audio using existing speech_handler_phase2.py,
    then passes the transcript into ask().
 
    Args:
        pipeline   : compiled LangGraph from build_agentic_pipeline()
        audio_path : path to audio file (mp3, wav, m4a, flac, etc.)
 
    Returns:
        Same dict as ask()
    """
    from speech_handler_phase2 import speech_to_query
 
    print(f"\n[Voice] Transcribing: {audio_path}")
    query = speech_to_query(audio_path=audio_path)

    if conversation_history is None:
        conversation_history = []
 
    capped_history = conversation_history[-6:]

    if not query:
        print("[Voice] Could not extract a valid query from audio.")
        return {"answer": "Could not transcribe audio.", "citations": [], "image_refs": []}, conversation_history or []


    print(f"[Voice] Transcript: '{query}'")

    #Call the observed graph if observability is TRUE
    if _LANGFUSE_ENABLED:
        result =  observed_ask(pipeline, query, conversation_history=capped_history)

        # Still update history even in observed mode
        updated_history = _append_turn(capped_history, query, result.get("answer", ""))
        return result, updated_history
    
    return ask(pipeline, query, conversation_history=capped_history)


if __name__ == "__main__":
    import sys
 
    history = []
    # ------------------------------------------------------------------
    # Optional: run ingestion first if index doesn't exist
    # ------------------------------------------------------------------
    if not os.path.exists("vectorstore/"):
        print("No vectorstore found — running ingestion first...")
        from ingest_legal import ingest_legal
        ingest_legal()

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    pipeline = build_graph()
 
    # ------------------------------------------------------------------
    # Test queries — exercise all pipeline paths
    # ------------------------------------------------------------------
    test_queries = [
        # Happy path — should find good chunks in corpus
        #"What are the main causes of hallucination in large language models?",
 
        # Comparative — tests query_type-aware rewriting if needed
       # "How does RAG compare to fine-tuning for reducing hallucinations?",
      
        # Likely needs Tavily — recent/specific
        #"What are the latest RAG papers published in 2025?",

        "Secondary structural elements in homologue protiens",
 
        # Image modality — tests figure caption retrieval
        #"Show me architecture diagrams for RAG systems",
    ]
 
    # Voice test (uncomment and provide a real audio file to test)
    # result = ask_voice(pipeline, "test_audio.mp3")
 
    while True:
        q = input("You: ")
        if q == "exit()":
            break
        result, history = ask(pipeline, q,conversation_history=history)
        print("\n")

    # Flush all Langfuse traces before exit
    langfuse_flush()
    langfuse_shutdown()
