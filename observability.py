import os
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from langfuse import observe, get_client


lf = get_client()

if lf.auth_check():
    print("Connected to Langfuse Cloud!")
else:
    print("Authentication failed. Check your keys.")



#adding underscore to fn names to denote they are meant to be private(not enforced to bes)
from agents.context_resolver import resolve_context_node as _resolve_context_node
from agents.ambiguity_detector import detect_ambiguity_node as _detect_ambiguity_node
from agents.situation_classifier import classify_situation_node as _classify_situation_node
from agents.scope_guard import scope_guard_node as _scope_guard_node
from agents.kb_miss_node import kb_miss_node as _kb_miss_node
from agents.retriever        import retrieve_chunks_node  as _retrieve_chunks_node
from agents.relevance_grader import grade_chunks_node   as _grade_chunks_node
from agents.synthesizer      import stream_answer      as _synthesize_node
#from agents.relevance_grader import MAX_RETRIES


def _append_turn(history: list[dict], user_query: str, assistant_answer: str) -> list[dict]:
    """Append a completed turn to the history list and return the updated list."""
    return history + [
        {"role": "user",      "content": user_query},
        {"role": "assistant", "content": assistant_answer},
    ]


#wrapper for resolve context node
@observe(name="resolve_context")
def observed_resolve_context_node(state):
    
    #doing this cos dict might use call by reference and ip and op might end up having the same value in langfuse, create a copy instead./
    orig_input = str(state.get("raw_query"))

    result = _resolve_context_node(state)

    lf.update_current_span(input=orig_input, output=state.get("raw_query", ""))

    return result


@observe(name="detect_ambiguity")
def observed_detect_ambiguity_node(state):
    result    = _detect_ambiguity_node(state)
    ambiguity = result.get("ambiguity")
 
    lf.update_current_span(
        input  = state.get("raw_query", ""),
        output = str(ambiguity.is_ambiguous) if ambiguity else "unknown",
        metadata = {
            "is_ambiguous":          ambiguity.is_ambiguous if ambiguity else None,
            "clarifying_questions":  ambiguity.clarifying_questions if ambiguity else [],
            "reason":                ambiguity.reason if ambiguity else "",
        }
    )
    return result


@observe(name="classify_situation")
def observed_classify_situation_node(state):
    result    = _classify_situation_node(state)
    situation = result.get("situation")
 
    lf.update_current_span(
        input  = state.get("raw_query", ""),
        output = situation.situation if situation else "unknown",
        metadata = {
            "situation":       situation.situation       if situation else None,
            "document_filter": situation.document_filter if situation else [],
            "confidence":      situation.confidence      if situation else 0.0,
            "out_of_scope":    result.get("out_of_scope", False),
        }
    )
    return result


@observe(name="retrieve_chunks")
def observed_retrieve_chunks_node(state):
    """
    Traced wrapper for the Retriever node.
    Imported and used inside pipeline_agentic.py instead of the bare function.
    """
    result = _retrieve_chunks_node(state)
 
    lf.update_current_span(
        input    = state.get("rewritten_query", ""),
        output   = f"{len(result.get('chunks', []))} chunks retrieved",
        metadata = {
            "query":        state.get("rewritten_query", ""),
            "chunks_found": len(result.get("chunks", [])),
            "retry_count":  state.get("retry_count", 0),
        }
    )
    return result


@observe(name="grade_chunks")
def observed_grade_chunks_node(state):
    """Traced wrapper for RelevanceGrader node."""
    result = _grade_chunks_node(state)
 
    grader_output = result.get("grader_output")
    if grader_output:
        # Log per-chunk scores as a score array
        chunk_scores = {
            f"chunk_{g.chunk_index}_score": g.relevance_score
            for g in grader_output.chunk_grades
        }
 
        lf.update_current_span(
            input    = state.get("rewritten_query", ""),
            output   = grader_output.rerouting_decision,
            metadata={
                "routing_decision":  grader_output.rerouting_decision,
                "confidence_reason": grader_output.confidence_reason,
                "chunks_graded":     len(grader_output.chunk_grades),
                "chunks_passed":     len(grader_output.passed_chunks),
                "chunks_filtered":   len(grader_output.failed_chunks),
                "retry_count":       state.get("retry_count", 0),
                **chunk_scores,
            }
        )

        if grader_output.chunk_grades:
            top_score = max(g.relevance_score for g in grader_output.chunk_grades)
            #adds a score to the response of this particular step(grading)
            lf.score_current_span(
                name  = "top_chunk_relevance_score",
                value = top_score,
            )
 
    return result


 


@observe(name="scope_guard")
def observed_scope_guard_node(state):
    result = _scope_guard_node(state)
 
    lf.update_current_span(
        input    = state.get("raw_query", ""),
        output   = result.get("answer", "")[:300],
        metadata = {
            "situation": state.get("situation").situation if state.get("situation") else "unknown",
        }
    )
    return result

@observe(name="kb_miss")
def observed_kb_miss_node(state):
    result = _kb_miss_node(state)
 
    lf.update_current_span(
        input    = state.get("raw_query", ""),
        output   = result.get("answer", "")[:300],
        metadata = {
            "situation":       state.get("situation").situation if state.get("situation") else "unknown",
            "document_filter": state.get("document_filter", []),
            "grader_reason":   state.get("grader_output").confidence_reason if state.get("grader_output") else "",
        }
    )
    return result



@observe(name="synthesize")
def observed_synthesize_node(state):
    """Traced wrapper for Synthesizer node."""
    result = _synthesize_node(state)
 
    lf.update_current_span(
        input    = state.get("rewritten_query", ""),
        output   = result.get("answer", "")[:500],   # cap for dashboard readability
        metadata = {
            "answer_length":    len(result.get("answer", "")),
            "citations_count":  len(result.get("citations", [])),
            "image_refs_count": len(result.get("image_refs", [])),
            "used_tavily":      len(state.get("web_results") or []) > 0,
            "modality":         state["analysis"].modality if state.get("analysis") else "text",
        }
    )
    return result


@observe(name="synthesize_stream")
def observed_synthesize_stream_node(state):
    """Traced wrapper for Synthesizer node."""
    result = _synthesize_node(state)

    full_answer_buffer = []
    #Using a try...finally to make sure if the user stops the streaming mid-way, 
    #the update span still runs and results are logged.
    try:
        for chunk in result:
            yield chunk
            # Accumulate text for the log
            full_answer_buffer.append(chunk.replace("data:", "").replace("\n\n", ""))
    finally:
        # This block runs even if the connection is closed early
        full_text = "".join(full_answer_buffer)
        lf.update_current_span(
            output=full_text[:500],
            metadata={"answer_length": len(full_text)} # etc.
        )
 
    #lf.update_current_span(
    #    input    = state.get("rewritten_query", ""),
    #    output   = result.get("answer", "")[:500],   # cap for dashboard readability
    #)
    return result




@observe(name="observed_pipeline_run")
def observed_ask(graph, query: str, conversation_history : list[dict]):

    from agents.agent_state import initial_state
    """
    Root trace for a full pipeline run.
 
    All node spans (analyze, retrieve, grade, rewrite, synthesize)
    are nested under this single trace in the Langfuse dashboard.
 
    Logs aggregate metrics at the run level:
      - total retry count
      - final routing decision
      - whether Tavily was used
      - total chunks processed
    """
    #from pipeline_agentic import ask as _ask
 
    # Tag the root trace with the raw query

    lf.set_current_trace_io(input=query)

    if conversation_history is None:
        conversation_history = []
    capped_history = conversation_history[-6:]

    #Making observed_ask independant of the ask in pipeline_agentic.py to get rid of circular imports
    #This function will be primarily called when the pipeline_agentic.py is run..
    #The graph from pipeline_agentic will be passed as agruments.
    #Also, pipeline_agentic.py will use the observed functions defined in this file when langfuse is enabled.
    state = initial_state(query, history=capped_history)    
    result = graph.invoke(state)



    grader_output = result.get("grader_output")
    analysis      = result.get("analysis")
    retry_count   = result.get("retry_count", 0)
    used_tavily   = len(result.get("web_results") or []) > 0
 
    lf.set_current_trace_io(
        output = result.get("answer", "")[:500]
    )
    lf.update_current_span(
        metadata = {
            "query_type":       analysis.query_type if analysis else "unknown",
            "modality":         analysis.modality if analysis else "text",
            "retry_count":      retry_count,
            "used_tavily":      used_tavily,
            "routing_decision": grader_output.rerouting_decision if grader_output else "unknown",
            "citations_count":  len(result.get("citations", [])),
            "image_refs_count": len(result.get("image_refs", [])),
        }
    )
 
    # ------------------------------------------------------------------
    # Root trace scores — visible as metrics on the Langfuse dashboard
    # ------------------------------------------------------------------
    """
    lf.score_current_trace(
        name    = "retry_rate",
        value   = round(retry_count / MAX_RETRIES, 2),
        comment = f"{retry_count}/{MAX_RETRIES} retries used",
    )

    lf.score_current_trace(
        name  = "tavily_fallback_used",
        value = 1.0 if used_tavily else 0.0,
    )

    """
    if grader_output and grader_output.chunk_grades:
        top_score = max(g.relevance_score for g in grader_output.chunk_grades)
        lf.score_current_trace(
            name  = "top_grader_score",
            value = round(top_score, 4),
        )
    #updates the history when called in observed mode too
    updated_history = _append_turn(capped_history, query, result.get("answer", ""))
 
    # Print pipeline stats to console (same as original ask())
    _print_result(result, query)
 
    return result, updated_history


# ── Intake Pipeline Observed Nodes ────────────────────────────────────────────

from agents.intake_agent    import intake_agent_node     as _intake_agent_node
from agents.intake_agent    import decompose_claims_node as _decompose_claims_node
from agents.claim_retriever import claim_retrieval_node  as _claim_retrieval_node
from agents.edit_detector   import detect_edit_node      as _detect_edit_node
from agents.intake_synthesizer import (
    intake_synthesize_node as _intake_synthesize_node,
    stream_intake_answer   as _stream_intake_answer,
)


@observe(name="intake_agent")
def observed_intake_agent_node(state):
    result     = _intake_agent_node(state)
    case_log   = result.get("case_log")
    turn_count = result.get("intake_turn_count", 0)

    lf.update_current_span(
        input  = state.get("raw_query", ""),
        output = result.get("answer", "")[:300],
        metadata={
            "turn_count":              turn_count,
            "intake_ready":            "__INTAKE_READY__" in result.get("clarification_questions", []),
            "non_negotiables_filled":  case_log.non_negotiables_filled() if case_log else False,
            "incident_type":           case_log.incident_type            if case_log else None,
            "relief_sought":           case_log.relief_sought            if case_log else None,
        }
    )
    return result


@observe(name="decompose_claims")
def observed_decompose_claims_node(state):
    result     = _decompose_claims_node(state)
    decomposed = result.get("decomposed_claims")

    lf.update_current_span(
        input  = state.get("case_log").to_context_string() if state.get("case_log") else "",
        output = f"{len(decomposed.claims)} claims decomposed" if decomposed else "no claims",
        metadata={
            "claims_count": len(decomposed.claims) if decomposed else 0,
            "claims":       [
                {"id": c.claim_id, "text": c.claim_text[:80], "doc_filter": c.doc_filter}
                for c in decomposed.claims
            ] if decomposed else [],
        }
    )
    return result


@observe(name="claim_retrieval")
def observed_claim_retrieval_node(state):
    result        = _claim_retrieval_node(state)
    grader_output = result.get("grader_output")
    claim_results = result.get("claim_results", [])

    # Per-claim pass rates as flat metadata
    per_claim_stats = {
        f"claim_{r['claim_id']}_passed": r["passed"]
        for r in claim_results
    }
    per_claim_routing = {
        f"claim_{r['claim_id']}_routing": r["routing"]
        for r in claim_results
    }

    lf.update_current_span(
        input  = f"{len(state.get('decomposed_claims').claims)} claims" if state.get("decomposed_claims") else "0 claims",
        output = grader_output.rerouting_decision if grader_output else "unknown",
        metadata={
            "routing_decision":  grader_output.rerouting_decision  if grader_output else "N/A",
            "confidence_reason": grader_output.confidence_reason   if grader_output else "",
            "total_passed":      len(grader_output.passed_chunks)  if grader_output else 0,
            "total_graded":      len(grader_output.chunk_grades)   if grader_output else 0,
            "claims_count":      len(claim_results),
            **per_claim_stats,
            **per_claim_routing,
        }
    )

    if grader_output and grader_output.chunk_grades:
        top_score = max(g.relevance_score for g in grader_output.chunk_grades)
        lf.score_current_span(name="top_chunk_relevance_score", value=top_score)

    return result


@observe(name="detect_edit")
def observed_detect_edit_node(state):
    result = _detect_edit_node(state)

    lf.update_current_span(
        input  = state.get("raw_query", ""),
        output = result.get("answer", "")[:300],
        metadata={
            "edit_detected":   result.get("edit_detected", False),
            "edited_fields":   result.get("edited_fields", []),
            "awaiting_contradiction_resolution": result.get("awaiting_contradiction_resolution", False),
        }
    )
    return result


@observe(name="intake_synthesize")
def observed_intake_synthesize_node(state):
    result        = _intake_synthesize_node(state)
    claim_results = state.get("claim_results") or []

    lf.update_current_span(
        input  = state.get("case_log").to_context_string() if state.get("case_log") else "",
        output = result.get("answer", "")[:500],
        metadata={
            "answer_length":   len(result.get("answer", "")),
            "citations_count": len(result.get("citations", [])),
            "claims_count":    len(claim_results),
            "used_tavily":     len(state.get("web_results") or []) > 0,
            "covered_claims":  sum(1 for r in claim_results if r.get("routing") == "proceed"),
            "missed_claims":   sum(1 for r in claim_results if r.get("routing") == "detect"),
        }
    )
    return result


@observe(name="stream_intake_answer")
def observed_stream_intake_answer(state):
    gen = _stream_intake_answer(state)

    full_answer_buffer = []
    try:
        for chunk in gen:
            yield chunk
            full_answer_buffer.append(chunk.replace("data:", "").replace("\n\n", ""))
    finally:
        full_text     = "".join(full_answer_buffer)
        claim_results = state.get("claim_results") or []
        lf.update_current_span(
            output=full_text[:500],
            metadata={
                "answer_length": len(full_text),
                "claims_count":  len(claim_results),
                "used_tavily":   len(state.get("web_results") or []) > 0,
            }
        )


 
def _print_result(result: dict, query: str):
    grader_output = result.get("grader_output")
    analysis      = result.get("analysis")
 
    print(f"\n{'='*60}")
    print(f"ANSWER:\n{result.get('answer', '')}")
    print(f"{'='*60}")
 
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
 
    image_refs = result.get("image_refs", [])
    if image_refs:
        print(f"\nFIGURES ({len(image_refs)}):")
        for ref in image_refs:
            print(f"  [{ref.get('figure_type','?')}] {ref.get('caption','')[:80]}")
 
    if grader_output:
        print(f"\nPIPELINE STATS:")
        print(f"  Query type      : {analysis.query_type if analysis else 'N/A'}")
        print(f"  Modality        : {analysis.modality if analysis else 'N/A'}")
        print(f"  Retry count     : {result.get('retry_count', 0)}")
        print(f"  Routing decision: {grader_output.rerouting_decision}")
        print(f"  Tavily used     : {'yes' if result.get('web_results') else 'no'}")
        print(f"  Chunks graded   : {len(grader_output.chunk_grades)}")
        print(f"  Chunks passed   : {len(grader_output.passed_chunks)}")


def flush():

    #we NEED this flush because, the info gets sent from RAM to Langfuse UI in batches and at regular time intervals
    #whichever of the two happens, info flows.
    #and when our accumulated info does not fill the whole batch nor take the minimum for the push to get triggered, 
    #our script ends and the RAM is cleared and the info gets lost permanently.
    #To prevent this, we FLUSH the info in the RAM to langfuse when the script ends, manually...

    lf.flush()
    print("[Langfuse] All traces flushed.")

 
 
def shutdown():
    """
    Clean shutdown — flush all pending traces and close the SDK.
    Call at script exit or FastAPI shutdown event.
 
    FastAPI usage:
        @app.on_event("shutdown")
        async def on_shutdown():
            from observability import shutdown
            shutdown()
    """
    lf.shutdown()
    print("[Langfuse] SDK shut down cleanly.")