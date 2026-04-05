import os
import math
import sys
from typing import Literal
from dotenv import load_dotenv
 
from langchain_core.documents import Document
 
from .agent_state import GraphState
from .models import ChunkGrade, GraderOutput
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hybrid_retriever_phase2 import get_cross_encoder
 
load_dotenv()
 
SCORE_THRESHOLD    = 0.4
MIN_PASSING_CHUNKS = 2
 
 
def _sigmoid(score: float) -> float:
    return 1.0 / (1 + math.exp(-score))
 
 
def _make_routing_decision(
    passed_count: int,
    norm_scores:  list[float],
) -> tuple[Literal["proceed", "detect"], str]:
 
    top_score = max(norm_scores) if norm_scores else 0.0
 
    if passed_count >= MIN_PASSING_CHUNKS:
        return "proceed", f"{passed_count} chunks passed (top: {top_score:.2f})"
 
    return "detect", f"Only {passed_count}/{MIN_PASSING_CHUNKS} passed (top: {top_score:.2f})"
 
 
def _auto_reason(score: float) -> str:
    if score >= 0.8:
        return "High relevance — directly addresses the query."
    if score >= 0.4:
        return "Partial relevance — related topic but incomplete match."
    return "Low relevance — does not sufficiently address the query."
 
 
def grade_chunks_node(state: GraphState) -> dict:
    """
    Reads : state["rewritten_query"], state["chunks"]
    Writes: state["grader_output"]
    """
    query  = state.get("rewritten_query") or state.get("raw_query", "")
    chunks = state.get("chunks", [])
 
    if not chunks:
        routing, reason = _make_routing_decision(passed_count=0, norm_scores=[])
        return {
            "grader_output": GraderOutput(
                chunk_grades       = [],
                passed_chunks      = [],
                failed_chunks      = [],
                rerouting_decision = routing,
                confidence_reason  = "No chunks retrieved. " + reason,
            )
        }
 
    cross_encoder = get_cross_encoder()
    pairs       = [(query, doc.page_content) for doc in chunks]
    raw_scores  = cross_encoder.predict(pairs)
    norm_scores = [_sigmoid(s) for s in raw_scores]
 
    print(f"\n[RelevanceGrader] Scores: {[round(s, 2) for s in norm_scores]}")
 
    chunk_grades  = []
    passed_chunks = []
    failed_chunks = []
 
    for i, (doc, score) in enumerate(zip(chunks, norm_scores)):
        passed = score >= SCORE_THRESHOLD
        grade  = ChunkGrade(
            chunk_index     = i,
            relevance_score = score,
            passed          = passed,
            reason          = _auto_reason(score),
        )
        chunk_grades.append(grade)
 
        if passed:
            doc.metadata["grader_score"]  = score
            doc.metadata["grader_reason"] = grade.reason
            passed_chunks.append(doc)
        else:
            failed_chunks.append(doc)
 
    routing, confidence_reason = _make_routing_decision(
        passed_count = len(passed_chunks),
        norm_scores  = norm_scores,
    )
 
    print(f"\n[RelevanceGrader] Results:")
    print(f"  Chunks graded    : {len(chunks)}")
    print(f"  Passed           : {len(passed_chunks)}")
    print(f"  Filtered         : {len(failed_chunks)}")
    print(f"  Routing decision : {routing} — {confidence_reason}")
    for grade in chunk_grades:
        status = "PASS" if grade.passed else "FAIL"
        print(f"  {status} [{grade.chunk_index}] score={grade.relevance_score:.2f}")
 
    return {
        "grader_output": GraderOutput(
            chunk_grades       = chunk_grades,
            passed_chunks      = passed_chunks,
            failed_chunks      = failed_chunks,
            rerouting_decision = routing,
            confidence_reason  = confidence_reason,
        )
    }