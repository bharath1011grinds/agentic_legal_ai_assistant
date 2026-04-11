import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
 
from langchain_core.documents import Document
 
from .agent_state import GraphState
from .models import LegalClaim, ChunkGrade, GraderOutput
from hybrid_retriever_phase2 import get_cross_encoder
from agents.retriever import get_retriever

 
load_dotenv()

SCORE_THRESHOLD    = 0.8
MIN_PASSING_CHUNKS = 2
 
_API_KEYS = [
    k for k in [
        os.environ.get("GROQ_API_KEY"),
        os.environ.get("GROQ_API_KEY_2"),
        os.environ.get("GROQ_API_KEY_3"),
        os.environ.get("GROQ_API_KEY_4"),
    ] if k
]

#Init a global retriever to be shared by the workers, load once, use multiple times
SHARED_RETRIEVER = get_retriever()
SHARED_CROSS_ENCODER = get_cross_encoder()

def _sigmoid(score: float) -> float:
    return 1.0 / (1 + math.exp(-score))
 
 
def _auto_reason(score: float) -> str:
    if score >= 0.8:
        return "High relevance — directly addresses the claim."
    if score >= 0.4:
        return "Partial relevance — related topic but incomplete match."
    return "Low relevance — does not sufficiently address the claim."


def _retrieve_for_claim(claim: LegalClaim) -> tuple[int, list[Document]]:

 
    #retriever = get_retriever()

    try: 
        chunks = SHARED_RETRIEVER._get_relevant_documents(claim.claim_text)

        for doc in chunks:

            #tag the claim_id to the chunk to indicate the claim it was retrieved for 
            doc.metadata["source_claim_id"] = claim.claim_id
        print(f"[ClaimRetriever] Claim {claim.claim_id}: {len(chunks)} chunks | '{claim.claim_text[:60]}'")
        return claim.claim_id, chunks
  
    except Exception as e:
        print(f"[ClaimRetriever] Claim {claim.claim_id} error: {e}")
        return claim.claim_id, []
    

def _parallel_retrieve(claims : list[LegalClaim]) -> dict[int, list[Document]]:

    results: dict[int, list[Document]] = {}
    max_workers = min(len(claims), len(_API_KEYS) if _API_KEYS else 2, 4)

    print(f"[ClaimRetriever] {len(claims)} claims, {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor: #max_workers -> No of threads
        
        #computing a futures dict, this is where the parallel processing takes place, 1 in each thread...
        futures = {executor.submit(_retrieve_for_claim,  claim) : claim for claim in claims}

        for future in as_completed(futures): #this for loop processes whichever object is returned first, because of as_completed

            #future.result() extracts the result from the futures object once the task is completed
            #code is paused if the task is not completed, but as_completed takes care of that check.
            claim_id, chunks = future.result()
            results[claim_id] = chunks

    return results



#Using the reranker in our hybrid_retriever
def _grade_claim_chunks(
    claim:  LegalClaim,
    chunks: list[Document],
) -> tuple[list[ChunkGrade], list[Document], list[Document], str]:
    """
    Grades chunks for one claim using the cross-encoder.
    Scoring pair: (claim_text, chunk) — not the global query.
    Returns (chunk_grades, passed_chunks, failed_chunks, routing_decision).
    """
    if not chunks:
        return [], [], [], "detect"
 
    #cross_encoder = get_cross_encoder()
    pairs       = [(claim.claim_text, doc.page_content) for doc in chunks]
    raw_scores  = SHARED_CROSS_ENCODER.predict(pairs)
    norm_scores = [_sigmoid(s) for s in raw_scores]
 
    chunk_grades  = []
    passed_chunks = []
    failed_chunks = []
 
    for i, (doc, score) in enumerate(zip(chunks, norm_scores)):
        passed = score >= SCORE_THRESHOLD
        grade  = ChunkGrade(
            chunk_index     = i,
            relevance_score = round(score, 4),
            passed          = passed,
            reason          = _auto_reason(score),
        )
        chunk_grades.append(grade)
 
        if passed:
            doc.metadata["grader_score"]     = round(score, 4)
            doc.metadata["grader_reason"]    = grade.reason
            doc.metadata["graded_for_claim"] = claim.claim_id
            passed_chunks.append(doc)
        else:
            failed_chunks.append(doc)
 
    routing = "proceed" if len(passed_chunks) >= MIN_PASSING_CHUNKS else "detect"
    print(f"[ClaimRetriever] Claim {claim.claim_id} → {len(passed_chunks)}/{len(chunks)} passed | {routing}")
 
    return chunk_grades, passed_chunks, failed_chunks, routing


def _aggregate_routing(claim_routing: dict[int, str]) -> tuple[str, str]:
    decisions     = list(claim_routing.values())
    proceed_count = decisions.count("proceed")
    detect_count  = decisions.count("detect")
 
    if proceed_count == len(decisions):
        return "proceed", f"All {proceed_count} claims have sufficient chunks"
    if detect_count == len(decisions):
        return "detect",  "No claims have sufficient chunks — KB miss"
    return "partial", f"{proceed_count}/{len(decisions)} claims covered"

def claim_retrieval_node(state: GraphState) -> dict:
    """
    LangGraph node: parallel retrieval + claim-aware grading.
 
    Reads : state["decomposed_claims"]
    Writes: state["chunks"]        — all passed chunks across claims
            state["grader_output"] — aggregate grader output
            state["claim_results"] — per-claim detail for synthesizer
    """
    decomposed = state.get("decomposed_claims")
 
    if decomposed is None or not decomposed.claims:
        print("[ClaimRetriever] No decomposed claims — cannot retrieve")
        return {
            "chunks":        [],
            "grader_output": GraderOutput(
                chunk_grades       = [],
                passed_chunks      = [],
                failed_chunks      = [],
                rerouting_decision = "detect",
                confidence_reason  = "No decomposed claims available",
            ),
            "claim_results": [],
        }
 
    claims = decomposed.claims  # capped at 4 by decomposer
    print(f"[ClaimRetriever] Processing {len(claims)} claims")
 
    # Step 1 - Parallel retrieval
    claim_chunks = _parallel_retrieve(claims)
 
    # Step 2 - Grade per claim, collect results
    all_chunk_grades  : list[ChunkGrade] = []
    all_passed_chunks : list[Document]   = []
    all_failed_chunks : list[Document]   = []
    claim_routing     : dict[int, str]   = {}
    claim_results     : list[dict]       = []


    for claim in claims:
        chunks = claim_chunks.get(claim.claim_id, [])
        grades, passed, failed, routing = _grade_claim_chunks(claim, chunks)
 
        all_chunk_grades.extend(grades)
        all_passed_chunks.extend(passed)
        all_failed_chunks.extend(failed)
        claim_routing[claim.claim_id] = routing
 
        claim_results.append({
            "claim_id":   claim.claim_id,
            "claim_text": claim.claim_text,
            "legal_area": claim.legal_area,
            "routing":    routing,
            "passed":     len(passed),
            "total":      len(chunks),
        })


    # Step 3 - Aggregate routing
    aggregate_decision, aggregate_reason = _aggregate_routing(claim_routing)
    graph_decision = "proceed" if aggregate_decision in ("proceed", "partial") else "detect"
 
    print(f"\n[ClaimRetriever] {aggregate_decision} - {aggregate_reason}")
    print(f"[ClaimRetriever] Total passed chunks: {len(all_passed_chunks)}")

    return {
        "chunks": all_passed_chunks,
        "grader_output": GraderOutput(
            chunk_grades       = all_chunk_grades,
            passed_chunks      = all_passed_chunks,
            failed_chunks      = all_failed_chunks,
            rerouting_decision = graph_decision,
            confidence_reason  = aggregate_reason,
        ),
        "claim_results": claim_results,
    }
