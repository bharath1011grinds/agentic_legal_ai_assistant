import os
from typing import Generator
from dotenv import load_dotenv
 
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
 
from .agent_state import GraphState
from .models import CaseLog
from .synthesizer import _parse_citations
 
load_dotenv()

def _build_intake_system_prompt(
    case_log:      CaseLog,
    claim_results: list[dict],
    used_tavily:   bool,
) -> str:
 
    missed_claims = [
        r["claim_text"] for r in claim_results
        if r.get("routing") == "detect"
    ]
 
    missed_note = ""
    if missed_claims:
        missed_str  = "\n".join(f"  - {c}" for c in missed_claims)
        missed_note = (
            f"\nNOTE: The knowledge base had insufficient information for:\n"
            f"{missed_str}\n"
            f"For these issues only, tell the user we don't have enough coverage "
            f"and suggest indiankanoon.org or nalsa.gov.in."
        )
 
    #Sets the tone - drives the LLM in a particular direction
    relief_notes = {
        "know_my_rights":     "Focus on explaining rights clearly in plain language.",
        "file_complaint":     "Explain the complaint filing procedure step by step.",
        "get_bail":           "Prioritise bail provisions and procedure under CrPC.",
        "draft_document":     "Outline what the document should contain based on the legal situation.",
        "understand_charges": "Explain the charges and their legal implications under IPC.",
        "other":              "",
    }
    relief_note = relief_notes.get(case_log.relief_sought or "other", "")
 
    return f"""You are KnowYourRights, a legal assistant helping Indian citizens understand their rights
under the Constitution, IPC, and CrPC.
 
The user has completed a case intake. Their full situation is in the case log.
You are given claim-wise evidence — each legal claim has its own relevant sources.
 
RULES:
- Answer using ONLY the sources provided under each claim. Never fabricate law.
- Always cite the specific Section or Article number when making a legal claim.
- Write in plain English a non-lawyer can understand.
- Structure your answer by legal issue — one section per claim.
- Use a bold header for each claim e.g. **1. Arrest Without Warrant**
- End with a "What this means for you:" paragraph summarising practical next steps.
- If sources are insufficient for a claim, say so clearly rather than guessing.
{relief_note}{missed_note}
 
Citation format inline: "Under Section 41 CrPC..." or "Article 19(1)(a) guarantees..."
 
After your answer, output a CITATIONS block in this exact format:
---CITATIONS---
[CONSTITUTION] Article <number> | <title> | <url>
[IPC]          Section <number> | <title> | <url>
[CRPC]         Section <number> | <title> | <url>
[WEB]          <title> | <url>
---END---"""
 


def _format_chunk(doc: Document) -> str:
    """Formats a single chunk as a labelled evidence entry."""
    doc_type = doc.metadata.get("document_type", "")
    ref      = doc.metadata.get("section_number") or doc.metadata.get("article_number") or ""
    title    = doc.metadata.get("title", "")
    score    = doc.metadata.get("grader_score", "")
    is_web   = doc.metadata.get("source") == "tavily_web"
 
    if is_web:
        label = "[WEB]"
    elif doc_type == "constitution":
        label = "[CONSTITUTION]"
    elif doc_type == "ipc":
        label = "[IPC]"
    elif doc_type == "crpc":
        label = "[CRPC]"
    else:
        label = "[SOURCE]"
 
    header = label
    if ref:
        header += f" Section/Article {ref}"
    if title:
        header += f" — {title}"
    if score:
        header += f" (relevance: {score})"
 
    return f"  - {header}\n    {doc.page_content[:400].strip()}"


def _build_human_prompt(
    case_log:      CaseLog,
    chunks:        list[Document],
    claim_results: list[dict],
) -> str:
    
    lines = ["Claim-wise Evidence:"]
 
    for r in claim_results:
        claim_id   = r["claim_id"]
        claim_text = r["claim_text"]
        routing    = r["routing"]
        status     = "covered" if routing == "proceed" else "insufficient KB coverage"
 
        lines.append(f"\n[Claim {claim_id + 1}] {claim_text} ({status})")
        lines.append("Relevant sources:")
 
        claim_chunks = [
            doc for doc in chunks
            if doc.metadata.get("graded_for_claim") == claim_id
        ]
 
        if not claim_chunks:
            lines.append("  No relevant sources found.")
        else:
            for doc in claim_chunks:
                lines.append(_format_chunk(doc))
 
    return f"{case_log.to_context_string()}\n\n" + "\n".join(lines)


def intake_synthesize_node(state: GraphState) -> dict:
    chunks        = state.get("chunks", [])
    case_log      = state.get("case_log") or CaseLog()
    claim_results = state.get("claim_results") or []
    web_results   = state.get("web_results") or []
 
    if not chunks and not claim_results:
        return {
            "answer":    "I could not find relevant information in the knowledge base for your situation.",
            "citations": [],
        }
 
    used_tavily   = len(web_results) > 0
    system_prompt = _build_intake_system_prompt(case_log, claim_results, used_tavily)
    human_prompt  = _build_human_prompt(case_log, chunks, claim_results)
 
    llm = ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0,
        max_tokens  = 2048,
        api_key     = os.environ.get("GROQ_API_KEY"),
    )
 
    print(f"\n[IntakeSynthesizer] {len(claim_results)} claims | {len(chunks)} chunks")
 
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
 
    full_response = ""
    for chunk in llm.stream(messages):
        token = chunk.content
        print(token, flush=True, end="")
        full_response += token
 
    print()
 
    clean_answer, citations = _parse_citations(full_response)
    print(f"[IntakeSynthesizer] Done: {len(clean_answer)} chars | {len(citations)} citations")
 
    return {
        "answer":    clean_answer,
        "citations": citations,
    }
 


#FASTAPI streaming end-point:
def stream_intake_answer(state: GraphState) -> Generator[str, None, None]:
    chunks        = state.get("chunks", [])
    case_log      = state.get("case_log") or CaseLog()
    claim_results = state.get("claim_results") or []
    web_results   = state.get("web_results") or []
 
    used_tavily   = len(web_results) > 0
    system_prompt = _build_intake_system_prompt(case_log, claim_results, used_tavily)
    human_prompt  = _build_human_prompt(case_log, chunks, claim_results)
 
    llm = ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0,
        max_tokens  = 2048,
        streaming   = True,
        api_key     = os.environ.get("GROQ_API_KEY"),
    )
 
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]
 
    for chunk in llm.stream(messages):
        token = chunk.content
        if token:
            yield f"data:{token}\n\n"