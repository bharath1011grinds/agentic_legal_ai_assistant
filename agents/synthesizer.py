import os
from typing import Generator
from dotenv import load_dotenv
 
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
 
from .agent_state import GraphState
from .models import SituationClass
 
load_dotenv()
 
 
# ── System Prompt ─────────────────────────────────────────────────────────────
 
def build_system_prompt(situation: SituationClass | None, used_tavily: bool) -> str:
 
    situation_note = ""
    if situation:
        sit = situation.situation
        if sit == "arrest_detention":
            situation_note = "\nThe user may be in or facing an arrest/detention situation. Prioritise their immediate rights clearly."
        elif sit == "criminal_offence":
            situation_note = "\nThe user is involved in a criminal matter. Explain the relevant IPC provisions and procedures clearly."
        elif sit == "civil_rights":
            situation_note = "\nThe user is asking about fundamental rights. Cite the specific Article and sub-clause."
 
    tavily_note = (
        "\nNOTE: The knowledge base had insufficient information. "
        "Some sources below are from a live web search (marked [WEB]). Mention this to the user."
    ) if used_tavily else ""
 
    return f"""You are KnowYourRights, a legal assistant helping Indian citizens understand their rights under the Constitution, IPC, and CrPC.
 
RULES:
- Answer using ONLY the sources provided. Never fabricate law.
- Always cite the specific Section or Article number when making a legal claim.
- Write in plain English a non-lawyer can understand.
- If sources are insufficient, say so clearly rather than guessing.
- End every answer with a "What this means for you:" paragraph — 2-3 sentences summarising the practical implication.
- If sources conflict or a referenced section is not in the KB, flag it explicitly.
- You have access to the last 3 conversation turns. Use them for context but prioritise the current query and sources.
{situation_note}{tavily_note}
 
Citation format inline: "Under Section 41 CrPC..." or "Article 19(1)(a) guarantees..."
 
After your answer, output a CITATIONS block in this exact format:
---CITATIONS---
[CONSTITUTION] Article <number> | <title> | <url>
[IPC]          Section <number> | <title> | <url>
[CRPC]         Section <number> | <title> | <url>
[WEB]          <title> | <url>
---END---"""
 
 
# ── Context Builder ───────────────────────────────────────────────────────────
 
def _build_context(chunks: list[Document]) -> str:
    lines = []
    for i, doc in enumerate(chunks):
        doc_type   = doc.metadata.get("document_type", "")
        is_web     = doc.metadata.get("source") == "tavily_web"
        is_summary = doc.metadata.get("chunk_type") == "plain_summary"
 
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
 
        ref   = doc.metadata.get("section_number") or doc.metadata.get("article_number") or ""
        title = doc.metadata.get("title", "")
        url   = doc.metadata.get("url", "")
        score = doc.metadata.get("grader_score", "")
 
        header = f"{label} [{i+1}]"
        if ref:
            header += f" Section/Article {ref}"
        if title:
            header += f" — {title}"
        if url:
            header += f" | {url}"
        if score:
            header += f" | relevance: {score}"
        if is_summary:
            header += " | [PLAIN LANGUAGE SUMMARY]"
 
        lines.append(f"\n{header}\n{doc.page_content[:600].strip()}")
 
    return "\n".join(lines)
 
 
# ── History Builder ───────────────────────────────────────────────────────────
 
def _build_history_messages(history: list[dict]) -> list:
    """Convert last 3 pairs (6 messages) of history to LangChain message objects."""
    messages = []
    for turn in history[-6:]:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages
 
 
# ── Citation Parser ───────────────────────────────────────────────────────────
 
def _parse_citations(answer_text: str) -> tuple[str, list[dict]]:
    if "---CITATIONS---" in answer_text:
        parts          = answer_text.split("---CITATIONS---")
        clean_answer   = parts[0].strip()
        citation_block = parts[1].replace("---END---", "").strip()
    else:
        clean_answer   = answer_text.strip()
        citation_block = ""
 
    citations = []
    for line in citation_block.splitlines():
        line = line.strip()
        if not line:
            continue
 
        if line.startswith("[WEB]"):
            parts = line.lstrip("[WEB]").strip().split("|")
            parts = [p.strip() for p in parts]
            citations.append({
                "type":           "web",
                "title":          parts[0] if len(parts) > 0 else "",
                "url":            parts[1] if len(parts) > 1 else "",
                "section_number": "",
                "document_type":  "web",
            })
        else:
            if line.startswith("[CONSTITUTION]"):
                doc_type = "constitution"
                line = line.lstrip("[CONSTITUTION]").strip()
            elif line.startswith("[IPC]"):
                doc_type = "ipc"
                line = line.lstrip("[IPC]").strip()
            elif line.startswith("[CRPC]"):
                doc_type = "crpc"
                line = line.lstrip("[CRPC]").strip()
            else:
                doc_type = "unknown"
 
            parts = [p.strip() for p in line.split("|")]
            citations.append({
                "type":           "legal",
                "document_type":  doc_type,
                "section_number": parts[0] if len(parts) > 0 else "",
                "title":          parts[1] if len(parts) > 1 else "",
                "url":            parts[2] if len(parts) > 2 else "",
            })
 
    return clean_answer, citations
 
 
# ── LangGraph Node ────────────────────────────────────────────────────────────
 
def synthesize_node(state: GraphState) -> dict:
    chunks      = state.get("chunks", [])
    situation   = state.get("situation")
    web_results = state.get("web_results") or []
    query       = state.get("rewritten_query") or state.get("raw_query", "")
    history     = state.get("history") or []
 
    if not chunks:
        return {
            "answer":    "I could not find relevant information in the knowledge base to answer your question.",
            "citations": [],
        }
 
    used_tavily   = len(web_results) > 0
    system_prompt = build_system_prompt(situation=situation, used_tavily=used_tavily)
    context       = _build_context(chunks)
    human_prompt  = f"Query: {query}\n\nSources:\n{context}"
 
    messages = (
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        streaming=True,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
 
    print(f"\n[Synthesizer] Generating answer...")
    print(f"  Query     : {query[:80]}")
    print(f"  Chunks    : {len(chunks)} ({len(web_results)} from Tavily)")
    print(f"  Situation : {situation.situation if situation else 'unknown'}")
    print(f"  History   : {len(history)} turns")
    print(f"\n{'='*60}\n")
 
    full_response = ""
    for chunk in llm.stream(messages):
        token = chunk.content
        print(token, flush=True, end="")
        full_response += token
 
    print(f"\n{'='*60}")
 
    clean_answer, citations = _parse_citations(full_response)
    print(f"\n[Synthesizer] Done: {len(clean_answer)} chars | {len(citations)} citations")
 
    return {
        "answer":    clean_answer,
        "citations": citations,
    }
 
 
# ── FastAPI Streaming Entry Point ─────────────────────────────────────────────
 
def stream_answer(state: GraphState) -> Generator[str, None, None]:
    """
    Yields SSE tokens for FastAPI StreamingResponse.
    Called by server.py after all upstream nodes have run.
    """
    chunks      = state.get("chunks", [])
    situation   = state.get("situation")
    web_results = state.get("web_results") or []
    query       = state.get("rewritten_query") or state.get("raw_query", "")
    history     = state.get("history") or []
 
    used_tavily   = len(web_results) > 0
    system_prompt = build_system_prompt(situation=situation, used_tavily=used_tavily)
    context       = _build_context(chunks)
    human_prompt  = f"Query: {query}\n\nSources:\n{context}"
 
    messages = (
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        streaming=True,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
 
    for chunk in llm.stream(messages):
        token = chunk.content
        if token:
            yield f"data:{token}\n\n"
 