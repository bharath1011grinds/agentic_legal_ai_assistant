import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
 
from .agent_state import GraphState
 
load_dotenv()
 
_RESOURCES = {
    "constitution": "indiankanoon.org for the full Constitution text, or sci.gov.in for Supreme Court judgments.",
    "ipc":          "indiankanoon.org for IPC sections, or the Ministry of Law at legislative.gov.in",
    "crpc":         "indiankanoon.org for CrPC sections, or your nearest District Legal Services Authority (DLSA).",
    "default":      "indiankanoon.org for case law and legislation, or nalsa.gov.in for free legal aid.",
}
 
_SYSTEM_PROMPT = """You are KnowYourRights, an Indian legal assistant with a focused knowledge base 
covering selected articles of the Constitution, IPC sections, and CrPC sections.
 
The retrieval system could not find sufficiently relevant information for this query.
Your job is to:
1. Acknowledge honestly that this specific query isn't well-covered in your current knowledge base
2. Suggest what type of law or section might be relevant (if you can reasonably infer it)
3. Direct the user to the resource provided
4. Keep it under 4 sentences. Never fabricate specific section numbers or legal claims."""
 
 
def kb_miss_node(state: GraphState) -> dict:
    """
    Reads : state["raw_query"], state["situation"], state["rewritten_query"]
    Writes: state["answer"], state["citations"]
    """
    query     = state.get("raw_query", "")
    situation = state.get("situation")
    doc_filter = state.get("document_filter", [])
 
    # Pick the most specific resource based on document filter
    resource = _RESOURCES["default"]
    for doc_type in doc_filter:
        if doc_type in _RESOURCES:
            resource = _RESOURCES[doc_type]
            break
 
    human_prompt = (
        f"User query: {query}\n\n"
        f"Redirect resource: {resource}\n\n"
        f"Write a short, honest response acknowledging the KB gap."
    )
 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=200,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
 
    print(f"\n[KBMiss] Insufficient chunks for: '{query[:80]}'")
    print(f"[KBMiss] Situation: {situation.situation if situation else 'out_of_scope'}")
    print(f"[KBMiss] Redirecting to: {resource}")
 
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ])
 
    answer = response.content.strip()
    print(f"[KBMiss] Response generated ({len(answer)} chars)")
 
    return {
        "answer":    answer,
        "citations": [],
    }