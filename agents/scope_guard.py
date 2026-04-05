"""LangGraph node: handles queries outside the KB scope gracefully.
Generates a honest, helpful response that redirects without hallucinating law.
 
Reads : state["raw_query"], state["situation"]
Writes: state["answer"], state["citations"]
 
This node is terminal — it writes the final answer and routes to END.
"""
 
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
 
from .agent_state import GraphState
 
load_dotenv()
 
# Situation-specific redirect resources
_REDIRECTS = {
    "employment": "the Labour Commissioner's office or a labour court. You can also visit https://labour.gov.in",
    "family":     "a family court or legal aid centre. Visit https://nalsa.gov.in for free legal aid.",
    "property":   "a civil court or revenue court. Visit https://nalsa.gov.in for free legal aid.",
    "default":    "a qualified lawyer or your nearest District Legal Services Authority (DLSA). Visit https://nalsa.gov.in for free legal aid.",
}
 
_SYSTEM_PROMPT = """You are KnowYourRights, an Indian legal assistant with a limited knowledge base 
covering the Indian Constitution (selected articles), IPC, and CrPC.
 
The user's query falls outside your knowledge base. Your job is to:
1. Acknowledge honestly that this topic is outside your current scope
2. Briefly explain what area of law it falls under
3. Direct them to the appropriate resource (provided to you)
4. Keep it under 4 sentences. No legal advice. No fabrication."""
 
 
def scope_guard_node(state: GraphState) -> dict:
 
    query     = state.get("raw_query", "")
    situation = state.get("situation")
    sit_type  = situation.situation if situation else "default"
 
    redirect = _REDIRECTS.get(sit_type, _REDIRECTS["default"])
 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=200,
        api_key=os.environ.get("GROQ_API_KEY_2"),
    )
 
    human_prompt = (
        f"User query: {query}\n\n"
        f"Redirect resource: {redirect}\n\n"
        f"Write a short, honest out-of-scope response."
    )
 
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]
 
    print(f"\n[ScopeGuard] Query out of scope: '{query[:80]}'")
    print(f"[ScopeGuard] Situation: {sit_type} → redirecting to: {redirect}")
 
    response = llm.invoke(messages)
    answer   = response.content.strip()
 
    print(f"[ScopeGuard] Response generated ({len(answer)} chars)")
 
    return {
        "answer":    answer,
        "citations": [],
    }
 