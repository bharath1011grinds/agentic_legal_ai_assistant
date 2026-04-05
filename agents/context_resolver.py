import os
import json
from dotenv import load_dotenv
 
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
 
from .agent_state import GraphState

load_dotenv()

# 1 turn = 1 user message + 1 assistant message = 2 entries in the list.
# 3 turns is enough context without bloating the prompt.
_MAX_HISTORY_TURNS = 3  # = last 6 entries in conversation_history


RESOLVER_SYSTEM_PROMPT = """You are a query resolver for KnowYourRights.ai, an Indian legal assistant.
 
Your job is to rewrite the user's latest query into a fully self-contained, retrieval-ready query
using the conversation history provided.
 
TWO things you must do:
1. RESOLVE references — replace pronouns and implicit references with their actual legal subjects.
   e.g. "can they do that?" → "can police arrest without warrant under CrPC?"
 
2. ENRICH with legal context — if prior conversation mentioned specific sections, articles, or rights,
   carry the relevant ones forward into the rewritten query so the retrieval system can find
   the right legal chunks without access to prior retrieved documents.
   e.g. if prior answer discussed Section 41 CrPC, and user asks "what are my rights in that case?",
   rewrite as "what are the rights of a person being arrested without warrant under Section 41 CrPC?"
 
Rules:
- If the query is already fully self-contained with no references to prior turns, return it UNCHANGED.
- Do NOT answer the query. Only rewrite it.
- Do NOT add legal information not grounded in the conversation history.
- Keep the rewritten query concise — 1 to 3 sentences max.
- Use proper legal terminology (Section X CrPC, Article X Constitution, IPC Section X).
 
Return ONLY a JSON object, no markdown, no preamble:
{
  "resolved_query": "the rewritten standalone retrieval-ready query"
}"""



def _build_resolver_prompt(raw_query: str, history: list[dict]) -> str:
    """Build the user message showing history + current query."""
    lines = ["Conversation so far:"]
    for turn in history:
        role    = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        # Truncate long assistant answers - we only need enough for reference resolution
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "...[truncated]"
        #append each of the previous prompt with roles as a STRING.
        lines.append(f"{role}: {content}")
    
    
    lines.append(f"\nLatest user query: {raw_query}")
    lines.append("\nRewrite the latest query to be fully self-contained.")
    #This results in a combined string, with each turn in a separate line and the instruction for the query to be re-written too...
    return "\n".join(lines)

def resolve_context_node(state: GraphState) -> dict:
    """
    LangGraph node: resolves conversational references in the raw query.
 
    Reads  : state["raw_query"], state["conversation_history"]
    Writes : state["raw_query"]  (overwrites with resolved version)
 
    Passthrough if conversation_history is empty (first turn).
    """
    raw_query = state["raw_query"]
    history   = state.get("history", [])
 
    # --- Passthrough on first turn ---
    if not history:
        print("[ContextResolver] First turn — passthrough, no resolution needed.")
        return {}   # empty dict = no state changes
 
    # Cap history to last N turns
    capped_history = history[-(2 * _MAX_HISTORY_TURNS):]
 
    print(f"[ContextResolver] Resolving query with {len(capped_history)//2} prior turn(s)...")
    print(f"   Raw query : {raw_query}")
 
    llm = ChatGroq(
        model       = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key     = os.environ.get("GROQ_API_KEY"),
        temperature = 0.0,
        max_tokens  = 256,
    )
 
    messages = [
        SystemMessage(content=RESOLVER_SYSTEM_PROMPT),
        HumanMessage(content=_build_resolver_prompt(raw_query, capped_history)),
    ]
 
    response = llm.invoke(messages)
    raw_text = response.content.strip()
 
    # Strip markdown fences if model ignores instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()
 
    try:
        parsed         = json.loads(raw_text)
        resolved_query = parsed["resolved_query"].strip()
        if not resolved_query:
            raise ValueError("Empty resolved_query in response")
    except Exception as e:
        # Safe fallback — keep original query, pipeline continues normally
        print(f"[ContextResolver] Parse error: {e} — keeping raw query as-is")
        resolved_query = raw_query
 
    print(f"   Resolved  : {resolved_query}")
 
    # Overwrite raw_query with the resolved version.
    # query_analyzer_node reads raw_query next, so it naturally gets the resolved query.
    # rewritten_query is also seeded here as a safe default (analyzer will overwrite it).
    return {
        "raw_query":      resolved_query, #raw query as well as rewritten query gets the resolved version
        "rewritten_query": resolved_query,
    }