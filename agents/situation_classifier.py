import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
 
from .agent_state import GraphState
from .models import SituationClass
 
load_dotenv()

_SYSTEM_PROMPT = """You are a legal situation classifier for an Indian legal assistant.
 
Classify the user's query into exactly one situation and identify which legal documents are relevant.
 
Situations:
- arrest_detention   : being arrested, stopped, detained, or questioned by police
- criminal_offence   : accused of a crime OR victim of a crime under IPC
- civil_rights       : fundamental rights under the Constitution (speech, equality, liberty, etc.)
- property           : disputes over moveable or immoveable property
- employment         : workplace rights, wrongful termination, labour issues
- family             : divorce, custody, inheritance, domestic violence
- out_of_scope       : topic outside Constitution/IPC/CrPC (tax, corporate, IP, consumer etc.)
 
Document filter rules:
- arrest_detention   -> ["crpc", "constitution"]
- criminal_offence   -> ["ipc", "crpc"]
- civil_rights       -> ["constitution"]
- property           -> ["ipc"]
- employment         -> ["ipc"]
- family             -> ["ipc"]
- out_of_scope       -> []
 
Respond ONLY in this JSON format, no other text:
{
  "situation": "<one of the 7 values above>",
  "document_filter": ["<doc_type>", ...],
  "confidence": <float 0-1>,
}"""


def classify_situation_node(state: GraphState) -> dict:
    import json
    query = state.get("raw_query", "")
    clarification = state.get("clarification_context", "")
    full_query = f"{query}\n{clarification}".strip() if clarification else query
 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=200,
        api_key=os.environ.get("GROQ_API_KEY_2"),
    )
 
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {full_query}"),
    ]
 
    print(f"\n[SituationClassifier] Classifying: '{query[:80]}'")

 
    response = llm.invoke(messages)
    raw_text = response.content.strip()
    if raw_text.startswith("```"):
        # Split by the fences and take the content inside
        parts = raw_text.split("```")
        if len(parts) >= 2:
            raw_text = parts[1]
        
        # Remove 'json' prefix if present
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]

    raw_text = raw_text.strip()
 
    try:
        parsed = json.loads(raw_text)
        situation = SituationClass(**parsed)

    except Exception as e:
        print(f"[SituationClassifier] Parse failed: {e} — defaulting to civil_rights")
        situation = SituationClass(
            situation="civil_rights",
            document_filter=["constitution", "ipc", "crpc"],
            confidence=0.0,
        )



    out_of_scope = situation.situation == "out_of_scope"
 
    print(f"[SituationClassifier] → {situation.situation} (conf: {situation.confidence:.2f})")
    print(f"[SituationClassifier] → Filter: {situation.document_filter}")
    if out_of_scope:
        print(f"[SituationClassifier] → Out of scope — will trigger ScopeGuard")
 
    return {
        "situation":       situation,
        "document_filter": situation.document_filter,
        "out_of_scope":    out_of_scope,
    }