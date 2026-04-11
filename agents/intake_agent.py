import os
import json
from dotenv import load_dotenv
 
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
 
from .agent_state import GraphState
from .models import CaseLog, CaseParties, LegalClaim, DecomposedClaims
 
load_dotenv()
 
MAX_TURNS = 12
 
# Field Extractor Prompt 
 
_EXTRACTOR_SYSTEM_PROMPT = """You are a legal case intake assistant for KnowYourRights.ai, 
an Indian legal assistant covering the Constitution, IPC, and CrPC.
 
Your job is to extract structured information from the user's message and update the case log.
 
Extract ONLY what the user has explicitly stated. Do NOT infer or assume.
If a field cannot be confidently extracted from this message, leave it as null.
 
For incident_type choose from:
  arrest | employment | tenancy | domestic | consumer | property | criminal_accusation | civil_rights | other
 
For parties.user_role choose from:
  accused | employee | tenant | victim | petitioner | other
 
For parties.opposing_party choose from:
  police | employer | landlord | family_member | government | other
 
For relief_sought choose from if needed:
  know_my_rights | file_complaint | get_bail | draft_document | understand_charges | other
 
Respond ONLY in this JSON format, no markdown, no preamble:
{
  "incident_type": "<value or null>",
  "what_happened": "<plain language summary or null>",
  "parties": {
    "user_role": "<value or null>",
    "opposing_party": "<value or null>"
  },
  "relief_sought": "<value or null>",
  "timeline": "<value or null>",
  "jurisdiction": "<value or null>",
  "evidence_available": ["<item>", ...],
  "prior_legal_action": "<value or null>",
  "specific_sections_mentioned": ["<section>", ...]
}"""
 
 
# Conversational Agent Prompt
 
_AGENT_SYSTEM_PROMPT = """You are a warm, professional legal intake assistant for KnowYourRights.ai,
an Indian legal assistant covering the Constitution, IPC, and CrPC.
 
Your goal is to gather enough information about the user's legal situation
so that a precise legal answer can be retrieved for them.
 
RULES:
- Ask ONE question at a time. Never ask multiple questions in one message.
- Be to the point but empathetic — the user may be in a stressful situation.
- Ask the most important missing field first (follow priority order below).
- If the user has already answered a question in their message, do NOT ask it again.
- Keep your messages short — 2-3 sentences maximum.
- Never give legal advice during intake. Just gather information.
- Never mention "case log", "fields", or "database" to the user.
 
Priority order for missing fields:
1. what_happened (only if completely unclear)
2. incident_type (only if not determinable from what_happened)
3. parties.opposing_party (who is this against?)
4. timeline (when did this happen?)
5. jurisdiction (where did this happen?)
6. evidence_available (do they have any documents?)
7. prior_legal_action (has anything been filed?)
8. relief_sought (what do they want from this?)
 
ONLY when all the fields are filled OR turn limit is reached:
End your message with exactly this phrase on a new line:
[INTAKE_READY]
 
Example ending message:
"Thank you, I have enough information to look into your situation. 
Click 'Get Legal Advice' whenever you're ready to proceed.
[INTAKE_READY]"
"""
 
 
# Decomposer Prompt
 
_DECOMPOSER_SYSTEM_PROMPT = """You are a legal claim decomposer for an Indian legal assistant.
 
Given a structured case log, decompose the situation into legally DISTINCT claims
that require retrieval from different parts of the law (Constitution, IPC, CrPC).
 
RULES:
- Maximum 6 claims.
- Merge similar claims that would retrieve from the same legal sections.
- Each claim must be retrieval-ready — written as a legal question or statement
  using proper Indian legal terminology.
- Assign the correct document types to retrieve for each claim.
- Claims must be genuinely legally distinct — different sections, different laws.
 
Document type options: constitution, ipc, crpc
 
Respond ONLY in this JSON format, no markdown, no preamble:
{
  "claims": [
    {
      "claim_id": 0,
      "claim_text": "retrieval-ready legal claim text",
      "legal_area": "e.g. CrPC arrest procedure",
      "doc_filter": ["crpc", "constitution"]
    }
  ]
}"""
 


def _build_history_messages(history : list[dict]):

#convert last 8 history entries into langchain message object...
    messages = []
    for turn in history[-8:]:
        role    = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _merge_extracted_into_log(existing : CaseLog, extracted : dict):

    log_dict = existing.model_dump()

    #Scalar Fields in the log
    scalar_fields = [
        "incident_type", "what_happened", "relief_sought",
        "timeline", "jurisdiction", "prior_legal_action",
    ]

    for field in scalar_fields:
        if log_dict.get(field) is None and extracted.get(field) is not None:
            log_dict[field] = extracted[field]
    
    #list fields in the log
    list_fields = ["evidence_available", "specific_sections_mentioned"]

    for field in list_fields:
        existing_items = log_dict.get(field, []) or []
        new_items = extracted.get(field, []) or []
        print(existing_items, new_items)
        #convert the merged list to a temporary dict to get rid of the duplicates and convert it to a list again.
        #Note: Order remains the same as it was inserted in while creating empty dicts.
        merged = list(dict.fromkeys(existing_items+new_items)) #merged and de-duped
        print(existing_items, new_items)
        #write the new list to the log_dict
        log_dict[field] = merged

    
    #process the nested parties fields

    existing_parties = log_dict.get('parties') or {}
    extracted_parties = extracted.get('parties') or {}

    #Adding this here, just in case model_dump does not recursively convert objs to dict...
    if isinstance(existing_parties, CaseParties):
        existing_parties = existing_parties.model_dump()
    
    #create a copy and pass all the values into the new dict
    merged_parties = {**existing_parties}
    for k, v in extracted_parties.items():
        if merged_parties.get(k) is None and v is not None:
            merged_parties[k] = v
    
    #only if any there is any field present for the CaseParties, create one...
    if any(v is not None for v in merged_parties.values()):

        #add the parties to the log_dict if they exist
        try:
            log_dict['parties'] = CaseParties(**merged_parties)
        except Exception:
            log_dict["parties"] = None
        
    else: 
        log_dict['parties'] = None
    

    try:
        return CaseLog(**log_dict)
    except Exception as e:
        print(f"[IntakeAgent] CaseLog merge failed: {e} — keeping existing log")
        return existing
    

def _extract_fields(raw_query: str, llm: ChatGroq) -> dict:

    #Calls the extractor LLM to pull structured fields from the user's message.
    #Returns a dict matching the CaseLog schema (values may be None).
   
    messages = [
        SystemMessage(content=_EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"User message: {raw_query}"),
    ]
 
    try:
        response = llm.invoke(messages)
        raw_text = response.content.strip()
 
        if raw_text.startswith("```"):
            parts    = raw_text.split("```")
            raw_text = parts[1]
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()
 
        return json.loads(raw_text)
 
    except Exception as e:
        print(f"[IntakeAgent] Field extraction failed: {e} — returning empty extraction")
        return {}
    
def _build_next_question(
    case_log:   CaseLog,
    history:    list[dict],
    turn_count: int,
    llm:        ChatGroq,
) -> str:
    """
    Calls the conversational agent LLM to generate the next question
    or the soft termination message.
    """
    log_summary = case_log.to_context_string()
    turns_left  = MAX_TURNS - turn_count
 
    user_prompt = (
        f"Current case log:\n{log_summary}\n\n"
        f"Turns remaining: {turns_left}\n\n"
        #f"Non-negotiables filled: {case_log.non_negotiables_filled()}\n\n"
        f"Generate the next intake message. "
        f"If all fields are filled or turns_remaining <= 0, end with [INTAKE_READY]."
    )
 
    messages = (
        [SystemMessage(content=_AGENT_SYSTEM_PROMPT)]
        + _build_history_messages(history)
        + [HumanMessage(content=user_prompt)]
    )
 
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"[IntakeAgent] Question generation failed: {e}")
        return (
            "Thank you for sharing your situation. "
            "Click 'Get Legal Advice' whenever you're ready to proceed.\n[INTAKE_READY]"
        )
    

def intake_agent_node(state: GraphState) -> dict:
    """
    LangGraph node: runs the conversational intake agent for one turn.
 
    Reads : state["raw_query"], state["case_log"], state["intake_turn_count"], state["history"]
    Writes: state["case_log"], state["intake_turn_count"], state["answer"]
    """
    raw_query   = state.get("raw_query", "")
    case_log    = state.get("case_log") or CaseLog()
    turn_count  = state.get("intake_turn_count", 0)
    history     = state.get("history") or []
 
    print(f"\n[IntakeAgent] Turn {turn_count + 1}/{MAX_TURNS} | Query: '{raw_query[:60]}'")
    print(f"[IntakeAgent] Non-negotiables filled: {case_log.non_negotiables_filled()}")
 
    llm = ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0.2,          # slight warmth for conversational tone
        max_tokens  = 300,
        api_key     = os.environ.get("GROQ_API_KEY"),
    )
    
    #extract info from the user message and merge it to the existing log...
    extracted = _extract_fields(raw_query=raw_query, llm=llm)
    updated_log =_merge_extracted_into_log(case_log, extracted)

    print(f"[IntakeAgent] Log after extraction - non-negotiables: {updated_log.non_negotiables_filled()}")
 
    #Increment turn count
    new_turn_count = turn_count + 1

    #Build the next agent question or termination message...
    agent_response = _build_next_question(case_log=case_log, history=history, turn_count=new_turn_count, llm=llm)


    #If the agent has all the info...[the intake ready is hardcoded in the prompt template..]
    intake_ready = "[INTAKE_READY]" in agent_response
    clean_response = agent_response.replace("[INTAKE_READY]", "").strip()

    if intake_ready:
        print(f"[IntakeAgent] Intake ready — non-negotiables filled or turn limit reached")
    else:
        print(f"[IntakeAgent] Continuing intake...")

 
    return {
        "case_log":          updated_log,
        "intake_turn_count": new_turn_count,
        "answer":            clean_response,
        # Signal to the server that the UI should show the "Get Legal Advice" button
        # We reuse the clarification_questions field as a lightweight flag channel
        # so the server can detect INTAKE_READY without a new state field.
        "clarification_questions": ["__INTAKE_READY__"] if intake_ready else [],
    }

def decompose_claims_node(state: GraphState):

    case_log = state.get("case_log")
 
    if case_log is None:
        print("[Decomposer] No case log found — cannot decompose")
        return {"decomposed_claims": None}
 
    print(f"\n[Decomposer] Decomposing case log into legal claims...")
    print(f"[Decomposer] Incident: {case_log.incident_type} | Relief: {case_log.relief_sought}")

    llm = ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0.0,
        max_tokens  = 600,
        api_key     = os.environ.get("GROQ_API_KEY"),
    )

    messages = [SystemMessage(content= _DECOMPOSER_SYSTEM_PROMPT), 
                HumanMessage(content=f"Case log:\n{case_log.to_context_string()}")]
    

    try:
        response = llm.invoke(messages)
        raw_text = response.content.strip()
 
        if raw_text.startswith("```"):
            parts    = raw_text.split("```")
            raw_text = parts[1]
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
        raw_text = raw_text.strip()
 
        parsed           = json.loads(raw_text)
        #converting the claims into a DecomposedClaims object
        #pydantic allows type coercion, the claims generated by the llm automatically gets assumed as LegalClaim(by pydantic) type
        #and the list of claims get converted into DecomposedClaims...
        decomposed       = DecomposedClaims(**parsed)
        
        print(f"[Decomposer] {len(decomposed.claims)} claims decomposed:")
        for claim in decomposed.claims:
            print(f"  [{claim.claim_id}] {claim.claim_text[:80]} | {claim.doc_filter}")
 
        return {"decomposed_claims": decomposed}
 
    except Exception as e:
        print(f"[Decomposer] Failed: {e} — falling back to single claim from case log")
 
        # Safe fallback — treat the whole case log as one claim
        fallback_claim = LegalClaim(
            claim_id   = 0,
            claim_text = case_log.what_happened or case_log.to_context_string(),
            legal_area = case_log.incident_type or "general",
            doc_filter = ["constitution", "ipc", "crpc"],
        )
        return {
            "decomposed_claims": DecomposedClaims(claims=[fallback_claim])
        }