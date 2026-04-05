import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
 
from .agent_state import GraphState
from .models import AmbiguityCheck
 
load_dotenv()

_SYSTEM_PROMPT = """You are an ambiguity detector for an Indian legal assistant called KnowYourRights.
 
Your job is to decide if a query has enough legal context to retrieve accurate information.
Apply different rules depending on whether the query is a STATEMENT or a QUESTION.
 
━━━ STATEMENTS (user describing their situation) ━━━
A statement MUST name a clear legal situation to be unambiguous.
Ask yourself: do I know WHAT happened AND roughly which law applies?
If either is missing, mark ambiguous.
 
  Ambiguous statements (situation too vague):
  - "I got arrested"           → arrested for what? bail? rights during custody? police procedure?
  - "I have a problem"         → no situation at all
  - "Something happened to me" → no situation at all
  - "I need legal help"        → no situation at all
 
  Unambiguous statements (situation is clear):
  - "Police arrested me without a warrant"  → arrest, CrPC
  - "My landlord won't return my deposit"   → tenancy law
  - "My boss fired me without notice"       → employment law
  - "Someone hit my car"                    → motor accident / tort
 
━━━ QUESTIONS (user asking about a legal topic) ━━━
A question MUST have a clear subject to be unambiguous.
Ask yourself: do I know WHAT legal topic or situation they are asking about?
 
  Ambiguous questions (no clear subject):
  - "What are my rights?"         → rights in which situation?
  - "What can they do to me?"     → who is 'they'? what situation?
  - "Is this legal?"              → is what legal?
  - "What should I do?"           → about what?
 
  Unambiguous questions (subject is clear):
  - "Can police arrest me without a warrant?"  → arrest powers, CrPC
  - "What is bail under section 436?"          → specific law cited
  - "Can my employer cut my salary?"           → employment law
  - "What is the punishment for theft?"        → IPC, clear topic
 
━━━ ALWAYS unambiguous — never flag these ━━━
- Query mentions a specific law, section, or article (e.g. "Section 498A", "Article 21")
- Query is a short clarification answer following a prior assistant question
  (e.g. "Police stopped me", "I was fired", "Landlord issue") — these are context answers, not standalone queries
 
If ambiguous, generate exactly 1 SHORT clarifying question with 2–4 option labels.
Format: "Question text|||Option1|Option2|Option3"
 
Respond ONLY in this JSON format, no markdown, no preamble:
{
  "is_ambiguous": <true|false>,
  "clarifying_questions": ["Question text|||Opt1|Opt2|Opt3"],
  "reason": "<one line>"
}
 
Examples:
Query: "I got arrested"
→ is_ambiguous: true
→ clarifying_questions: ["What do you need help with?|||Know my rights during custody|Apply for bail|Understand the charges|Police didn't follow procedure"]
 
Query: "what are my rights"
→ is_ambiguous: true
→ clarifying_questions: ["What is your situation?|||Police stopped me|I was fired from my job|Landlord is harassing me|Someone stole from me"]
 
Query: "can cops arrest me without proof"
→ is_ambiguous: false  (clear question, subject is arrest powers)
 
Query: "my employer didn't pay me this month"
→ is_ambiguous: false  (clear statement, employment/wages situation named)
 
Query: "section 498a"
→ is_ambiguous: false  (specific law cited)
 
Query: "Police stopped me"
→ is_ambiguous: false  (clarification answer, treat as unambiguous)
"""

def detect_ambiguity_node(state: GraphState) -> dict:
 
    query = state.get("raw_query", "")
 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=300,
        api_key=os.environ.get("GROQ_API_KEY_2"),
    )
 
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"User query: {query}"),
    ]
 
    print(f"\n[AmbiguityDetector] Checking: '{query[:80]}'")
 
    try:
        response = llm.invoke(messages)
        parser   = JsonOutputParser(pydantic_object=AmbiguityCheck)
        parsed   = parser.parse(response.content)
        ambiguity = AmbiguityCheck(**parsed)

        #storing the clarification questions in the state
        questions = ambiguity.clarifying_questions
        print(parsed)
    except Exception as e:
        print(f"[AmbiguityDetector] Parse failed: {e} - assuming not ambiguous")
        ambiguity = AmbiguityCheck(
            is_ambiguous=False,
            questions=[],
            reason="Parse error - defaulting to not ambiguous",
        )
 
    print(f"[AmbiguityDetector] → ambiguous={ambiguity.is_ambiguous} | {ambiguity.reason}")
 
    return {"ambiguity": ambiguity, "clarification_questions" : questions}


