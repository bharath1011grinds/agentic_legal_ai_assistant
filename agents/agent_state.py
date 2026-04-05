from typing import Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from .models import GraderOutput, QueryAnalysis, SituationClass, AmbiguityCheck
 
 
class GraphState(TypedDict):
 
    # ── Input ────────────────────────────────────────────────────────────────
    raw_query:              str
 
    # ── AmbiguityDetector writes these ───────────────────────────────────────
    ambiguity:              AmbiguityCheck | None
    clarification_context:  str           # user's answer to clarifying questions, appended to raw_query
 
    # ── SituationClassifier writes these ─────────────────────────────────────
    situation:              SituationClass | None
    document_filter:        list[str]     # ["constitution", "ipc", "crpc"] subset for retrieval
 
    # ── QueryAnalyzer writes these ───────────────────────────────────────────
    analysis:               QueryAnalysis | None
    rewritten_query:        str
 
    # ── Retriever writes this ────────────────────────────────────────────────
    chunks:                 list[Document]
 
    # ── RelevanceGrader writes these ─────────────────────────────────────────
    retry_count:            int
    grader_output:          GraderOutput | None
 
    # ── ScopeGuard writes this ───────────────────────────────────────────────
    out_of_scope:           bool
 
    # ── TavilyFallback writes this ───────────────────────────────────────────
    web_results:            list[Document] | None
 
    # ── Synthesizer writes these ─────────────────────────────────────────────
    answer:                 str
    citations:              list[dict]    # {type, section_number, title, document_type, url}
 
    # ── Conversation history (last 3 pairs, managed by server) ───────────────
    history:                list[dict]    # [{"role": "user"|"assistant", "content": str}]

    clarification_questions : list[str]
 
 
def initial_state(
    raw_query:             str,
    clarification_context: str        = "",
    history:               list[dict] = None,
) -> GraphState:
    return GraphState(
        raw_query             = raw_query,
        clarification_context = clarification_context,
        ambiguity             = None,
        situation             = None,
        document_filter       = [],
        analysis              = None,
        rewritten_query       = raw_query,
        chunks                = [],
        retry_count           = 0,
        grader_output         = None,
        out_of_scope          = False,
        web_results           = None,
        answer                = "",
        citations             = [],
        history               = history or [],
        clarification_questions = []


    )