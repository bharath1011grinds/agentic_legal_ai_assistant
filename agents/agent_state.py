from typing import Literal
from typing_extensions import TypedDict
from langchain_core.documents import Document
from .models import GraderOutput, QueryAnalysis, SituationClass, AmbiguityCheck, CaseLog, DecomposedClaims
 
 
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

    # ── Intake Agent writes these ─────────────────────────────────────────────
    case_log:               CaseLog | None        # progressively built during intake
    intake_turn_count:      int                   # tracks turns against the 8-turn limit
    intake_complete:        bool                  # True when user clicks "Get Legal Advice"
    decomposed_claims:      DecomposedClaims | None  # claim decomposer output

    # ── Intake edit detection ─────────────────────────────────────────────────
    edit_detected:          bool                  # True if current message is a log correction
    edited_fields:          list[str]             # which fields were patched this turn
    awaiting_contradiction_resolution : bool      # True if the previous turn was a contradiction, triggers a forced correction in 
                                                  # the next turn 
    claim_results:                     list[dict]    # per-claim retrieval + grading detail

def initial_state(
    raw_query:             str,
    clarification_context: str        = "",
    history:               list[dict] = None,
) -> GraphState:
    return GraphState(
        raw_query               = raw_query,
        clarification_context   = clarification_context,
        ambiguity               = None,
        situation               = None,
        document_filter         = [],
        analysis                = None,
        rewritten_query         = raw_query,
        chunks                  = [],
        retry_count             = 0,
        grader_output           = None,
        out_of_scope            = False,
        web_results             = None,
        answer                  = "",
        citations               = [],
        history                 = history or [],
        clarification_questions = [],
        case_log                = None,
        intake_turn_count       = 0,
        intake_complete         = False,
        decomposed_claims       = None,
        edit_detected           = False,
        edited_fields           = [],
        awaiting_contradiction_resolution = False,
        claim_results                     = [],


    )


def intake_initial_state(
    raw_query:        str,
    history:          list[dict] = None,
    case_log:         CaseLog    = None,
    intake_turn_count: int       = 0,
) -> GraphState:
    """
    Initial state for the intake graph.
    Carries forward an existing case_log and turn count
    so the graph can be re-invoked across multiple turns
    without losing progress.
    """
    return GraphState(
        raw_query               = raw_query,
        clarification_context   = "",
        ambiguity               = None,
        situation               = None,
        document_filter         = [],
        analysis                = None,
        rewritten_query         = raw_query,
        chunks                  = [],
        retry_count             = 0,
        grader_output           = None,
        out_of_scope            = False,
        web_results             = None,
        answer                  = "",
        citations               = [],
        history                 = history or [],
        clarification_questions = [],
        case_log                = case_log or CaseLog(),
        intake_turn_count       = intake_turn_count,
        intake_complete         = False,
        decomposed_claims       = None,
        edit_detected           = False,
        edited_fields           = [],
        awaiting_contradiction_resolution = False,
        claim_results                     = [],

    )
 