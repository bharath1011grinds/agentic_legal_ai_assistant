#NOTE: Was forced to create this file cos of circular imports between relevance_grader and agent_state

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.documents import Document
from typing import Literal

class QueryAnalysis(BaseModel):
 
    query_type: Literal["factual", "comparative", "exploratory"] = Field(
        description=(
            "factual: specific section/article lookup or definition. "
            "comparative: comparing two laws, rights, or offences. "
            "exploratory: open-ended survey of a legal topic."
        )
    )
 
    web_search_needed: bool = Field(
        description=(
            "True only if query requires recent case law, amendments post-2023, "
            "or information clearly outside the KB (constitution/IPC/CrPC)."
        )
    )
 
    entities: list[str] = Field(
        description=(
            "Key legal terms, section numbers, article numbers, offence names, "
            "or rights extracted from the query. Seeds BM25 + semantic search."
        )
    )
 
    rewritten_query: str = Field(
        description=(
            "Retrieval-friendly rewrite of the query using proper legal terminology. "
            "Expand colloquial terms (e.g. 'cops taking me in' → 'police arrest without warrant'). "
            "Add relevant IPC/CrPC/Constitution context."
        )
    )



class SituationClass(BaseModel):
 
    situation: Literal[
        "arrest_detention",
        "criminal_offence",
        "civil_rights",
        "property",
        "employment",
        "family",
        "out_of_scope",
    ] = Field(
        description=(
            "arrest_detention: user is being/was arrested, detained, or stopped by police. "
            "criminal_offence: user is accused of or victim of a crime under IPC. "
            "civil_rights: query about fundamental rights under the Constitution. "
            "property: disputes involving moveable/immoveable property. "
            "employment: workplace rights, labour law. "
            "family: divorce, custody, inheritance, domestic matters. "
            "out_of_scope: topic falls outside constitution/IPC/CrPC KB."
        )
    )
 
    document_filter: list[Literal["constitution", "ipc", "crpc"]] = Field(
        description=(
            "Which document types to bias retrieval toward for this situation. "
            "arrest_detention → ['crpc', 'constitution']. "
            "criminal_offence → ['ipc', 'crpc']. "
            "civil_rights → ['constitution']. "
            "property/employment/family → ['ipc']. "
            "out_of_scope → []."
        )
    )
 
    confidence: float = Field(
        description="Classification confidence between 0 and 1."

    )


class AmbiguityCheck(BaseModel):
 
    is_ambiguous: bool = Field(
        description=(
            "True if the query lacks enough context to retrieve accurately. "
            "e.g. 'what are my rights' with no situation context = ambiguous. "
            "'can police arrest me without a warrant for theft' = not ambiguous."
        )
    )
 
    clarifying_questions: list[str] = Field(
        description=(
            "1-2 short clarifying questions to resolve ambiguity. "
            "Empty list if is_ambiguous=False. "
            "Each question must have 2-4 short option labels appended as a list. "
            "Format: 'Question text|||Option1|Option2|Option3'"
        )
    )
 
    reason: str = Field(
        description="One line reason for the ambiguity decision."
    )
 


class ChunkGrade(BaseModel):
    
    chunk_index : int = Field(description="Index of the chunk in the Input list")
    relevance_score : float = Field(description="Normalized Relevance score between 0 and 1")
    passed : bool = Field(description="True if score >= SCORE_THRESHOLD")
    reason : str = Field("1 line LLM's reason for why the chunk passed or failed")


class GraderOutput(BaseModel):
 
    model_config = ConfigDict(arbitrary_types_allowed=True)
 
    chunk_grades: list[ChunkGrade] = Field(description="Per-chunk grades.")
    passed_chunks: list[Document] = Field(description="Chunks that passed the threshold.")
    failed_chunks: list[Document] = Field(description="Chunks that were filtered out.")
    rerouting_decision: Literal["proceed", "detect"] = Field(
        description="proceed: enough good chunks. rewrite: retry with rewritten query. fallback: go to Tavily or scope guard."
    )
    confidence_reason: str = Field(description="One line reason for routing decision.")
