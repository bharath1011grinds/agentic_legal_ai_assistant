#NOTE: Was forced to create this file cos of circular imports between relevance_grader and agent_state

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.documents import Document
from typing import Literal, Optional

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



class CaseParties(BaseModel):

    user_role : Literal["accused", "employee",
                        "tenant", "victim",
                        "petitioner", "other"] =Field(description="The role the user plays in this legal situation")
    

    opposing_party: Literal[
        "police",
        "employer",
        "landlord",
        "family_member",
        "government",
        "other",
    ] = Field(description="Who the user's situation is against.")


class CaseLog(BaseModel):

    #Non-Negotiables:

    #Optional here is used in place of str|None
    #Allows the field to have None values...
    #Even the non-negotiable fields have Optional because, they are filled incrementally by asking questions, not at instantiation

    incident_type: Optional[Literal[
        "arrest",
        "employment",
        "tenancy",
        "domestic",
        "consumer",
        "property",
        "criminal_accusation",
        "civil_rights",
        "other",
    ]] = Field(
        default=None,
        description=(
            "The broad legal category of the incident. "
            "Drives which documents to retrieve from the corpus."
        )
    )
 
    what_happened: Optional[str] = Field(
        default=None,
        description=(
            "Plain language summary of the core incident. "
            "This is the raw material for claim decomposition."
        )
    )
 
    parties: Optional[CaseParties] = Field(
        default=None,
        description="Who is involved — user's role and the opposing party."
    )
 
    relief_sought: Optional[Literal[
        "know_my_rights",
        "file_complaint",
        "get_bail",
        "draft_document",
        "understand_charges",
        "other",
    ]] = Field(
        default=None,
        description=(
            "What the user wants out of this interaction. "
            "Shapes the tone and direction of the synthesized answer."
        )
    )


    #Optional Enrichment fields

    timeline: Optional[str] = Field(
        default=None,
        description="When the incident happened (e.g. 'yesterday', '3 days ago', 'ongoing')."
    )
 
    jurisdiction: Optional[str] = Field(
        default=None,
        description="State or city where the incident occurred, if relevant."
    )
 
    evidence_available: list[str] = Field(
        default_factory=list,
        description=(
            "Documents or evidence the user has access to. "
            "e.g. ['FIR copy', 'employment contract', 'rent agreement']"
        )
    )
 
    prior_legal_action: Optional[str] = Field(
        default=None,
        description="Whether any legal action has already been filed or taken."
    )
 
    specific_sections_mentioned: list[str] = Field(
        default_factory=list,
        description=(
            "Any specific legal sections or articles the user already mentioned. "
            "e.g. ['Section 498A IPC', 'Article 21 Constitution']"
        )
    )
 
    open_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Fields the intake agent could not confidently fill. "
            "Passed to synthesizer to flag gaps in the answer."
        )
    )

    def non_negotiables_filled(self) -> bool:

        #Returns true only when ALL 4 non-negotiable fields are filled.
        return all([self.incident_type, self.what_happened, self.relief_sought, self.parties])
    

    def to_context_string(self) -> str:
        """
        Serializes the case log into a rich context string for the
        retrieval graph's synthesizer and claim decomposer.
        """
        lines = ["=== CASE LOG ==="]
        if self.incident_type:
            lines.append(f"Incident type   : {self.incident_type}")
        if self.what_happened:
            lines.append(f"What happened   : {self.what_happened}")
        if self.parties:
            lines.append(f"User role       : {self.parties.user_role}")
            lines.append(f"Opposing party  : {self.parties.opposing_party}")
        if self.relief_sought:
            lines.append(f"Relief sought   : {self.relief_sought}")
        if self.timeline:
            lines.append(f"Timeline        : {self.timeline}")
        if self.jurisdiction:
            lines.append(f"Jurisdiction    : {self.jurisdiction}")
        if self.evidence_available:
            lines.append(f"Evidence        : {', '.join(self.evidence_available)}")
        if self.prior_legal_action:
            lines.append(f"Prior action    : {self.prior_legal_action}")
        if self.specific_sections_mentioned:
            lines.append(f"Sections cited  : {', '.join(self.specific_sections_mentioned)}")
        if self.open_questions:
            lines.append(f"Open questions  : {', '.join(self.open_questions)}")
        lines.append("=== END CASE LOG ===")
        return "\n".join(lines)
    

class LegalClaim(BaseModel):
    """
    A single legally distinct claim decomposed from the case log.
    Each claim drives one independent retrieval + grading cycle.
    """
    claim_id:    int  = Field(description="Index of this claim, starting from 0.")
    claim_text:  str  = Field(description="Retrieval-ready legal claim text.")
    legal_area:  str  = Field(description="Which law this primarily falls under e.g. CrPC, IPC, Constitution.")
    doc_filter:  list[Literal["constitution", "ipc", "crpc"]] = Field(
        description="Which document types to retrieve for this claim.")
    

class DecomposedClaims(BaseModel):
    """Output of the claim decomposer - list of legally distinct claims."""
    claims: list[LegalClaim] = Field(
        description=(
            "List of legally distinct claims extracted from the case log. "
            "Maximum 4 claims. Similar claims are merged into one."
        )
    )