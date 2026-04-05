"""
ingest_legal.py — KnowYourRights.ai ingestion pipeline

Directory structure expected:
    legal_docs/
        constitution/   *.pdf
        ipc/            *.pdf
        crpc/           *.pdf

Outputs:
    vectorstore/knowyourrights/   FAISS index (dual-embedded: legal text + plain language)

Usage:
    python ingest_legal.py
    python ingest_legal.py --docs_dir legal_docs --index_dir vectorstore/knowyourrights
    python ingest_legal.py --no_summary   (skip Haiku calls, faster for testing)
"""

import os
import re
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_MODEL    = "claude-haiku-4-5-20251001"
MAX_CHUNK_WORDS  = 300
OVERLAP_WORDS    = 40

AMENDMENT_PATTERNS = [
    r'\[Substituted by[^\]]*?\]',
    r'\[Inserted by[^\]]*?\]',
    r'\[Omitted by[^\]]*?\]',
    r'\[Prior to[^\]]*?\]',
    r'\[Added by[^\]]*?\]',
]
EDITORIAL_PATTERN  = r'\[Editorial comment.*?\]'
FOOTER_PATTERN     = r'(?:Section|Article)\s+[\dA-Z]+\s+in\s+The[^\n]+\nIndian Kanoon[^\n]+'
REFERENCES_PATTERN = r'References[\s\S]*$'

TOPIC_MAP = {
    "constitution": {
        "speech":     ["speech", "expression", "press", "publish", "silence"],
        "assembly":   ["assemble", "assembly", "procession", "meeting", "union"],
        "equality":   ["equality", "equal", "discrimination", "caste"],
        "life":       ["life", "liberty", "personal", "dignity", "privacy"],
        "property":   ["property", "residence", "settle", "reside"],
        "profession": ["profession", "occupation", "trade", "business"],
        "arrest":     ["arrest", "detention", "preventive", "custody"],
    },
    "ipc": {
        "theft":      ["theft", "steal", "moveable", "possession", "dishonest"],
        "assault":    ["assault", "hurt", "force", "attack", "grievous"],
        "murder":     ["murder", "culpable homicide", "death", "kill"],
        "fraud":      ["fraud", "cheating", "deceive", "misrepresent"],
        "punishment": ["imprisonment", "fine", "sentence", "punish", "rigorous"],
    },
    "crpc": {
        "arrest":        ["arrest", "warrant", "cognisable", "without warrant"],
        "bail":          ["bail", "bond", "release", "surety"],
        "trial":         ["trial", "charge", "hearing", "session"],
        "investigation": ["investigation", "inquiry", "fir", "report"],
        "notice":        ["notice", "appear", "comply", "summons"],
    },
}


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_ipc_crpc(text: str) -> str:
    for p in AMENDMENT_PATTERNS:
        text = re.sub(p, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(FOOTER_PATTERN, "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_constitution(text: str) -> str:
    text = re.sub(EDITORIAL_PATTERN, "", text, flags=re.DOTALL)
    text = re.sub(FOOTER_PATTERN, "", text, flags=re.IGNORECASE)
    text = re.sub(REFERENCES_PATTERN, "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ── Metadata helpers ──────────────────────────────────────────────────────────

def extract_topic_tags(text: str, doc_type: str) -> list[str]:
    text_lower = text.lower()
    tags = [tag for tag, kws in TOPIC_MAP.get(doc_type, {}).items() if any(kw in text_lower for kw in kws)]
    return tags or ["general"]


def extract_cross_refs(text: str) -> list[str]:
    refs = re.findall(r'\b(?:section|article|s\.)\s*(\d+[A-Z]?(?:\(\d+\))?)', text, flags=re.IGNORECASE)
    return list(set(refs))


def extract_key_cases(text: str) -> list[str]:
    cases = re.findall(r'[A-Z][a-zA-Z\s]+v\.\s+[A-Z][a-zA-Z\s]+(?:\(\d{4}\))?', text)
    return [c.strip() for c in cases[:5]]


def extract_url(text: str) -> str:
    m = re.search(r'https?://[^\s\]]+', text)
    return m.group(0) if m else ""


# ── Sliding window splitter ───────────────────────────────────────────────────

def sliding_chunks(text: str, meta: dict) -> list[Document]:
    """Split text into overlapping word-count-bounded chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + MAX_CHUNK_WORDS, len(words))
        chunk = Document(page_content=" ".join(words[start:end]), metadata=meta)
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - OVERLAP_WORDS
    return chunks


# ── IPC Chunking ──────────────────────────────────────────────────────────────

def chunk_ipc(raw_text: str, filename: str) -> list[Document]:
    url  = extract_url(raw_text)
    text = clean_ipc_crpc(raw_text)

    m       = re.match(r'(\d+[A-Z]?)\.\s+([^—\n]+)', text.strip())
    sec_num = m.group(1) if m else ""
    title   = m.group(2).strip() if m else ""

    illus_m = re.search(r'\nIllustrations?\s*\n', text, re.IGNORECASE)

    base_meta = {
        "document_type":  "ipc",
        "section_number": sec_num,
        "title":          title,
        "url":            url,
        "source":         filename,
        "topic_tags":     extract_topic_tags(text, "ipc"),
        "cross_refs":     extract_cross_refs(text),
    }

    chunks = []
    if illus_m:
        body  = text[:illus_m.start()].strip()
        illus = text[illus_m.start():].strip()
        chunks += sliding_chunks(body,  {**base_meta, "chunk_type": "definition"})
        chunks += sliding_chunks(illus, {**base_meta, "chunk_type": "illustrations"})
    else:
        chunks += sliding_chunks(text,  {**base_meta, "chunk_type": "definition"})

    return chunks


# ── CrPC Chunking ─────────────────────────────────────────────────────────────

def _split_crpc_sections(text: str) -> list[tuple[str, str, str]]:
    """Split CrPC text into (section_number, title, body) tuples. Handles inline sub-sections (e.g. 41A)."""
    pattern = re.compile(r'\n(\d+[A-Z]?)\.\s+([^\n.]+\.)\s*\n', re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        m       = re.match(r'(\d+[A-Z]?)\.\s+([^\n]+)\n', text.strip())
        sec_num = m.group(1) if m else ""
        title   = m.group(2).strip() if m else ""
        return [(sec_num, title, text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1), m.group(2).strip(), text[start:end].strip()))

    return sections


def chunk_crpc(raw_text: str, filename: str) -> list[Document]:
    url      = extract_url(raw_text)
    text     = clean_ipc_crpc(raw_text)
    sections = _split_crpc_sections(text)
    chunks   = []

    for sec_num, title, body in sections:
        meta = {
            "document_type":  "crpc",
            "section_number": sec_num,
            "title":          title,
            "url":            url,
            "source":         filename,
            "topic_tags":     extract_topic_tags(body, "crpc"),
            "cross_refs":     extract_cross_refs(body),
            "chunk_type":     "section",
        }
        chunks += sliding_chunks(body, meta)

    return chunks


# ── Constitution Chunking ─────────────────────────────────────────────────────

def chunk_constitution(raw_text: str, filename: str) -> list[Document]:
    url       = extract_url(raw_text)
    key_cases = extract_key_cases(raw_text)   # extract before editorial strip
    text      = clean_constitution(raw_text)

    m       = re.match(r'(\d+[A-Z]?)\.\s+([^\n]+)', text.strip())
    art_num = m.group(1) if m else ""
    title   = m.group(2).strip() if m else ""

    meta = {
        "document_type":  "constitution",
        "article_number": art_num,
        "title":          title,
        "url":            url,
        "source":         filename,
        "key_cases":      key_cases,
        "topic_tags":     extract_topic_tags(text, "constitution"),
        "cross_refs":     extract_cross_refs(text),
        "chunk_type":     "article",
    }

    return sliding_chunks(text, meta)


# ── Plain Language Summary (dual embedding) ───────────────────────────────────\
def generate_plain_summary(chunk:Document) -> str:

    api_keys = [os.environ.get(f"GROQ_API_KEY_{n}") for n in range(1,6)]
    # 1. Setup metadata and content as before
    doc_type = chunk.metadata.get("document_type", "legal")
    ref = chunk.metadata.get("section_number") or chunk.metadata.get("article_number")
    ref_str = f" (Section/Article {ref})" if ref else ""
    
    system_instruction = (
        "You are a legal expert who explains laws to Indian citizens in simple, everyday English. "
        "Provide ONLY the summary text. No intro phrases, no JSON, and no legal jargon."
    )
    prompt = (
        f"Summarise this {doc_type} provision{ref_str} in 1-2 plain sentences: \n\n"
        f"{chunk.page_content[:800]}"
    )
    messages = [SystemMessage(content=system_instruction), HumanMessage(content=prompt)]

    # 2. Key rotation logic
    current_key_index = 0

    while current_key_index < len(api_keys):
        try:
            # Initialize LLM with the current key
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                api_key=api_keys[current_key_index], 
                temperature=0.1
            )
            
            result = llm.invoke(messages)
            return result.content.strip()
        except Exception as e:
            # Check if the error is a 429 Rate Limit
            if "429" in str(e):
                print(f"Rate limit hit for key {current_key_index}. Switching to next key...")
                current_key_index += 1
                if current_key_index >= len(api_keys):
                    return "Error: All API keys exhausted due to rate limits."
                continue  # Retry the loop with the next key
            else:
                # Handle other types of errors (auth, connectivity, etc.)
                return f"Summary unavailable due to error: {str(e)}"

    return "Summary unavailable: No valid keys remaining."


def make_summary_chunk(chunk: Document, summary: str) -> Document:
    return Document(
        page_content=summary,
        metadata={
            **chunk.metadata,
            "chunk_type":        "plain_summary",
            "parent_chunk_type": chunk.metadata.get("chunk_type", ""),
        }
    )


# ── Embedding & Index ─────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )


def build_or_update_index(docs: list[Document], index_path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    new_index = FAISS.from_documents(docs, embeddings)

    if os.path.exists(index_path):
        print(f"  Merging into existing index at '{index_path}'...")
        existing = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        existing.merge_from(new_index)
        existing.save_local(index_path)
        return existing

    os.makedirs(index_path, exist_ok=True)
    new_index.save_local(index_path)
    print(f"  Index saved at '{index_path}'")
    return new_index


# ── Entrypoint ────────────────────────────────────────────────────────────────

CHUNKERS = {
    "constitution": chunk_constitution,
    "ipc":          chunk_ipc,
    "crpc":         chunk_crpc,
}


def ingest_legal(
    docs_dir:   str  = "legal_docs",
    index_path: str  = "vectorstore/knowyourrights",
    summarize:  bool = True,
) -> FAISS:

    embeddings = get_embeddings()
    all_chunks: list[Document] = []
    stats = {k: 0 for k in CHUNKERS}

    for doc_type, chunker in CHUNKERS.items():
        folder = Path(docs_dir) / doc_type
        if not folder.exists():
            print(f"[ingest] Skipping '{doc_type}' — not found: {folder}")
            continue

        pdfs = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
        print(f"\n[ingest] {doc_type.upper()} — {len(pdfs)} PDFs")

        for pdf_path in pdfs:
            print(f"  {pdf_path.name}")
            pages    = PyPDFLoader(str(pdf_path)).load()
            raw_text = "\n".join(p.page_content for p in pages)
            chunks   = chunker(raw_text, pdf_path.name)

            if summarize:
                summary_chunks = []
                for chunk in chunks:
                    if chunk.metadata.get("chunk_type") == "plain_summary":
                        continue
                    try:
                        summary = generate_plain_summary(chunk)
                        summary_chunks.append(make_summary_chunk(chunk, summary))
                    except Exception as e:
                        print(f"    [warn] Summary failed: {e}")
                chunks += summary_chunks
                print(f"    {len(chunks)//2} legal + {len(summary_chunks)} summary chunks")
            else:
                print(f"    {len(chunks)} chunks")

            all_chunks.extend(chunks)
            stats[doc_type] += len(chunks)

    print(f"\n[ingest] Total: {len(all_chunks)} chunks")
    for k, v in stats.items():
        print(f"  {k:<14}: {v}")

    print(f"\n[ingest] Building index...")
    index = build_or_update_index(all_chunks, index_path, embeddings)
    print("[ingest] Done.\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir",   default="legal_docs")
    parser.add_argument("--index_dir",  default="vectorstore/knowyourrights")
    parser.add_argument("--no_summary", action="store_true", help="Skip Haiku summary calls (faster for testing)")
    args = parser.parse_args()

    ingest_legal(
        docs_dir   = args.docs_dir,
        index_path = args.index_dir,
        summarize  = not args.no_summary,
    )