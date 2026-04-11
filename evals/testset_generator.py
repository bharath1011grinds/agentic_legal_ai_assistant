import os
import json
import time
import itertools
from pathlib import Path
from dotenv import load_dotenv
 
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph
from ragas.run_config import RunConfig
from ragas.testset.transforms import HeadlinesExtractor, SummaryExtractor, KeyphrasesExtractor, TitleExtractor


from ragas.testset.transforms import default_transforms

load_dotenv()
 
# ── Config ────────────────────────────────────────────────────────────────────
 
DOCS_ROOT   = Path("legal_docs")
OUTPUT_PATH = Path("evals/testset.json")
TESTSET_SIZE = 30
 
# Distribution: 50% single-hop factual, 25% multi-hop abstract, 25% multi-hop specific
# Single-hop is most relevant for a legal Q&A assistant
QUERY_DISTRIBUTION = [
    #Factual, can be answered looking at 1 doc
    (SingleHopSpecificQuerySynthesizer, 0.5),

    #Factual but needs info from multiple docs(chunks)
    (MultiHopAbstractQuerySynthesizer,  0.25),

    #Comparison type qs, no direct answer present in the docs, needs sysnthesis
    (MultiHopSpecificQuerySynthesizer,  0.25),
]

def load_legal_docs() -> list:
    """
    Loads all documents from legal_docs/constitution/, legal_docs/ipc/, legal_docs/crpc/.
    Supports .pdf and .txt files.
    Tags each document with its source corpus in metadata.
    """
    docs = []
    corpus_dirs = {
        "constitution": DOCS_ROOT / "constitution",
        "ipc":          DOCS_ROOT / "ipc",
        "crpc":         DOCS_ROOT / "crpc",
    }
 
    for corpus, dirpath in corpus_dirs.items():
        if not dirpath.exists():
            print(f"[Loader] Warning: {dirpath} not found — skipping")
            continue
 
        files = list(dirpath.glob("*.pdf")) + list(dirpath.glob("*.txt"))
        print(f"[Loader] {corpus}: {len(files)} files")
 
        for fpath in files:
            try:
                if fpath.suffix == ".PDF":
                    loader = PyMuPDFLoader(str(fpath))

                else:
                    loader = TextLoader(str(fpath), encoding="utf-8")

                #Loads each page in the pdf as a Document object.
                loaded = loader.load()
                #print("post*****")
 
                # Tag every page/chunk with corpus metadata
                for doc in loaded:
                    doc.metadata["corpus"]    = corpus
                    doc.metadata["source"]    = fpath.name
                    doc.metadata["file_path"] = str(fpath)
 
                docs.extend(loaded)
                print(f"  Loaded {fpath.name} — {len(loaded)} pages")
 
            except Exception as e:
                print(f"  Error loading : {e} — skipping")
 
    print(f"\n[Loader] Total documents loaded: {len(docs)}")
    return docs

#Cycle the api keyss
#Gotta make sure this is GLOBAL and not inside the fn
api_cycler = itertools.cycle(["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_4", "GROQ_API_KEY"])

#Return the LLM as a LangchainLLM wrapper cos, RAGAs expects it that way to be LLM agnostic and to carry out evals seamlessly
def build_ragas_llm() -> LangchainLLMWrapper:
    """
    Wraps Groq LLM for RAGAs.
    Uses a lower temperature and conservative max_tokens for generation stability.
    """


    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=1024,
        # THIS IS THE KEY FIX:
        model_kwargs={"response_format": {"type": "json_object"}}, 

        #rotate the keys with next
        api_key=os.environ.get(next(api_cycler)),
    )

    return LangchainLLMWrapper(llm)



#Same wiith the embeddings, return as a wrapper cos RAGAs expects it, to be agnostic to the type of embeddings used.
def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Wraps HuggingFace embeddings for RAGAs diversity scoring.
    Reuses the same model as your vectorstore for consistency.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name    = "all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(embeddings)



def generate_testset(docs: list) -> list:
    """
    Runs RAGAs TestsetGenerator on the loaded documents.
    Returns a list of dicts ready to be saved as JSON.
    """
    ragas_llm        = build_ragas_llm()
    ragas_embeddings = build_ragas_embeddings()


    #call the llm builder explicitly to cycle the keys
    synthesizers = [
        (synthesizer_cls(llm=build_ragas_llm()), weight)
        for synthesizer_cls, weight in QUERY_DISTRIBUTION
    ]


    generator = TestsetGenerator(
        llm             = ragas_llm,
        embedding_model = ragas_embeddings,
    )
 
    # RunConfig controls timeouts and retries
    # Groq free tier has rate limits so we add generous timeouts
    run_config = RunConfig(
        timeout    = 120,
        max_retries = 5,
        max_wait    = 90,
    )


    print(f"\n[Generator] Generating {TESTSET_SIZE} test questions...")
    print(f"[Generator] Distribution: {[(cls.__name__, w) for cls, w in QUERY_DISTRIBUTION]}")
    print(f"[Generator] This may take several minutes due to Groq rate limits...\n")


    #Need to add the full list of transforms RAGAs makes manually cos, the headlinesplitter fired before the headlineextractor
    #Init the llm repititively to cycle the keys
    custom_transforms = [
        HeadlinesExtractor(llm=build_ragas_llm()),
        SummaryExtractor(llm=build_ragas_llm()),
        KeyphrasesExtractor(llm=build_ragas_llm()),
        TitleExtractor(llm=build_ragas_llm())
    ]

    testset = generator.generate_with_langchain_docs(
        documents          = docs,
        testset_size       = TESTSET_SIZE,
        query_distribution = synthesizers,
        transforms=custom_transforms,
        run_config         = run_config,
        raise_exceptions   = False,   # don't crash on individual failures
    )
 
    return testset
 

def save_testset(testset, output_path: Path):
    """
    Converts RAGAs testset to a clean JSON format and saves it.
 
    Output schema per item:
    {
        "question":     str,
        "ground_truth": str,
        "contexts":     [],        # empty at generation time — filled during eval
        "metadata": {
            "synthesizer_type": str,
            "corpus":           str,
        }
    }
 
    NOTE: contexts are left empty here. They get populated during eval
    when we run your actual retriever against each question.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    records = []
    df = testset.to_pandas()

    for _, row in df.iterrows():
        record = {
            "question":     str(row.get("user_input", "")),
            "ground_truth": str(row.get("reference", "")),
            "contexts":     [],   # populated by eval script
            "metadata": {
                "synthesizer_type": str(row.get("synthesizer_name", "")),
                "corpus":           str(row.get("reference_contexts", [""])[0][:30] if row.get("reference_contexts") else ""),
            }
        }
        if record["question"]:   # skip empty questions
            records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
 
    print(f"\n[Generator] Saved {len(records)} questions to {output_path}")
    return records


if __name__ == "__main__":
    start = time.time()
 
    print("=" * 60)
    print("RAGAs Testset Generator — KnowYourRights")
    print("=" * 60)
 
    # Step 1 — Load docs
    docs = load_legal_docs()
    if not docs:
        print("[Error] No documents loaded. Check legal_docs/ directory.")
        exit(1)
 
    # Step 2 — Generate
    testset = generate_testset(docs)
 
    # Step 3 — Save
    records = save_testset(testset, OUTPUT_PATH)
 
    elapsed = time.time() - start
    print(f"\n[Done] {len(records)} questions generated in {elapsed:.1f}s")
    print(f"Next step: run  python eval/run_eval.py")