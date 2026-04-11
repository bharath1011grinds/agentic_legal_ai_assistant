import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv


# Add project root to path so we can import pipeline modules
#0 puts the added folder(our root) the first priority of the search list.. i.e. our root becomes the first element in the 
#search list.

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
 
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


#CONFIG: 

TESTSET_PATH = Path("eval/testset.json")
RESULTS_PATH = Path("eval/results.json")
CSV_PATH     = Path("eval/results_table.csv")


def load_pipeline():
    """
    Loads your existing retriever and synthesizer.
    Returns (retriever, llm) ready to use.
    """
    from agents.retriever import get_retriever
    from hybrid_retriever_phase2 import get_cross_encoder
 
    retriever    = get_retriever()
    cross_encoder = get_cross_encoder()
 
    llm = ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0,
        max_tokens  = 1024,
        api_key     = os.environ.get("GROQ_API_KEY"),
    )
 
    return retriever, cross_encoder, llm


import math
 
SCORE_THRESHOLD    = 0.4
MIN_PASSING_CHUNKS = 2
 
def _sigmoid(score: float) -> float:
    return 1.0 / (1 + math.exp(-score))
 

def run_pipeline_for_question(
    question:     str,
    retriever,
    cross_encoder,
    llm:          ChatGroq,
) -> tuple[str, list[str]]:
    """
    Runs retrieval + grading + synthesis for a single question.
    Returns (answer, contexts) where contexts is a list of chunk texts.
 
    Uses your existing retriever and cross-encoder directly —
    no LangGraph overhead needed for eval.
    """
    # Step 1 — Retrieve
    try:
        chunks = retriever._get_relevant_documents(question)
    except Exception as e:
        print(f"  [Pipeline] Retrieval error: {e}")
        return "", []
 
    if not chunks:
        return "No relevant information found.", []
 
    # Step 2 — Grade with cross-encoder
    pairs      = [(question, doc.page_content) for doc in chunks]
    raw_scores = cross_encoder.predict(pairs)
    norm_scores = [_sigmoid(s) for s in raw_scores]
 
    passed_chunks = [
        doc for doc, score in zip(chunks, norm_scores)
        if score >= SCORE_THRESHOLD
    ]
 
    if not passed_chunks:
        passed_chunks = chunks[:2]  # fallback — use top 2 even if below threshold
 
    contexts = [doc.page_content for doc in passed_chunks]

    # Step 3 — Synthesize
    from langchain_core.messages import SystemMessage, HumanMessage
    
#We ask for citations here to make sure the AI is looking at the right laws to answer...
#This is a use-case specific thing we are doing. 
#Something similar for a health AI would be to cite the reason(an underlying medical condition or sum) behind the prescription...
    system_prompt = """You are KnowYourRights, a legal assistant for Indian citizens.
Answer using ONLY the sources provided. Cite specific Section or Article numbers.
Write in plain English. Be concise and accurate."""
 
    context_block = "\n\n".join(
        f"[Source {i+1}]\n{ctx[:600]}" for i, ctx in enumerate(contexts)
    )
 
    human_prompt = f"Question: {question}\n\nSources:\n{context_block}"
 
    try:
        response     = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        answer = response.content.strip()
    except Exception as e:
        print(f"  [Pipeline] Synthesis error: {e}")
        answer = ""
 
    return answer, contexts



#Build RAGAs Components
def build_ragas_components():
    ragas_llm = LangchainLLMWrapper(ChatGroq(
        model       = "llama-3.3-70b-versatile",
        temperature = 0,
        max_tokens  = 1024,
        api_key     = os.environ.get("GROQ_API_KEY"),
    ))
 
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name    = "all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings": True},
    ))
 
    return ragas_llm, ragas_embeddings


def run_ragas_eval(samples: list[SingleTurnSample], ragas_llm, ragas_embeddings) -> dict:
    """
    Runs RAGAs metrics on the collected samples.
    Metrics:
      - Faithfulness         : is the answer grounded in the retrieved contexts?
      - AnswerRelevancy      : does the answer address the question?
      - ContextPrecision     : are the retrieved contexts actually useful?
    """
    dataset = EvaluationDataset(samples=samples)
 
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        LLMContextPrecisionWithoutReference(llm=ragas_llm),
    ]
 
    run_config = RunConfig(
        timeout     = 120,
        max_retries = 3,
        max_wait    = 60,
    )
 
    print(f"\n[RAGAs] Scoring {len(samples)} samples...")
    print("[RAGAs] Metrics: Faithfulness, AnswerRelevancy, ContextPrecision")
    print("[RAGAs] This may take a few minutes...\n")
 
    result = evaluate(
        dataset        = dataset,
        metrics        = metrics,
        llm            = ragas_llm,
        embeddings     = ragas_embeddings,
        run_config     = run_config,
        raise_exceptions = False,
        show_progress  = True,
    )
 
    return result


 
def save_results(result, samples: list[SingleTurnSample], testset: list):
    """
    Saves per-question scores and aggregate to JSON and CSV.
    """
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    df = result.to_pandas()
 
    # Aggregate scores
    aggregate = {
        metric: round(float(df[metric].mean()), 4)
        for metric in df.columns
        if metric not in ("user_input", "response", "retrieved_contexts", "reference")
        and df[metric].dtype in ("float64", "float32")
    }
 
    # Per-question records
    per_question = []
    for i, row in df.iterrows():
        record = {
            "question":     row.get("user_input", testset[i]["question"] if i < len(testset) else ""),
            "answer":       row.get("response", ""),
            "ground_truth": testset[i]["ground_truth"] if i < len(testset) else "",
            "scores": {
                metric: round(float(row[metric]), 4)
                for metric in aggregate
                if metric in row and row[metric] is not None
            }
        }
        per_question.append(record)
 
    output = {
        "aggregate": aggregate,
        "per_question": per_question,
        "total_questions": len(per_question),
    }
 
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
 
    # Save CSV for README table
    df.to_csv(CSV_PATH, index=False)
 
    return aggregate
 
 
def print_summary(aggregate: dict, total: int, elapsed: float):
    print("\n" + "=" * 60)
    print("RAGAs Evaluation Results — KnowYourRights")
    print("=" * 60)
    print(f"Questions evaluated : {total}")
    print(f"Time taken          : {elapsed:.1f}s")
    print()
    print(f"{'Metric':<35} {'Score':>8}")
    print("-" * 44)
    for metric, score in aggregate.items():
        bar = "█" * int(score * 20)
        flag = " ✓" if score >= 0.7 else " ✗"
        print(f"{metric:<35} {score:>6.4f}  {bar}{flag}")
    print("=" * 60)
    print(f"\nDetailed results : {RESULTS_PATH}")
    print(f"CSV table        : {CSV_PATH}")


if __name__ == "__main__":
    start = time.time()
 
    print("=" * 60)
    print("RAGAs Evaluation Runner — KnowYourRights")
    print("=" * 60)
 
    # Step 1 — Load testset
    if not TESTSET_PATH.exists():
        print(f"[Error] Testset not found at {TESTSET_PATH}")
        print("Run:  python eval/generate_testset.py  first")
        sys.exit(1)
 
    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
 
    print(f"[Eval] Loaded {len(testset)} questions from {TESTSET_PATH}")
 
    # Step 2 — Load pipeline
    print("[Eval] Loading retriever and cross-encoder...")
    retriever, cross_encoder, synth_llm = load_pipeline()

    # Step 3 — Run pipeline for each question
    samples = []
    for i, item in enumerate(testset):
        question = item["question"]
        print(f"\n[{i+1}/{len(testset)}] {question[:80]}")
 
        answer, contexts = run_pipeline_for_question(
            question      = question,
            retriever     = retriever,
            cross_encoder = cross_encoder,
            llm           = synth_llm,
        )
 
        print(f"  → {len(contexts)} contexts | answer: {answer[:60]}...")
 
        # Update testset record with retrieved contexts
        testset[i]["contexts"] = contexts
 
        sample = SingleTurnSample(
            user_input          = question,
            response            = answer,
            retrieved_contexts  = contexts,
            reference           = item.get("ground_truth", ""),
        )
        samples.append(sample)
 
        # Small delay to respect Groq rate limits
        time.sleep(1.5)
 
    # Save testset with populated contexts back to disk
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)
 
    # Step 4 — RAGAs scoring
    ragas_llm, ragas_embeddings = build_ragas_components()
    result = run_ragas_eval(samples, ragas_llm, ragas_embeddings)
 
    # Step 5 — Save and print
    elapsed   = time.time() - start
    aggregate = save_results(result, samples, testset)
    print_summary(aggregate, len(samples), elapsed)