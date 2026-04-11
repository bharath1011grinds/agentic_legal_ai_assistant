# ⚖️ KnowYourRights.ai

An agentic RAG pipeline that routes, classifies, and retrieves before answering — built for Indian citizens to understand their legal rights under the Constitution, IPC, and CrPC.

---

## What It Does

KnowYourRights.ai is an AI-powered legal assistant that helps Indian citizens navigate their fundamental rights and legal procedures. It supports two modes:

**Quick Mode** — Ask a legal question and get a cited, plain-English answer backed by retrieved legal sources.

**Intake Mode** — Describe your situation conversationally. The system builds a structured case log, decomposes it into legal claims, retrieves relevant law for each claim, and delivers a comprehensive legal briefing.

---

## Why Agentic?

This is not a simple LLM wrapper. Every query passes through a decision graph before an answer is generated:

- **Context Resolver** — rewrites the query using conversation history for multi-turn coherence
- **Ambiguity Detector** — decides whether to ask clarifying questions or proceed
- **Situation Classifier** — identifies the legal domain (arrest, civil rights, criminal offence, etc.) and applies the appropriate document filter
- **Relevance Grader** — scores retrieved chunks and decides whether they are good enough to synthesize from, or whether to route to a KB miss response
- **Scope Guard** — catches out-of-scope queries (tax, corporate, IP law) and redirects gracefully
- **KB Miss Node** — handles cases where the knowledge base lacks sufficient information honestly, without hallucinating law

Each node is a real decision point that changes the execution path — not a linear chain of LLM calls.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agentic Pipeline | LangGraph |
| LLM | Llama 3.3 70B via Groq |
| Embeddings | Sentence Transformers |
| Retrieval | FAISS + BM25 Hybrid Search |
| Reranking | Cross-Encoder |
| Web Fallback | Tavily Search API |
| Observability | Langfuse |
| Backend | FastAPI (SSE streaming) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Knowledge Base

The KB covers:
- 🇮🇳 Indian Constitution — selected fundamental rights articles
- ⚖️ Indian Penal Code (IPC)
- 📋 Code of Criminal Procedure (CrPC)

Queries outside this scope (tax law, corporate law, intellectual property, consumer disputes) are detected and routed to the Scope Guard, which redirects users to appropriate legal resources.

---

## Limitations

This project is developed for **learning purposes** and comes with the following limitations:

- **Limited Knowledge Base** — The KB covers only select provisions of the Constitution, IPC, and CrPC. Queries outside this scope hit the KB miss node and receive an honest "insufficient information" response rather than a hallucinated answer.
- **No Database** — User session data and conversation history are stored in-memory only. There is no persistent storage — sessions are lost on server restart. This means the system cannot provide a complete, stateful experience across sessions.
- **Not Legal Advice** — This system is an educational tool. It is not a substitute for consultation with a qualified lawyer. Always verify any legal information with a professional before acting on it.
- **English Only** — The system currently supports English queries only.

---

## Project Structure

```
├── server.py                   # FastAPI server — all endpoints + SSE streaming
├── pipeline_agentic.py         # Quick mode LangGraph pipeline
├── intake_graph.py             # Intake conversation graph
├── intake_retrieval_graph.py   # Parallel claim retrieval graph
├── hybrid_retriever_phase2.py  # FAISS + BM25 hybrid retriever
├── observability.py            # Langfuse traced wrappers for all nodes
├── agents/
│   ├── context_resolver.py
│   ├── ambiguity_detector.py
│   ├── situation_classifier.py
│   ├── retriever.py
│   ├── relevance_grader.py
│   ├── scope_guard.py
│   ├── kb_miss_node.py
│   ├── synthesizer.py
│   ├── intake_agent.py
│   ├── claim_retriever.py
│   ├── edit_detector.py
│   └── intake_synthesizer.py
└── vectorstore/                # FAISS index (Constitution, IPC, CrPC)
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Primary Groq API key |
| `GROQ_API_KEY_1` | Groq key for load distribution |
| `GROQ_API_KEY_2` | Groq key for classifiers |
| `GROQ_API_KEY_3` | Groq key for retrieval nodes |
| `GROQ_API_KEY_4` | Groq key for intake nodes |
| `TAVILY_API_KEY` | Tavily web search fallback |
| `LANGFUSE_PUBLIC_KEY` | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | Langfuse observability |

---

*Built as a learning project exploring agentic RAG system design, LangGraph orchestration, and production observability patterns.*