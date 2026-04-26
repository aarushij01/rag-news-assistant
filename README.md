# RAG News Assistant — Retrieval-Augmented Generation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.9-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20(Free)-purple)
![Security](https://img.shields.io/badge/Security-Hardened-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> A production-style RAG system that ingests news articles, stores them in a vector database, and answers natural language questions with cited, faithful responses — with a full security layer defending against prompt injection, SSRF attacks, and URL abuse.

---

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  Security   │  ← Input validation, injection detection, URL allowlist
│   Layer     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────────┐
│  Retriever  │────▶│   ChromaDB           │
│             │     │  (Vector Store)      │
│ Top-5 chunk │     │  sentence-transformers│
│  retrieval  │     │  embeddings (local)  │
└──────┬──────┘     └──────────────────────┘
       │
       ▼
┌─────────────┐
│  Generator  │  ← Groq API (free) — llama-3.1-8b-instant
│             │
│ Cited answer│
│ + sources   │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Flask UI   │  ← 3-tab web interface (Ask / Add Articles / Knowledge Base)
└─────────────┘
```

---

## Security Features

This project implements a dedicated security layer (`src/security.py`) that defends against real-world RAG vulnerabilities:

| Threat | Defence |
|--------|---------|
| Prompt injection | Pattern matching against 10+ injection templates |
| URL abuse / SSRF | Domain allowlist + private IP blocking |
| Token flooding | Input length limits (configurable via .env) |
| API key exposure | Loaded from .env, never hardcoded, .env in .gitignore |
| Malicious documents | Text sanitisation before vector store insertion |
| Jailbreaking | System prompt with clear context delimiters |

---

## Demo Screenshots

### Ask — AI Act Fine Query
![Ask Demo](reports/figures/demo_ask.png)

### Solar Panels Query
![Solar Demo](reports/figures/demo_solar.png)

### Adding Articles
![Add Articles](reports/figures/demo_add.png)

### Knowledge Base
![Knowledge Base](reports/figures/demo_kb.png)

### Security — Prompt Injection Blocked
![Security Demo](reports/figures/demo_security.png)

---

## Results

| Metric | Score |
|--------|-------|
| Mean Answer Relevancy | ~75% |
| Mean Faithfulness | ~80% |
| Mean Chunks Retrieved | 2-3 per query |
| Avg Response Time | ~2 seconds |

*Evaluated on 5-question ground truth set using keyword overlap and faithfulness scoring*

---

## Project Structure

```
rag-news-assistant/
├── src/
│   ├── security.py     # Input validation, injection detection, URL allowlist
│   ├── ingest.py       # Document fetching, chunking, embedding, vector store
│   ├── retriever.py    # Semantic similarity search + context formatting
│   ├── generator.py    # Groq LLM answer generation with safe prompt building
│   ├── evaluate.py     # Faithfulness + relevancy evaluation
│   └── demo.py         # Flask web interface (3 tabs)
├── tests/
│   ├── test_security.py   # 20 security tests (injection, SSRF, etc.)
│   └── test_ingest.py     # Chunking and deduplication tests
├── data/
│   └── chroma_db/         # Vector store (auto-generated)
├── reports/figures/       # Demo screenshots
├── .env.example           # Template — copy to .env and add Groq API key
├── requirements.txt
└── Makefile
```

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/aarushij01/rag-news-assistant.git
cd rag-news-assistant

# 2. Install dependencies
pip install -r requirements.txt
pip install flask

# 3. Get a free Groq API key
# Go to console.groq.com → sign up (no credit card) → API Keys → Create Key

# 4. Configure API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=gsk_your_key_here

# 5. Ingest sample documents
PYTHONPATH=. python src/ingest.py

# 6. Launch the web interface
PYTHONPATH=. python src/demo.py
# Open http://127.0.0.1:5001
```

---

## How to Use

**Ask tab** — type any question about your ingested articles and get a cited answer with source references and relevance stats.

**Add Articles tab** — paste any news article text (or add by URL from BBC, Reuters, Guardian, AP, FT) to expand the knowledge base instantly.

**Knowledge Base tab** — view all articles currently stored and how many chunks each contains.

---

## What I Learned

- Chunking strategy matters enormously — chunk size 500 with 100-char overlap outperformed both 200 and 1000 in retrieval precision
- Security in RAG is underrated — prompt injection via malicious documents is a real threat; delimiter-based prompt construction mitigates it
- Faithfulness ≠ relevancy — a model can give a relevant-sounding answer not supported by retrieved context (hallucination)
- Deduplication by content hash prevents the same article being re-ingested and inflating retrieval results
- Groq's llama-3.1-8b-instant is surprisingly capable for factual extraction tasks and generates responses in under 2 seconds

## Limitations

- Extractive focus — performs best when the answer is explicitly stated in an article
- English only
- Allowlist limits to 6 news domains — intentional for security, but reduces flexibility
- No conversation memory — each query is independent

---

## References

- Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Es et al. (2023) — RAGAS: Automated Evaluation of Retrieval Augmented Generation
- OWASP LLM Top 10 (2023) — Security risks in LLM applications
