# RAG News Assistant — Retrieval-Augmented Generation

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.9-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![Security](https://img.shields.io/badge/Security-Hardened-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> A production-style RAG system that ingests news articles, stores them in a vector database, and answers natural language questions with cited, faithful responses — with a full security layer.

**[Live Demo →](https://huggingface.co/spaces/YOUR_USERNAME/rag-news-assistant)**

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
┌─────────────┐     ┌──────────────────┐
│  Retriever  │────▶│   ChromaDB       │
│             │     │  (Vector Store)  │
│ Top-5 chunk │     │  sentence-trans  │
│  retrieval  │     │  embeddings      │
└──────┬──────┘     └──────────────────┘
       │
       ▼
┌─────────────┐
│  Generator  │  ← Claude claude-haiku-4-5-20251001 via Anthropic API
│             │
│ Cited answer│
│ + sources   │
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

## Results

| Metric | Score |
|--------|-------|
| Mean Answer Relevancy | 84% |
| Mean Faithfulness | 79% |
| Mean Chunks Retrieved | 4.2 / 5 |

*Evaluated on 5-question ground truth set — see `reports/rag_evaluation.csv`*

---

## Project Structure

```
rag-news-assistant/
├── src/
│   ├── security.py     # Input validation, injection detection, URL allowlist
│   ├── ingest.py       # Document fetching, chunking, embedding, vector store
│   ├── retriever.py    # Semantic similarity search + context formatting
│   ├── generator.py    # LLM answer generation with safe prompt building
│   ├── evaluate.py     # Faithfulness + relevancy evaluation
│   └── demo.py         # Gradio web interface (3 tabs)
├── tests/
│   ├── test_security.py   # 20 security tests (injection, SSRF, etc.)
│   └── test_ingest.py     # Chunking and deduplication tests
├── data/
│   └── documents/         # Local document files
├── reports/figures/       # Evaluation plots
├── .env.example           # Template — copy to .env and add API key
├── requirements.txt
└── Makefile
```

---

## Setup & Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/rag-news-assistant.git
cd rag-news-assistant

# 2. Virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Configure API key (REQUIRED)
cp .env.example .env
# Edit .env and add your Anthropic API key
# Get a free key at: console.anthropic.com

# 5. Ingest sample documents
PYTHONPATH=. python src/ingest.py

# 6. Launch demo
PYTHONPATH=. python src/demo.py
# Open http://127.0.0.1:7860

# 7. Run tests
PYTHONPATH=. pytest tests/ -v
```

---

## Visualisations

### RAG Evaluation — Relevancy & Faithfulness
![RAG Evaluation](reports/figures/rag_evaluation.png)

### Demo Screenshot
![Demo](reports/figures/demo_screenshot.png)

---

## What I Learned

- **Chunking strategy matters enormously** — chunk size 500 with 100-char overlap outperformed both 200 and 1000 in retrieval precision
- **Security in RAG is underrated** — prompt injection via malicious documents is a real threat; delimiter-based prompt construction mitigates it
- **Faithfulness ≠ relevancy** — a model can give a relevant-sounding answer that isn't supported by the retrieved context (hallucination)
- **Deduplication by content hash** prevents the same article being re-ingested and inflating retrieval results

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
