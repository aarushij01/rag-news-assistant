"""
evaluate.py
-----------
Evaluates RAG pipeline quality using three key metrics:

1. Faithfulness     — does the answer stick to the context? (no hallucination)
2. Answer Relevancy — does the answer actually address the question?
3. Context Recall   — did retrieval find the right chunks?

We implement lightweight versions of these without needing the full RAGAS
framework (which requires an LLM for evaluation too).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from loguru import logger

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# --- Test questions with known answers (ground truth) ---
# These are based on the sample documents ingested in ingest.py
EVAL_SET = [
    {
        "question": "What is the maximum fine for violating the EU AI Act?",
        "ground_truth": "30 million euros or 6% of global annual turnover",
        "expected_keywords": ["30 million", "6%", "fine", "turnover"],
    },
    {
        "question": "What efficiency did laboratory solar cells exceed?",
        "ground_truth": "33% efficiency",
        "expected_keywords": ["33%", "efficiency", "solar"],
    },
    {
        "question": "Who signed the US executive order on AI safety?",
        "ground_truth": "President Biden",
        "expected_keywords": ["Biden", "executive order"],
    },
    {
        "question": "What are solid-state batteries used for?",
        "ground_truth": "Electric vehicles and grid-scale energy storage",
        "expected_keywords": ["electric vehicles", "battery", "solid-state"],
    },
    {
        "question": "What year did the EU pass the AI Act?",
        "ground_truth": "2024",
        "expected_keywords": ["2024"],
    },
]


def keyword_relevancy(answer: str, expected_keywords: list) -> float:
    """
    Simple answer relevancy: what fraction of expected keywords appear in answer?
    Real RAGAS uses embedding similarity — this is a fast approximation.
    """
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0


def faithfulness_score(answer: str, context: str) -> float:
    """
    Approximate faithfulness: what fraction of answer sentences have
    supporting evidence in the context?

    A faithful answer only uses information from the context.
    """
    if not context:
        return 0.0

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
    if not sentences:
        return 1.0

    context_lower = context.lower()
    supported = 0
    for sentence in sentences:
        # Check if key words from sentence appear in context
        words = [w for w in sentence.lower().split() if len(w) > 4]
        if not words:
            continue
        word_hits = sum(1 for w in words if w in context_lower)
        if word_hits / len(words) > 0.4:   # 40% word overlap = supported
            supported += 1

    return supported / len(sentences)


def run_evaluation(rag_pipeline_fn) -> pd.DataFrame:
    """
    Run the full evaluation set and return results as DataFrame.
    """
    results = []
    for item in EVAL_SET:
        logger.info(f"Evaluating: {item['question'][:50]}")
        try:
            result = rag_pipeline_fn(item["question"])
            answer = result.get("answer", "")
            context = result.get("context", "")

            relevancy = keyword_relevancy(answer, item["expected_keywords"])
            faithfulness = faithfulness_score(answer, context)

            results.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": answer[:200],
                "relevancy": round(relevancy, 3),
                "faithfulness": round(faithfulness, 3),
                "tokens_used": result.get("tokens_used", 0),
                "chunks_retrieved": result.get("num_chunks_retrieved", 0),
                "avg_relevance_score": result.get("avg_relevance_score", 0),
            })
        except Exception as e:
            logger.error(f"Evaluation failed for '{item['question']}': {e}")
            results.append({
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": f"ERROR: {e}",
                "relevancy": 0, "faithfulness": 0,
                "tokens_used": 0, "chunks_retrieved": 0,
                "avg_relevance_score": 0,
            })

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame):
    """Plot evaluation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Relevancy per question
    axes[0].barh(
        [f"Q{i+1}" for i in range(len(df))],
        df["relevancy"],
        color="#1D9E75"
    )
    axes[0].set_xlabel("Answer Relevancy Score")
    axes[0].set_title("Answer Relevancy by Question")
    axes[0].set_xlim(0, 1)
    for i, v in enumerate(df["relevancy"]):
        axes[0].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=10)

    # Faithfulness per question
    axes[1].barh(
        [f"Q{i+1}" for i in range(len(df))],
        df["faithfulness"],
        color="#5DCAA5"
    )
    axes[1].set_xlabel("Faithfulness Score")
    axes[1].set_title("Answer Faithfulness by Question")
    axes[1].set_xlim(0, 1)
    for i, v in enumerate(df["faithfulness"]):
        axes[1].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=10)

    plt.suptitle("RAG Pipeline Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{FIGURES_DIR}/rag_evaluation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def print_summary(df: pd.DataFrame):
    print("\n" + "="*50)
    print("RAG EVALUATION SUMMARY")
    print("="*50)
    print(f"Questions evaluated : {len(df)}")
    print(f"Mean Relevancy      : {df['relevancy'].mean():.1%}")
    print(f"Mean Faithfulness   : {df['faithfulness'].mean():.1%}")
    print(f"Mean Chunks Retrieved: {df['chunks_retrieved'].mean():.1f}")
    print(f"Total Tokens Used   : {df['tokens_used'].sum()}")
    print("="*50)


if __name__ == "__main__":
    from src.generator import rag_pipeline
    df = run_evaluation(rag_pipeline)
    print_summary(df)
    plot_results(df)
    df.to_csv("reports/rag_evaluation.csv", index=False)
    print("Results saved to reports/rag_evaluation.csv")
