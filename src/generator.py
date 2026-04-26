"""
generator.py
------------
Generation stage of RAG — takes retrieved context and generates an answer.

Uses Groq API (free) with llama-3.1-8b-instant model.
Groq is extremely fast (tokens per second >> OpenAI) and has a generous free tier.

Get your free API key at: console.groq.com
Security: API key loaded from .env, never hardcoded.
"""

import os
from typing import List, Dict
from loguru import logger
from dotenv import load_dotenv
from groq import Groq

from src.security import sanitise_query, build_safe_prompt
from src.retriever import retrieve_and_format

load_dotenv()

MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_RESPONSE", 1000))


def get_client() -> Groq:
    """
    Create Groq client with API key from environment.
    Raises clear error if key is missing.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "1. Go to console.groq.com and sign up (free, no card)\n"
            "2. Click API Keys -> Create API Key\n"
            "3. Copy the key and paste it in your .env file as:\n"
            "   GROQ_API_KEY=gsk_...\n"
        )
    return Groq(api_key=api_key)


def generate_answer(
    query: str,
    context: str,
    sources: List[str],
    model: str = "llama-3.1-8b-instant",
) -> Dict:
    """
    Generate an answer using Groq with the retrieved context.

    Returns a dict with:
      - answer: the generated text
      - sources: list of cited sources
      - model: which model was used
      - tokens_used: for tracking
    """
    query = sanitise_query(query)

    if not context:
        return {
            "answer": "I couldn't find any relevant articles to answer your question. "
                      "Please add more documents or try a different question.",
            "sources": [],
            "model": model,
            "tokens_used": 0,
        }

    prompt = build_safe_prompt(query, context)
    client = get_client()

    logger.info(f"Generating answer for: '{query[:50]}' using {model}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual news research assistant. "
                    "Only answer based on the provided context. "
                    "If the answer is not in the context, say so clearly. "
                    "Always cite which article your answer comes from."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.1,   # low temperature = more factual, less creative
    )

    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    logger.info(f"Generated {len(answer)} char answer using {tokens_used} tokens")

    return {
        "answer": answer,
        "sources": sources,
        "model": model,
        "tokens_used": tokens_used,
    }


def rag_pipeline(query: str) -> Dict:
    """
    Full RAG pipeline: query -> retrieve -> generate -> return.
    This is the main entry point for the application.
    """
    logger.info(f"RAG pipeline started for: '{query[:50]}'")

    context, sources, chunks = retrieve_and_format(query)
    result = generate_answer(query, context, sources)

    result["num_chunks_retrieved"] = len(chunks)
    result["query"] = query
    result["context"] = context

    if chunks:
        result["avg_relevance_score"] = round(
            sum(c["similarity_score"] for c in chunks) / len(chunks), 3
        )
    else:
        result["avg_relevance_score"] = 0.0

    return result


if __name__ == "__main__":
    test_queries = [
        "What are the main requirements of the EU AI Act?",
        "What efficiency have solar panels reached?",
        "What is the fine for violating the AI Act?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {query}")
        try:
            result = rag_pipeline(query)
            print(f"A: {result['answer'][:300]}...")
            print(f"Sources: {result['sources']}")
            print(f"Tokens used: {result['tokens_used']}")
        except EnvironmentError as e:
            print(f"Setup needed:\n{e}")
            break
