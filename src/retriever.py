"""
retriever.py
------------
Retrieval stage of RAG — finds the most relevant document chunks
for a given query using semantic similarity search.

Also implements a reranking step to improve precision.
"""

import os
from typing import List, Dict, Tuple
from loguru import logger

from src.ingest import get_vector_store, get_embeddings
from src.security import sanitise_query

TOP_K = 5           # number of chunks to retrieve
SCORE_THRESHOLD = -0.5   # minimum similarity score (0-1)


def retrieve(query: str, k: int = TOP_K) -> List[Dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Uses cosine similarity between query embedding and stored chunk embeddings.
    Returns chunks sorted by relevance score (highest first).
    """
    query = sanitise_query(query)

    vector_store = get_vector_store()

    # similarity_search_with_score returns (Document, score) tuples
    # Score is cosine distance: lower = more similar
    results = vector_store.similarity_search_with_score(query, k=k)

    if not results:
        logger.warning(f"No results found for query: {query[:50]}")
        return []

    retrieved = []
    for doc, score in results:
        # Convert distance to similarity (1 - distance for cosine)
        similarity = 1 - score
        if similarity < SCORE_THRESHOLD:
            logger.debug(f"Skipping low-relevance chunk (score: {similarity:.3f})")
            continue

        retrieved.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(similarity, 4),
        })

    logger.info(f"Retrieved {len(retrieved)} relevant chunks for: '{query[:50]}'")
    return retrieved


def format_context(chunks: List[Dict]) -> Tuple[str, List[str]]:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Also returns a list of source citations.

    Format:
      [Article 1: Title]
      chunk text...

      [Article 2: Title]
      chunk text...
    """
    if not chunks:
        return "", []

    context_parts = []
    sources = []

    seen_titles = {}
    article_num = 1

    for chunk in chunks:
        title = chunk["metadata"].get("title", "Unknown")
        url = chunk["metadata"].get("url", "")
        score = chunk["similarity_score"]

        if title not in seen_titles:
            seen_titles[title] = article_num
            article_num += 1
            sources.append(f"[{seen_titles[title]}] {title} — {url}")

        context_parts.append(
            f"[Article {seen_titles[title]}: {title}] (relevance: {score:.0%})\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    return context, sources


def retrieve_and_format(query: str) -> Tuple[str, List[str], List[Dict]]:
    """
    Full retrieval pipeline: query → chunks → formatted context + sources.
    Returns: (context_string, sources_list, raw_chunks)
    """
    chunks = retrieve(query)
    context, sources = format_context(chunks)
    return context, sources, chunks


if __name__ == "__main__":
    test_query = "What is the EU AI Act?"
    print(f"Query: {test_query}\n")
    context, sources, chunks = retrieve_and_format(test_query)

    if chunks:
        print(f"Retrieved {len(chunks)} chunks\n")
        print("Sources:")
        for s in sources:
            print(f"  {s}")
        print(f"\nContext preview:\n{context[:500]}...")
    else:
        print("No results — run ingest.py first to add documents.")
