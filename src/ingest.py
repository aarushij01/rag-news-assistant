"""
ingest.py
---------
Document ingestion pipeline — the first stage of RAG.

Takes news articles (URLs or text files), chunks them intelligently,
embeds them using sentence-transformers, and stores in ChromaDB.

Chunking strategy matters a lot:
  - Too small: each chunk loses context
  - Too large: retrieval becomes imprecise
  - We use semantic chunking with overlap to balance both
"""

import os
import hashlib
import requests
from typing import List, Dict
from loguru import logger
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.security import validate_url, sanitise_document_text

# Chunking parameters — tuned for news articles
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks (preserves context at boundaries)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # fast, good quality, runs locally (no API needed)
CHROMA_DIR = "data/chroma_db"


def get_embeddings():
    """Load the sentence-transformer embedding model."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(embeddings=None):
    """Load or create the ChromaDB vector store."""
    if embeddings is None:
        embeddings = get_embeddings()
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="news_articles",
    )


def fetch_article_text(url: str) -> Dict:
    """
    Fetch article text from a URL.
    Uses newspaper3k for clean text extraction (removes ads, nav, etc.)
    Falls back to basic requests + BeautifulSoup if newspaper3k fails.
    """
    # Security check first — always validate before making any network request
    url = validate_url(url)

    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title or url
        logger.info(f"Fetched via newspaper3k: {title[:60]}")
        return {"title": title, "text": text, "url": url, "source": "newspaper3k"}

    except Exception as e:
        logger.warning(f"newspaper3k failed ({e}), falling back to requests")
        headers = {"User-Agent": "Mozilla/5.0 (research bot; contact: your@email.com)"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = " ".join(soup.get_text().split())
        title = soup.title.string if soup.title else url
        return {"title": title, "text": text, "url": url, "source": "beautifulsoup"}


def load_text_file(path: str) -> Dict:
    """Load a plain text or markdown file as a document."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    title = os.path.basename(path)
    return {"title": title, "text": text, "url": f"file://{path}", "source": "file"}


def chunk_document(doc: Dict) -> List[Dict]:
    """
    Split a document into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on:
      1. Paragraphs (\n\n) first
      2. Then sentences (. ! ?)
      3. Then words
      4. Then characters (last resort)
    This preserves meaning better than fixed-size splitting.
    """
    text = sanitise_document_text(doc["text"])
    if not text:
        logger.warning(f"Empty document after sanitisation: {doc['title']}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    chunks = splitter.split_text(text)

    result = []
    for i, chunk in enumerate(chunks):
        # Content hash = deduplication key (prevents same article being added twice)
        content_hash = hashlib.md5(chunk.encode()).hexdigest()
        result.append({
            "text": chunk,
            "metadata": {
                "title": doc["title"],
                "url": doc["url"],
                "source": doc["source"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_hash": content_hash,
            }
        })

    logger.info(f"'{doc['title'][:50]}' → {len(chunks)} chunks")
    return result


def ingest_documents(docs: List[Dict], vector_store=None) -> int:
    """
    Add a list of documents to the vector store.
    Skips duplicates based on content hash.
    Returns number of new chunks added.
    """
    if vector_store is None:
        vector_store = get_vector_store()

    # Get existing hashes to avoid duplicates
    try:
        existing = vector_store.get()
        existing_hashes = set(
            m.get("content_hash", "") for m in existing.get("metadatas", [])
        )
    except Exception:
        existing_hashes = set()

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        for chunk in chunks:
            if chunk["metadata"]["content_hash"] not in existing_hashes:
                all_chunks.append(chunk)
                existing_hashes.add(chunk["metadata"]["content_hash"])

    if not all_chunks:
        logger.info("No new chunks to add (all duplicates)")
        return 0

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [c["metadata"]["content_hash"] for c in all_chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.info(f"Added {len(all_chunks)} new chunks to vector store")
    return len(all_chunks)


def ingest_urls(urls: List[str]) -> int:
    """Convenience function: fetch + ingest a list of URLs."""
    docs = []
    for url in tqdm(urls, desc="Fetching articles"):
        try:
            doc = fetch_article_text(url)
            docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")

    return ingest_documents(docs)


# Sample news URLs for demo/testing
SAMPLE_URLS = [
    "https://www.bbc.com/news/technology-67881954",
    "https://www.bbc.com/news/science-environment-68143453",
]


if __name__ == "__main__":
    # Quick test with sample files instead of live URLs (more reliable)
    sample_docs = [
        {
            "title": "AI Regulation in 2024",
            "text": """
            Artificial intelligence regulation has become a major policy priority worldwide.
            The European Union passed the AI Act in 2024, creating the world's first
            comprehensive legal framework for AI systems. The Act classifies AI systems
            by risk level, with high-risk applications like medical diagnosis and credit
            scoring facing strict requirements. Companies must ensure transparency,
            human oversight, and data quality. Violations can result in fines of up
            to 30 million euros or 6% of global annual turnover.

            In the United States, President Biden signed an executive order on AI safety
            requiring federal agencies to develop AI governance frameworks. The order
            focused on safety testing, privacy protection, and preventing AI-enabled
            discrimination. Tech companies including Google, Microsoft, and Anthropic
            voluntarily committed to safety standards while formal legislation remains
            under debate in Congress.
            """,
            "url": "file://sample1.txt",
            "source": "file"
        },
        {
            "title": "Climate Technology Breakthroughs",
            "text": """
            Renewable energy technology achieved several milestones in 2024. Solar panel
            efficiency records were broken multiple times, with laboratory cells exceeding
            33% efficiency compared to the 20-22% typical in commercial panels. The cost
            of utility-scale solar dropped below 2 cents per kilowatt-hour in some markets,
            making it the cheapest electricity source in history.

            Battery storage technology also advanced rapidly. Solid-state batteries,
            which promise higher energy density and improved safety over lithium-ion,
            entered limited commercial production. Several automakers announced plans
            to use solid-state batteries in electric vehicles by 2026. Grid-scale
            iron-air batteries, which can store energy for days rather than hours,
            began deployment in the United States.
            """,
            "url": "file://sample2.txt",
            "source": "file"
        },
    ]

    print("Ingesting sample documents...")
    n = ingest_documents(sample_docs)
    print(f"Done — added {n} chunks to vector store at {CHROMA_DIR}")
