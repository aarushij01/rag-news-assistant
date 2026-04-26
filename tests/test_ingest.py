"""Tests for ingest.py"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingest import chunk_document


def test_chunk_document_basic():
    doc = {
        "title": "Test Article",
        "text": "This is a test article. " * 50,
        "url": "file://test.txt",
        "source": "test"
    }
    chunks = chunk_document(doc)
    assert len(chunks) > 0
    assert all("text" in c for c in chunks)
    assert all("metadata" in c for c in chunks)


def test_chunk_document_metadata():
    doc = {
        "title": "My Article",
        "text": "Content here. " * 100,
        "url": "https://bbc.com/test",
        "source": "test"
    }
    chunks = chunk_document(doc)
    for chunk in chunks:
        assert chunk["metadata"]["title"] == "My Article"
        assert chunk["metadata"]["url"] == "https://bbc.com/test"
        assert "content_hash" in chunk["metadata"]
        assert "chunk_index" in chunk["metadata"]


def test_chunk_document_empty_text():
    doc = {"title": "Empty", "text": "", "url": "file://x", "source": "test"}
    chunks = chunk_document(doc)
    assert chunks == []


def test_chunk_deduplication_by_hash():
    doc = {"title": "Test", "text": "Same content. " * 50, "url": "file://x", "source": "test"}
    chunks1 = chunk_document(doc)
    chunks2 = chunk_document(doc)
    hashes1 = {c["metadata"]["content_hash"] for c in chunks1}
    hashes2 = {c["metadata"]["content_hash"] for c in chunks2}
    assert hashes1 == hashes2
