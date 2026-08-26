"""
Shared context/citation formatting for retrieved chunks.

Used by both the single-shot chat pipeline (`chat_service.py`) and the agent pipeline
(`agent_graph.py`) to build LLM-facing context strings and response-facing source citations
identically, instead of maintaining separate copies of the same logic.
"""
from typing import List

from langchain_core.documents import Document

from src.models.chat_models import SourceReference


def build_context(chunks: List[Document]) -> str:
    """
    Build the LLM-facing context string from retrieved chunks.

    Args:
        chunks: Retrieved chunks (best-first) to ground the answer in.

    Returns:
        A single string with one "Source: ... / Content: ..." section per chunk, separated by
        `---`, suitable for interpolation into a QA prompt.
    """
    sections = []
    for chunk in chunks:
        metadata = chunk.metadata
        sections.append(
            "Source: {company} {doc_type} ({filing_date})\n\nContent:\n{content}".format(
                company=metadata.get("company_name", "Unknown company"),
                doc_type=metadata.get("doc_type", "filing"),
                filing_date=metadata.get("filing_date", "unknown date"),
                content=chunk.page_content,
            )
        )
    return "\n\n---\n\n".join(sections)


def to_sources(chunks: List[Document]) -> List[SourceReference]:
    """
    Build response-facing source citations directly from retrieved-chunk metadata.

    Deliberately never reads from the LLM's answer, so citations can't be hallucinated. Chunks
    are deduped by (company_name, doc_type, filing_date); since `chunks` is expected best-first
    (e.g. reranker order), the first (highest-scoring) chunk per source wins.

    Args:
        chunks: Retrieved chunks (best-first), the same list passed to `build_context`.

    Returns:
        One `SourceReference` per distinct (company_name, doc_type, filing_date), in `chunks`'
        order, each carrying that chunk's `relevance_score` (from `metadata["rerank_score"]`, when
        present).
    """
    seen = set()
    sources: List[SourceReference] = []
    for chunk in chunks:
        metadata = chunk.metadata
        key = (
            metadata.get("company_name"),
            metadata.get("doc_type"),
            metadata.get("filing_date"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceReference(
                company_name=metadata.get("company_name"),
                doc_type=metadata.get("doc_type"),
                filing_date=metadata.get("filing_date"),
                cik=metadata.get("cik"),
                # Retrieval-time metadata from the local cross-encoder reranker, not
                # ingestion-time metadata like the fields above - still never LLM-generated.
                relevance_score=metadata.get("rerank_score"),
            )
        )
    return sources
