"""
Pure retrieval-fusion logic: BM25 scoring, Reciprocal Rank Fusion, and rerank-score selection.

Deliberately has no dependency on Chroma, HuggingFaceEmbeddings, or CrossEncoder, so it can be
unit-tested with plain fixture Documents (no real model loads) and imported without triggering
RetrievalService's module-level singleton construction.
"""
import hashlib
import logging
import re
from typing import List, Optional, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """
    Split text into lowercased word tokens for BM25 indexing/querying.

    Args:
        text: Raw text to tokenize (a chunk's page_content, or a query string).

    Returns:
        List of lowercased `\\w+` tokens, in order, with all punctuation/whitespace dropped.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(documents: Sequence[Document]) -> Optional[BM25Okapi]:
    """
    Build a BM25 index over a fixed corpus of documents.

    The returned index is positionally tied to `documents`: `bm25_top_n` must be called with
    the same document list (or a list produced the same way) to get correct results back.

    Args:
        documents: The corpus to index, in the order BM25 should treat as their identity.

    Returns:
        A `BM25Okapi` index over `documents`' tokenized page_content, or `None` if `documents`
        is empty (`BM25Okapi` errors on an empty corpus, so callers must check for `None`).
    """
    if not documents:
        return None
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    return BM25Okapi(tokenized_corpus)


def bm25_top_n(
    index: Optional[BM25Okapi], documents: Sequence[Document], query: str, n: int
) -> List[Document]:
    """
    Return the top-n documents for a query from a pre-built BM25 index.

    Args:
        index: A `BM25Okapi` index built from `documents` via `build_bm25_index`, or `None`.
        documents: The same document list `index` was built from (positionally tied to it).
        query: The raw query text (tokenized internally the same way as the indexed corpus).
        n: Maximum number of documents to return.

    Returns:
        Up to `n` documents from `documents`, ranked by BM25 score against `query`, highest
        first. Empty list if `index` is `None` or `documents` is empty.
    """
    if index is None or not documents:
        return []
    return index.get_top_n(tokenize(query), list(documents), n=n)


def _content_fallback_id(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _dedup_key(doc: Document) -> str:
    return doc.id or _content_fallback_id(doc)


def dedup_documents(documents: Sequence[Document]) -> List[Document]:
    """
    Deduplicate documents by identity (`Document.id`, falling back to a content hash), keeping
    the first occurrence of each. Used wherever documents accumulated across multiple separate
    retrieval calls need deduping before being used as context/citations, using the same identity
    rule as `reciprocal_rank_fusion` below.

    Args:
        documents: Documents to dedupe, in the order to prefer (first occurrence wins).

    Returns:
        `documents` with duplicates removed, original order preserved.
    """
    seen = set()
    deduped = []
    for doc in documents:
        key = _dedup_key(doc)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]], k: int = 60
) -> List[Document]:
    """
    Fuse multiple ranked document lists into one ranking via Reciprocal Rank Fusion.

    Each document's fused score is the sum of `1 / (k + rank)` over every ranked list it
    appears in (1-indexed rank), so documents ranked highly in multiple lists float to the top.
    Documents are deduped across lists by `Document.id` (falling back to a content hash when
    `id` is falsy, so two `id=None` documents with identical content are still treated as the
    same document).

    Args:
        ranked_lists: One or more ranked document lists (e.g. vector search results, BM25
            results), each already ordered best-first.
        k: RRF smoothing constant (standard default 60, per Cormack et al.) — larger values
            flatten the influence of rank position.

    Returns:
        The union of all input documents (deduped), ordered by descending fused score.
    """
    scores: dict = {}
    doc_by_key: dict = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = _dedup_key(doc)
            doc_by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [doc_by_key[key] for key in ordered_keys]


def select_top_k_by_score(
    docs: Sequence[Document],
    scores: Sequence[float],
    k: int,
    score_key: str = "rerank_score",
) -> List[Document]:
    """
    Sort documents by an external score (e.g. cross-encoder output) and take the top k.

    Args:
        docs: Documents to score/select from.
        scores: Scores parallel to `docs` (`scores[i]` is `docs[i]`'s score).
        k: Maximum number of documents to return.
        score_key: Metadata key the score is stashed under on the returned copies.

    Returns:
        Up to `k` fresh `Document` copies (originals are never mutated), sorted by descending
        score, each with `metadata[score_key]` set to its score as a `float`.
    """
    paired = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    top = []
    for doc, score in paired[:k]:
        top.append(
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, score_key: float(score)},
                id=doc.id,
            )
        )
    return top


def build_ticker_filter(tickers: Optional[Sequence[str]]) -> Optional[dict]:
    """
    Build a Chroma metadata `where` filter that scopes a search to specific ticker(s).

    Args:
        tickers: Ticker symbols to filter on (any case), or `None`/empty to filter nothing.

    Returns:
        `None` if `tickers` is falsy (no filtering should be applied). Otherwise a Chroma
        `where` filter matching chunks whose `metadata["ticker"]` (always uppercased at
        ingestion/query time) equals the single given ticker (`{"ticker": {"$eq": TICKER}}`),
        or is one of the given tickers when more than one is provided
        (`{"ticker": {"$in": [TICKER, ...]}}`).
    """
    if not tickers:
        return None
    normalized = [t.upper() for t in tickers]
    if len(normalized) == 1:
        return {"ticker": {"$eq": normalized[0]}}
    return {"ticker": {"$in": normalized}}


def filter_documents_by_tickers(
    documents: Sequence[Document], tickers: Optional[Sequence[str]]
) -> List[Document]:
    """
    Filter an in-memory document list down to just the given ticker(s).

    Used to scope the BM25 candidate pool the same way `build_ticker_filter` scopes the Chroma
    vector search, since BM25 has no query-time filtering of its own — a fresh index must be
    built from the filtered subset (see `build_bm25_index`).

    Args:
        documents: Documents to filter, each expected to carry `metadata["ticker"]` when it
            belongs to a specific company (chunks without a `ticker` key are excluded whenever
            a filter is applied).
        tickers: Ticker symbols to keep (any case), matched case-insensitively. `None`/empty
            means "no filter" — `documents` is returned unchanged.

    Returns:
        The subset of `documents` whose `metadata["ticker"]` case-insensitively matches one of
        `tickers`, in their original order. Returns `documents` unchanged (same list) if
        `tickers` is falsy.
    """
    if not tickers:
        return list(documents)
    normalized = {t.upper() for t in tickers}
    return [
        doc
        for doc in documents
        if str(doc.metadata.get("ticker", "")).upper() in normalized
    ]
