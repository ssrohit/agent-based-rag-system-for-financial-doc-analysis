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
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(documents: Sequence[Document]) -> Optional[BM25Okapi]:
    if not documents:
        return None
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    return BM25Okapi(tokenized_corpus)


def bm25_top_n(
    index: Optional[BM25Okapi], documents: Sequence[Document], query: str, n: int
) -> List[Document]:
    if index is None or not documents:
        return []
    return index.get_top_n(tokenize(query), list(documents), n=n)


def _content_fallback_id(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _dedup_key(doc: Document) -> str:
    return doc.id or _content_fallback_id(doc)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]], k: int = 60
) -> List[Document]:
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
