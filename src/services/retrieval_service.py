import asyncio
import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import settings
from src.constants import DEFAULT_COLLECTION_NAME
from src.services.hybrid_retrieval import (
    bm25_top_n,
    build_bm25_index,
    reciprocal_rank_fusion,
    select_top_k_by_score,
)
from src.utils.singleton import Singleton

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
RERANK_CANDIDATE_POOL = 15


class RetrievalService(metaclass=Singleton):
    """
    Read-oriented access to the Chroma vector store populated by ingestion.

    Hybrid retrieval pipeline: vector similarity search and BM25 keyword search each
    produce a candidate pool, fused via Reciprocal Rank Fusion, then reordered by a local
    cross-encoder reranker before the top-k is returned. BM25 has no incremental-update API,
    so its in-memory index is rebuilt wholesale by `refresh()` rather than updated
    per-document; ingestion is responsible for calling `refresh()` once new chunks are added.
    """

    def __init__(self) -> None:
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_PATH,
            embedding_function=self.embedding_function,
            collection_name=DEFAULT_COLLECTION_NAME,
        )
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device="cpu")
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_documents: List[Document] = []
        self.refresh()

    def refresh(self) -> None:
        """
        Rebuild the in-memory BM25 index from whatever is currently persisted in Chroma.

        Note: this reads via a separate Chroma/PersistentClient instance than the one
        DocumentProcessor writes through (same persist path/collection name) - both work off
        Chroma's durable on-disk writes, so this sees data written by any process, but keep in
        mind having multiple concurrently-open PersistentClients on the same path is a known
        Chroma footgun (SQLite locking), pre-existing in this codebase and not addressed here.
        """
        try:
            data = self.vector_store.get(include=["documents", "metadatas"])
            documents = [
                Document(page_content=text, metadata=metadata or {}, id=doc_id)
                for doc_id, text, metadata in zip(
                    data["ids"], data["documents"], data["metadatas"]
                )
            ]
            self._bm25_documents = documents
            self._bm25_index = build_bm25_index(documents)
            logger.info("BM25 index refreshed with %d chunks", len(documents))
        except Exception as error:
            logger.error("Error while refreshing BM25 index: %s", str(error), exc_info=True)
            raise error

    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        try:
            vector_docs = await self.vector_store.asimilarity_search(
                query, k=VECTOR_CANDIDATE_K
            )
            bm25_docs = await asyncio.to_thread(
                bm25_top_n, self._bm25_index, self._bm25_documents, query, BM25_CANDIDATE_K
            )
            fused = reciprocal_rank_fusion([vector_docs, bm25_docs])[:RERANK_CANDIDATE_POOL]
            if not fused:
                return []

            scores = await asyncio.to_thread(
                self.cross_encoder.predict, [(query, doc.page_content) for doc in fused]
            )
            return select_top_k_by_score(fused, scores, k)
        except Exception as error:
            logger.error(
                "Error while retrieving chunks for query %r: %s",
                query,
                str(error),
                exc_info=True,
            )
            raise error


# Expose a singleton instance for standard imports
retrieval_service = RetrievalService()
