import logging
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.constants import DEFAULT_COLLECTION_NAME
from src.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class RetrievalService(metaclass=Singleton):
    """
    Read-oriented access to the Chroma vector store populated by ingestion.

    Kept as its own service (rather than inlined in chat_service) so that swapping in
    hybrid retrieval + reranking later doesn't require touching the chat service.
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

    async def retrieve(self, query: str, k: int = 5) -> List[Document]:
        try:
            return await self.vector_store.asimilarity_search(query, k=k)
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
