import datetime
import logging
from typing import Optional

import chromadb

from src.config import settings
from src.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class ChromaDbService(metaclass=Singleton):
    def __init__(self) -> None:
        self.config = settings
        self.client = chromadb.PersistentClient(path=self.config.CHROMA_PERSIST_PATH)

    def create_collection(
        self, collection_name: str, description: Optional[str] = None
    ) -> bool:
        try:
            self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": description,
                    "created": str(datetime.datetime.now()),
                },
            )
            return True
        except Exception as error:
            logger.error(
                "Error while creating collection %s: %s",
                collection_name,
                str(error),
                exc_info=True,
            )
            raise error

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        try:
            collection = self.client.get_collection(collection_name)
            return collection
        except Exception as error:
            logger.error(
                "Error while getting collection %s: %s",
                collection_name,
                str(error),
                exc_info=True,
            )
            raise error

    def delete_collection(self, collection_name: str) -> None:
        try:
            logger.debug("Deleting collection %s", collection_name)
            self.client.delete_collection(collection_name)
        except Exception as error:
            logger.error(
                "Error while deleting collection %s: %s",
                collection_name,
                str(error),
            )
            raise error
