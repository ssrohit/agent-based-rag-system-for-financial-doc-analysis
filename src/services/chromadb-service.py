import chromadb
import datetime
from logging import Logger

from src.utils.singleton import Singleton

logger = Logger()


class ChromaDbService(metaclass=Singleton):
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")

    def create_collection(self, collection_name: str, description: str = None):
        try:
            self.client.create_collection(
                name=collection_name,
                metadata={description: description, created: str(datetime.now())},
            )
            return True
        except Exception as error:
            logger.error(
                "Error while creating collection %s: %s",
                collection_name,
                str(e),
                exc_info=True,
            )
            raise error

    def get_collection(self, collection_name: str):
        try:
            collection = self.client.get_collection(collection_name)
            return collection
        except Exception as error:
            logger.error(
                "Error while getting collection %s: %s",
                collection_name,
                str(e),
                exc_info=True,
            )
            raise error
            
    def delete_collection(self, collection_name: str):
        try:
            logger.debug("Deleting collection %s", collection_name)
            self.client.delete_collection(collection_name)
        except Exception as error:
            logger.error("Error while deleting collection %s: %s",collection_name, str(error))
            raise error
