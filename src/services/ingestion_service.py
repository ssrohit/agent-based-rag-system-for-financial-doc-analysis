import logging
import asyncio
from typing import cast

from models.chat_models import UserMessage
from models.ingestion_models import SymbolExtraction
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings
from src.core.sec_filings_downloader import SecFilingsDownloader
from src.document_processor.parse_documents import DocumentProcessor
from langfuse import get_client


langfuse = get_client()
logger = logging.getLogger(__name__)
sec_files_downloader = SecFilingsDownloader(
    company_name="Payne Enterprise",
    mail="surigd@gmail.com",
    path_to_download="sec_files",
)


async def ingest_data(userQuery: UserMessage):
    message_data = userQuery.user_message
    symbol_extractor_prompt = langfuse.get_prompt("dev/symbol-extractor")
    prompt = symbol_extractor_prompt.compile(user_message=message_data)
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=settings.GOOGLE_API_KEY
    )
    structured_model = model.with_structured_output(
        schema=SymbolExtraction, method="json_schema"
    )
    response = cast(SymbolExtraction, await structured_model.ainvoke(prompt))
    logger.debug("Response : %s", response.model_dump_json())
    downloaded_files = await asyncio.to_thread(
        sec_files_downloader.download_filings,
        tickers=response.symbol,
        limit=5,
        before=response.to_date,
        after=response.from_date,
    )
    doc_processor = DocumentProcessor("test_ingestion")
    for file_path in downloaded_files:
        await asyncio.to_thread(doc_processor.extract_data, str(file_path))
    logger.debug("Downloaded %d filing(s): %s", len(downloaded_files), downloaded_files)
