import logging
import asyncio
from pathlib import Path
from typing import Optional


from src.models.chat_models import UserMessage
from src.models.ingestion_models import SymbolExtraction
from src.constants import DEFAULT_COLLECTION_NAME
from src.core.sec_filings_downloader import SecFilingsDownloader
from src.document_processor.parse_documents import DocumentProcessor
from src.services.retrieval_service import retrieval_service
from langfuse import get_client
from langfuse import observe
from src.services.llm_service import llm_service
from src.utils.timing import log_duration


langfuse = get_client()
logger = logging.getLogger(__name__)
sec_files_downloader = SecFilingsDownloader(
    company_name="Payne Enterprise",
    mail="surigd@gmail.com",
    path_to_download="sec_files",
)


def _ticker_from_downloaded_file(file_path: Path) -> Optional[str]:
    """SecFilingsDownloader names files '{TICKER}_{FORM_TYPE}_{ACCESSION}_{filename}';
    recover the ticker for this specific file rather than guessing from the (possibly
    multi-ticker) LLM-extracted symbol list, so multi-company ingestion runs don't mislabel
    every chunk with the same ticker. Uppercased so it matches the case-normalized filter
    query-time retrieval builds in `hybrid_retrieval.build_ticker_filter` (Level 4).

    Args:
        file_path: Path to a file produced by `SecFilingsDownloader.download_filings`.

    Returns:
        The uppercased ticker prefix of the filename, or `None` if the filename has no
        recognizable ticker prefix.
    """
    parts = file_path.name.split("_", 1)
    return parts[0].upper() if parts and parts[0] else None

@observe()
async def ingest_data(userQuery: UserMessage):
    message_data = userQuery.user_message
    symbol_extractor_prompt = langfuse.get_prompt("dev/symbol-extractor")
    prompt = symbol_extractor_prompt.compile(user_message=message_data)
    
    response = await llm_service.ainvoke_structured(
        prompt=prompt,
        schema=SymbolExtraction,
        langfuse_prompt=symbol_extractor_prompt
    )
    logger.debug("Response : %s", response.model_dump_json())
    with log_duration(logger, f"SEC filings download (symbol={response.symbol})"):
        downloaded_files = await asyncio.to_thread(
            sec_files_downloader.download_filings,
            tickers=response.symbol,
            limit=5,
            before=response.to_date,
            after=response.from_date,
        )
    if not downloaded_files:
        logger.warning("No files downloaded for symbol %s. Skipping document processing.", response.symbol)
        return {"status": "no_files_downloaded", "symbol": response.symbol}
    doc_processor = DocumentProcessor(DEFAULT_COLLECTION_NAME)
    with log_duration(logger, f"Document processing for {len(downloaded_files)} file(s)"):
        for file_path in downloaded_files:
            ticker = _ticker_from_downloaded_file(file_path)
            await asyncio.to_thread(doc_processor.extract_data, str(file_path), ticker=ticker)
    with log_duration(logger, "BM25 refresh after ingestion"):
        await asyncio.to_thread(retrieval_service.refresh)
    logger.debug("Downloaded %d filing(s): %s", len(downloaded_files), downloaded_files)
    return {
        "status": "success",
        "symbol": response.symbol,
        "downloaded_files": [str(f) for f in downloaded_files]
    }
