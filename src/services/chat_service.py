"""
Chat Service to process user messages
"""
import logging
import asyncio
from typing import cast

from src.models.chat_models import UserMessage, SymbolExtractionResponse
from src.config import settings
from src.core.sec_filings_downloader import SecFilingsDownloader

from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import get_client, observe

logger = logging.getLogger(__name__)
langfuse = get_client()
sec_files_downloader = SecFilingsDownloader(
    company_name="Payne Enterprise",
    mail="surigd@gmail.com",
    path_to_download="sec_files",
)
@observe()
async def process_user_message(data: UserMessage) -> SymbolExtractionResponse:
    message_data = data.user_message
    symbol_extractor_prompt = langfuse.get_prompt("dev/symbol-extractor")
    prompt = symbol_extractor_prompt.compile(user_message=message_data)
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=settings.GOOGLE_API_KEY
    )
    structured_model = model.with_structured_output(
        schema=SymbolExtractionResponse, method="json_schema"
    )
    response = cast(SymbolExtractionResponse, await structured_model.ainvoke(prompt))
    logger.debug("Response : %s", response.model_dump_json())
    downloaded_files = await asyncio.to_thread(
        sec_files_downloader.download_filings,
        tickers=response.symbol,
        limit=5,
        before=response.to_date,
        after=response.from_date,
    )

    logger.debug("Downloaded %d filing(s): %s", len(downloaded_files), downloaded_files)
    return response