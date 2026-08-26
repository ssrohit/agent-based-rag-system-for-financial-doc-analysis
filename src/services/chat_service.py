"""
Chat Service to process user messages
"""
import logging

from langfuse import get_client, observe

from src.models.chat_models import (
    ChatResponse,
    FinancialAnswer,
    QueryEntities,
    UserMessage,
)
from src.services.llm_service import llm_service
from src.services.rag_formatting import build_context, to_sources
from src.services.retrieval_service import retrieval_service
from src.utils.timing import log_duration

logger = logging.getLogger(__name__)
langfuse = get_client()


async def _extract_query_entities(query: str) -> QueryEntities:
    """
    Extract which company (if any) a chat question is about, via a dedicated Langfuse-managed
    prompt ('dev/query-entity-extraction') and structured LLM output. Used to scope retrieval to
    that company's chunks - see `process_user_message`.

    Args:
        query: The user's natural-language question.

    Returns:
        A `QueryEntities` with `tickers` uppercased; `tickers` is an empty list when the
        question doesn't name or clearly imply a specific company.
    """
    entity_prompt = langfuse.get_prompt("dev/query-entity-extraction")
    prompt = entity_prompt.compile(question=query)
    result = await llm_service.ainvoke_structured(
        prompt=prompt,
        schema=QueryEntities,
        langfuse_prompt=entity_prompt,
    )
    return QueryEntities(tickers=[t.upper() for t in result.tickers])


@observe()
async def process_user_message(data: UserMessage) -> ChatResponse:
    """
    Answer a free-text financial question grounded in the ingested filings.

    Flow: extract which company (if any) the question is about, retrieve top-k chunks scoped to
    that company via hybrid retrieval, ground an LLM answer in those chunks, and return it with
    source citations built directly from chunk metadata (never LLM-generated).

    Args:
        data: The incoming user message wrapper (`data.user_message` is the raw question text).

    Returns:
        A `ChatResponse`. If nothing relevant is retrievable, `answer` is an explicit
        short-circuit message instead of an LLM-generated one: naming the extracted ticker(s)
        when the question named a company that has no ingested chunks, or a generic
        "nothing ingested yet" message when the question named no company at all.
    """
    query = data.user_message
    with log_duration(logger, f"process_user_message total (query={query!r})"):
        query_entities = await _extract_query_entities(query)
        tickers = query_entities.tickers or None
        chunks = await retrieval_service.retrieve(query, k=5, tickers=tickers)

        if not chunks:
            logger.warning(
                "No relevant chunks found for query: %s (tickers=%s)", query, tickers
            )
            if tickers:
                return ChatResponse(
                    answer=(
                        "I don't have any ingested filings for {tickers} yet. Try ingesting "
                        "their filings first.".format(tickers=", ".join(tickers))
                    )
                )
            return ChatResponse(
                answer=(
                    "I don't have any ingested filings that are relevant to this question yet. "
                    "Try ingesting the relevant company's filings first."
                )
            )

        context = build_context(chunks)
        qa_prompt = langfuse.get_prompt("dev/financial-qa")
        prompt = qa_prompt.compile(context=context, question=query)

        result = await llm_service.ainvoke_structured(
            prompt=prompt,
            schema=FinancialAnswer,
            langfuse_prompt=qa_prompt,
        )
        logger.debug("Response: %s", result.model_dump_json())

        return ChatResponse(
            answer=result.answer,
            key_metrics=result.key_metrics,
            caveats=result.caveats,
            sources=to_sources(chunks),
        )
