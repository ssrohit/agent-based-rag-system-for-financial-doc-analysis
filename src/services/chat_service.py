"""
Chat Service to process user messages
"""
import logging
from typing import List

from langchain_core.documents import Document
from langfuse import get_client, observe

from src.models.chat_models import ChatResponse, FinancialAnswer, SourceReference, UserMessage
from src.services.llm_service import llm_service
from src.services.retrieval_service import retrieval_service

logger = logging.getLogger(__name__)
langfuse = get_client()


def _build_context(chunks: List[Document]) -> str:
    sections = []
    for chunk in chunks:
        metadata = chunk.metadata
        sections.append(
            "Source: {company} {doc_type} ({filing_date})\n\nContent:\n{content}".format(
                company=metadata.get("company_name", "Unknown company"),
                doc_type=metadata.get("doc_type", "filing"),
                filing_date=metadata.get("filing_date", "unknown date"),
                content=chunk.page_content,
            )
        )
    return "\n\n---\n\n".join(sections)


def _to_sources(chunks: List[Document]) -> List[SourceReference]:
    seen = set()
    sources: List[SourceReference] = []
    for chunk in chunks:
        metadata = chunk.metadata
        key = (
            metadata.get("company_name"),
            metadata.get("doc_type"),
            metadata.get("filing_date"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceReference(
                company_name=metadata.get("company_name"),
                doc_type=metadata.get("doc_type"),
                filing_date=metadata.get("filing_date"),
                cik=metadata.get("cik"),
            )
        )
    return sources


@observe()
async def process_user_message(data: UserMessage) -> ChatResponse:
    query = data.user_message
    chunks = await retrieval_service.retrieve(query, k=5)

    if not chunks:
        logger.warning("No relevant chunks found for query: %s", query)
        return ChatResponse(
            answer=(
                "I don't have any ingested filings that are relevant to this question yet. "
                "Try ingesting the relevant company's filings first."
            )
        )

    context = _build_context(chunks)
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
        sources=_to_sources(chunks),
    )
