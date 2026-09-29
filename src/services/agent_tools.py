"""
Tools available to the agent loop (`agent_graph.py`).

Each tool is a plain async function returning `(llm_facing_text, retrieved_documents)`. The agent
node binds `StructuredTool` wrappers (built here) to the model purely so it can generate valid
tool-call schemas/arguments; the graph's own `tools` node (in `agent_graph.py`) calls these
functions directly with the parsed arguments instead of going through LangChain's generic
tool-execution machinery, because it needs both the LLM-facing string *and* the raw
`List[Document]` back (for citations/state), and hand-rolling that one small dispatch is simpler
than fighting LangChain's `content_and_artifact` tool-message plumbing for three tools.
"""
import asyncio
import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langfuse import get_client
from pydantic import BaseModel, Field

from src.services.calculator import safe_eval
from src.services.rag_formatting import build_context
from src.services.retrieval_service import retrieval_service
from src.utils.observability import observe_child
from src.utils.timing import log_duration

logger = logging.getLogger(__name__)
langfuse = get_client()

RETRIEVE_K = 5


class SearchFilingsInput(BaseModel):
    query: str = Field(description="A specific, self-contained search query over the ingested "
                        "SEC filings, e.g. 'total revenue fiscal year 2023'.")
    tickers: Optional[List[str]] = Field(
        default=None,
        description="US stock ticker(s) to scope the search to, if known (e.g. from the "
        "question). Omit or leave null to search across every ingested company.",
    )


class CompareCompaniesInput(BaseModel):
    tickers: List[str] = Field(
        description="Two or more US stock tickers to compare, e.g. ['AAPL', 'MSFT']."
    )
    aspect: str = Field(
        description="The specific financial aspect to compare, e.g. 'total revenue', "
        "'revenue growth', 'net income', 'total assets'."
    )


class CalculateInput(BaseModel):
    expression: str = Field(
        description="A numeric arithmetic expression using only numbers, + - * / % ** and "
        "parentheses, e.g. '(123.4 - 100.2) / 100.2 * 100' for a percent change. No variables, "
        "function calls, or units."
    )


@observe_child(as_type="tool", capture_output=False)
async def search_filings(
    query: str, tickers: Optional[List[str]] = None
) -> Tuple[str, List[Document]]:
    """Search the ingested SEC filings for a specific fact. Use for single-company questions, or
    to gather one piece of evidence toward a larger multi-part question.

    Args:
        query: A specific, self-contained search query.
        tickers: US stock ticker(s) to scope the search to, or `None` to search everything
            ingested.

    Returns:
        A tuple of (LLM-facing formatted context string, the raw retrieved `Document`s).
    """
    docs = await retrieval_service.retrieve(query, k=RETRIEVE_K, tickers=tickers)
    if not docs:
        scope = f" for {', '.join(tickers)}" if tickers else ""
        langfuse.update_current_span(output={"doc_count": 0})
        return f"No relevant filing chunks found{scope} for query: {query!r}", []
    langfuse.update_current_span(
        output={
            "doc_count": len(docs),
            "tickers": sorted({d.metadata.get("ticker") for d in docs if d.metadata.get("ticker")}),
        }
    )
    return build_context(docs), docs


@observe_child(as_type="tool", capture_output=False)
async def compare_companies(tickers: List[str], aspect: str) -> Tuple[str, List[Document]]:
    """Gather evidence to compare two or more companies on a specific financial aspect. Runs a
    separate scoped search per ticker in parallel and returns the evidence grouped by company, so
    it is never accidentally attributed to the wrong one.

    Args:
        tickers: Two or more US stock tickers to compare.
        aspect: The specific financial aspect to compare (e.g. "total revenue").

    Returns:
        A tuple of (LLM-facing side-by-side formatted context string grouped by ticker, the raw
        retrieved `Document`s across all tickers, flattened).
    """
    with log_duration(logger, f"compare_companies fan-out ({len(tickers)} ticker(s))"):
        per_ticker_docs = await asyncio.gather(
            *[
                retrieval_service.retrieve(f"{aspect} {ticker}", k=RETRIEVE_K, tickers=[ticker])
                for ticker in tickers
            ]
        )

    sections = []
    all_docs: List[Document] = []
    for ticker, docs in zip(tickers, per_ticker_docs):
        all_docs.extend(docs)
        section_body = build_context(docs) if docs else "No relevant filing chunks found."
        sections.append(f"## {ticker}\n\n{section_body}")

    langfuse.update_current_span(
        output={"doc_counts_by_ticker": {t: len(d) for t, d in zip(tickers, per_ticker_docs)}}
    )
    return "\n\n---\n\n".join(sections), all_docs


@observe_child(as_type="tool")
async def calculate(expression: str) -> Tuple[str, List[Document]]:
    """Evaluate a numeric arithmetic expression. Use this instead of doing arithmetic mentally,
    e.g. to compute a percent change or ratio from two figures already retrieved.

    Args:
        expression: An arithmetic expression using only numbers, `+ - * / % **`, and parentheses.

    Returns:
        A tuple of (LLM-facing result or error string, empty document list - this tool never
        retrieves evidence).
    """
    try:
        result = safe_eval(expression)
        return f"{expression} = {result}", []
    except ZeroDivisionError:
        return f"Error evaluating {expression!r}: division by zero.", []
    except (ValueError, SyntaxError, TypeError) as error:
        logger.debug("Rejected calculator expression %r: %s", expression, error)
        return (
            f"Error evaluating {expression!r}: unsupported or invalid expression. Only numbers, "
            "+ - * / % ** and parentheses are supported.",
            [],
        )


def build_agent_tool_schemas() -> List[StructuredTool]:
    """
    Build the `StructuredTool` wrappers bound to the model via `bind_tools`, purely so the model
    can see each tool's name/description/argument schema and produce valid tool-call requests.
    The graph's `tools` node dispatches by name back to the plain async functions above rather
    than invoking these wrappers - see module docstring.

    Returns:
        `[search_filings, compare_companies, calculate]` as `StructuredTool` schema objects, in
        that order.
    """
    return [
        StructuredTool.from_function(
            coroutine=search_filings,
            name="search_filings",
            description=search_filings.__doc__.splitlines()[0],
            args_schema=SearchFilingsInput,
        ),
        StructuredTool.from_function(
            coroutine=compare_companies,
            name="compare_companies",
            description=compare_companies.__doc__.splitlines()[0],
            args_schema=CompareCompaniesInput,
        ),
        StructuredTool.from_function(
            coroutine=calculate,
            name="calculate",
            description=calculate.__doc__.splitlines()[0],
            args_schema=CalculateInput,
        ),
    ]


# name -> underlying async function, for the graph's `tools` node to dispatch tool_calls by name.
TOOL_FUNCTIONS = {
    "search_filings": search_filings,
    "compare_companies": compare_companies,
    "calculate": calculate,
}
