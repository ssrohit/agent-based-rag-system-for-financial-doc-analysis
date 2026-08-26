from typing import List, Optional

from pydantic import BaseModel, Field

from src.models.chat_models import ChatResponse


class AnswerGrade(BaseModel):
    """Structured-output schema for the agent's reflection step: an LLM judges a draft answer
    against the evidence actually gathered so far, so a bounded retry loop can send the agent back
    for more evidence when the draft isn't grounded or doesn't fully address the question, instead
    of returning it as-is."""

    grounded: bool = Field(
        description="True if every factual claim in the draft answer is directly supported by "
        "the provided evidence."
    )
    complete: bool = Field(
        description="True if the draft answer fully addresses every part of the user's "
        "question given the evidence gathered so far."
    )
    feedback: Optional[str] = Field(
        default=None,
        description="If not grounded or not complete, concrete guidance on what additional "
        "evidence or tool calls are needed. Null if grounded and complete.",
    )


class AgentChatResponse(ChatResponse):
    """Response shape for the agent endpoint (`/agent/ask`); extends `ChatResponse` with
    transparency fields specific to the agent loop, without changing the existing
    answer/key_metrics/caveats/sources contract other consumers of `ChatResponse` rely on."""

    tool_calls: List[str] = Field(
        default_factory=list,
        description="Ordered, human-readable log of tool invocations made while answering "
        "(e.g. \"search_filings(query='...', tickers=['AAPL'])\"), for transparency into the "
        "agent's reasoning - not itself a citation.",
    )
    reflection_retried: bool = Field(
        default=False,
        description="True if the reflection step judged the first draft insufficient and "
        "triggered an additional evidence-gathering round.",
    )
