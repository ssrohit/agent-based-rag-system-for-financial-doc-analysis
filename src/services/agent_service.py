"""
Agent entrypoint: wires the LangGraph agent (`agent_graph.py`) to real tools (`agent_tools.py`),
real models/prompts (`llm_service`/Langfuse), and returns an `AgentChatResponse` - the agent
counterpart to `chat_service.process_user_message`.
"""
import logging

from langfuse import get_client, observe

from src.models.agent_models import AgentChatResponse
from src.models.chat_models import UserMessage
from src.services.agent_graph import AgentState, build_agent_graph, initial_state
from src.services.agent_tools import TOOL_FUNCTIONS, build_agent_tool_schemas
from src.services.hybrid_retrieval import dedup_documents
from src.services.llm_service import llm_service
from src.services.rag_formatting import to_sources
from src.utils.timing import log_duration

logger = logging.getLogger(__name__)
langfuse = get_client()

_AGENT_SYSTEM_PROMPT_NAME = "dev/agent-system"
_FINANCIAL_QA_PROMPT_NAME = "dev/financial-qa"
_ANSWER_GRADER_PROMPT_NAME = "dev/answer-grader"


def _model_factory(prompt_name: str):
    """
    Build an `agent_graph.ModelFactory` (a no-arg callable returning a fresh chat model) that
    resolves its model via `llm_service.get_chat_model`, using the named Langfuse prompt's own
    `config`/`metadata` for model-name resolution - so each node's model choice reflects its own
    prompt, not a single global default.

    The prompt is re-fetched on every call (rather than once, up front) so a model-name change
    published in Langfuse takes effect without a server restart; `langfuse.get_prompt` caches
    internally, so this is not a fresh network round-trip on every agent-loop iteration.
    """
    return lambda: llm_service.get_chat_model(prompt=langfuse.get_prompt(prompt_name))


def _get_agent_system_prompt():
    return langfuse.get_prompt(_AGENT_SYSTEM_PROMPT_NAME).compile()


def _get_draft_prompt(context: str, question: str):
    return langfuse.get_prompt(_FINANCIAL_QA_PROMPT_NAME).compile(context=context, question=question)


def _get_grade_prompt(question: str, context: str, draft_answer_text: str):
    prompt = langfuse.get_prompt(_ANSWER_GRADER_PROMPT_NAME)
    return prompt.compile(question=question, context=context, draft_answer=draft_answer_text)


def build_default_agent_app():
    """
    Build the agent graph wired to production dependencies: `llm_service`-backed models (with
    automatic Langfuse tracing/model-resolution), Langfuse-managed prompts, and the real
    retrieval-backed tools in `agent_tools.py`.

    Returns:
        A compiled LangGraph app, as returned by `build_agent_graph`.
    """
    return build_agent_graph(
        agent_model_factory=_model_factory(_AGENT_SYSTEM_PROMPT_NAME),
        draft_model_factory=_model_factory(_FINANCIAL_QA_PROMPT_NAME),
        grade_model_factory=_model_factory(_ANSWER_GRADER_PROMPT_NAME),
        tool_schemas=build_agent_tool_schemas(),
        tool_functions=TOOL_FUNCTIONS,
        get_agent_system_prompt=_get_agent_system_prompt,
        get_draft_prompt=_get_draft_prompt,
        get_grade_prompt=_get_grade_prompt,
    )


agent_app = build_default_agent_app()


@observe()
async def run_agent(data: UserMessage) -> AgentChatResponse:
    """
    Answer a free-text financial question via the agent loop: bounded tool-calling (document
    search / company comparison / calculator) followed by a bounded reflection
    (draft -> grade -> retry-or-accept) before returning.

    Args:
        data: The incoming user message wrapper (`data.user_message` is the raw question text).

    Returns:
        An `AgentChatResponse` with the final (possibly reflection-amended) answer, sources built
        directly from retrieved-chunk metadata (never LLM-generated), and a human-readable log of
        every tool call made along the way.
    """
    query = data.user_message
    with log_duration(logger, f"run_agent total (query={query!r})"):
        final_state: AgentState = await agent_app.ainvoke(initial_state(query))

    draft = final_state["draft_answer"]
    deduped_chunks = dedup_documents(final_state["retrieved_chunks"])
    logger.debug(
        "Agent run complete for query=%r: %d tool call(s), %d unique chunk(s), reflection_attempts=%d",
        query,
        len(final_state["tool_call_log"]),
        len(deduped_chunks),
        final_state["reflection_attempts"],
    )

    return AgentChatResponse(
        answer=draft.answer,
        key_metrics=draft.key_metrics,
        caveats=draft.caveats,
        sources=to_sources(deduped_chunks),
        tool_calls=final_state["tool_call_log"],
        reflection_retried=final_state["reflection_attempts"] > 0,
    )
