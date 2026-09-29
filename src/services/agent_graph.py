"""
A bounded ReAct-style tool-calling agent loop followed by a Self-RAG/CRAG-style reflection step
(draft -> grade -> retry-with-feedback-or-accept).

`build_agent_graph` is a factory (not a module-level singleton) specifically so tests can inject
fake model/prompt factories and exercise the graph's control flow (tool-loop routing, iteration
caps, reflection retry) without hitting a real LLM. The real, non-test wiring lives in
`agent_service.py`.

Two explicit state counters bound the loop instead of relying on LangGraph's `recursion_limit`:
hitting `recursion_limit` raises and aborts the whole request, whereas the goal here is graceful
degradation - return the best answer gathered so far, with an honest caveat, never a 500.
"""
import logging
import operator
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langfuse import get_client
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from src.models.agent_models import AnswerGrade
from src.models.chat_models import FinancialAnswer
from src.services.hybrid_retrieval import dedup_documents
from src.services.rag_formatting import build_context
from src.utils.observability import observe_child
from src.utils.timing import log_duration

logger = logging.getLogger(__name__)
langfuse = get_client()

MAX_TOOL_ITERATIONS = 4
MAX_REFLECTION_RETRIES = 1
MAX_TOOL_ITERATIONS_PER_RETRY = 2

ToolFn = Callable[..., Awaitable[Tuple[str, List[Document]]]]
ModelFactory = Callable[[], BaseChatModel]
PromptFactory = Callable[..., Any]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    retrieved_chunks: Annotated[List[Document], operator.add]
    draft_answer: Optional[FinancialAnswer]
    tool_call_log: Annotated[List[str], operator.add]
    loop_count: int
    reflection_attempts: int
    grade_verdict: Optional[str]


def initial_state(question: str) -> AgentState:
    """Build the starting state for one `agent_app.ainvoke(...)` run."""
    return AgentState(
        messages=[HumanMessage(content=question)],
        question=question,
        retrieved_chunks=[],
        draft_answer=None,
        tool_call_log=[],
        loop_count=0,
        reflection_attempts=0,
        grade_verdict=None,
    )


def _draft_answer_to_text(answer: FinancialAnswer) -> str:
    lines = [f"Answer: {answer.answer}"]
    if answer.key_metrics:
        lines.append("Key metrics: " + "; ".join(answer.key_metrics))
    if answer.caveats:
        lines.append("Caveats: " + "; ".join(answer.caveats))
    return "\n".join(lines)


def _format_tool_call(name: str, args: Dict[str, Any]) -> str:
    formatted_args = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return f"{name}({formatted_args})"


def build_agent_graph(
    *,
    agent_model_factory: ModelFactory,
    draft_model_factory: ModelFactory,
    grade_model_factory: ModelFactory,
    tool_schemas: Sequence[StructuredTool],
    tool_functions: Dict[str, ToolFn],
    get_agent_system_prompt: PromptFactory,
    get_draft_prompt: PromptFactory,
    get_grade_prompt: PromptFactory,
    max_tool_iterations: int = MAX_TOOL_ITERATIONS,
    max_reflection_retries: int = MAX_REFLECTION_RETRIES,
    max_tool_iterations_per_retry: int = MAX_TOOL_ITERATIONS_PER_RETRY,
):
    """
    Compile the agent graph: agent (tool-calling) <-> tools, then draft_answer -> grade ->
    finalize, with grade able to route back to agent (bounded by `max_reflection_retries`).

    All model/prompt access is injected as factories rather than constructed here, so tests can
    swap in fakes (see `tests/services/test_agent_graph.py`) and production wiring
    (`agent_service.py`) can route every call through `llm_service`/Langfuse as usual.

    Args:
        agent_model_factory: Returns a fresh (unbound) chat model for the tool-calling agent
            node; `.bind_tools(tool_schemas)` is applied here, per invocation.
        draft_model_factory: Returns a fresh chat model for the `draft_answer` node's structured
            `FinancialAnswer` output.
        grade_model_factory: Returns a fresh chat model for the `grade` node's structured
            `AnswerGrade` output.
        tool_schemas: `StructuredTool` schema wrappers bound to the agent model so it can emit
            valid tool calls (see `agent_tools.build_agent_tool_schemas`).
        tool_functions: Maps each tool's name to the underlying async function the `tools` node
            actually calls (see `agent_tools.TOOL_FUNCTIONS`) - not called through `tool_schemas`.
        get_agent_system_prompt: `() -> compiled prompt` (prepended to `state["messages"]` on
            every agent-node call; not persisted into state).
        get_draft_prompt: `(context: str, question: str) -> compiled prompt` for `draft_answer`.
        get_grade_prompt: `(question: str, context: str, draft_answer_text: str) -> compiled
            prompt` for `grade`.
        max_tool_iterations: Base cap on agent<->tools rounds before the first draft; exceeding
            it forces a transition to `draft_answer` regardless of further tool requests. Each
            grade-triggered retry raises the effective cap by `max_tool_iterations_per_retry` (see
            below), so a retry can actually gather new evidence instead of being silently starved
            of tool budget by rounds already spent before the first draft.
        max_reflection_retries: Cap on how many times `grade` may route back to `agent` for more
            evidence; a verdict of "retry" after the cap is exhausted finalizes anyway with an
            appended caveat.
        max_tool_iterations_per_retry: Extra agent<->tools rounds granted for each reflection
            retry already used, on top of `max_tool_iterations`. Without this, a retry triggered
            after the base budget is already spent would have no tool calls available to it and
            would just re-draft from identical evidence before being rejected again.

    Returns:
        A compiled LangGraph app (`.ainvoke(state) -> AgentState`).
    """

    @observe_child(as_type="agent", capture_input=False)
    async def agent_node(state: AgentState) -> dict:
        langfuse.update_current_span(
            input={"loop_count": state["loop_count"], "message_count": len(state["messages"])}
        )
        system_prompt = get_agent_system_prompt()
        prompt_messages = list(system_prompt) if isinstance(system_prompt, list) else [system_prompt]
        model = agent_model_factory().bind_tools(list(tool_schemas))
        with log_duration(logger, f"agent node LLM call (loop_count={state['loop_count']})"):
            response = await model.ainvoke(prompt_messages + list(state["messages"]))
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        # Each reflection retry already spent raises the effective cap, so a retry isn't starved
        # of tool budget by rounds spent before the first draft (see max_tool_iterations_per_retry
        # docstring above).
        effective_cap = max_tool_iterations + state["reflection_attempts"] * max_tool_iterations_per_retry
        if tool_calls and state["loop_count"] < effective_cap:
            return "tools"
        return "draft_answer"

    @observe_child(as_type="chain", capture_input=False, capture_output=False)
    async def tools_node(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        tool_messages: List[BaseMessage] = []
        new_docs: List[Document] = []
        log_entries: List[str] = []

        langfuse.update_current_span(
            input={
                "tool_calls": [
                    _format_tool_call(call["name"], call.get("args", {})) for call in last.tool_calls
                ]
            }
        )

        for call in last.tool_calls:
            name = call["name"]
            args = call.get("args", {})
            fn = tool_functions.get(name)
            if fn is None:
                content, docs = f"Unknown tool: {name}", []
            else:
                try:
                    with log_duration(logger, f"tool call ({name})"):
                        content, docs = await fn(**args)
                except Exception as error:  # tool failure shouldn't crash the whole run
                    logger.warning("Tool %s failed with args %s: %s", name, args, error)
                    content, docs = f"Tool {name} failed: {error}", []
            tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            new_docs.extend(docs)
            log_entries.append(_format_tool_call(name, args))

        langfuse.update_current_span(
            output={"tool_call_log": log_entries, "new_doc_count": len(new_docs)}
        )

        return {
            "messages": tool_messages,
            "retrieved_chunks": new_docs,
            "tool_call_log": log_entries,
            "loop_count": state["loop_count"] + 1,
        }

    @observe_child(as_type="chain", capture_input=False)
    async def draft_answer_node(state: AgentState) -> dict:
        langfuse.update_current_span(
            input={"question": state["question"], "evidence_chunk_count": len(state["retrieved_chunks"])}
        )
        deduped = dedup_documents(state["retrieved_chunks"])
        context = build_context(deduped) if deduped else "No evidence was retrieved."
        prompt = get_draft_prompt(context, state["question"])
        model = draft_model_factory().with_structured_output(FinancialAnswer, method="json_schema")
        with log_duration(logger, "draft_answer node LLM call"):
            draft = await model.ainvoke(prompt)
        return {"draft_answer": draft}

    @observe_child(as_type="chain", capture_input=False)
    async def grade_node(state: AgentState) -> dict:
        langfuse.update_current_span(
            input={"question": state["question"], "evidence_chunk_count": len(state["retrieved_chunks"])}
        )
        deduped = dedup_documents(state["retrieved_chunks"])
        context = build_context(deduped) if deduped else "No evidence was retrieved."
        draft_text = _draft_answer_to_text(state["draft_answer"])
        prompt = get_grade_prompt(state["question"], context, draft_text)
        model = grade_model_factory().with_structured_output(AnswerGrade, method="json_schema")
        with log_duration(logger, "grade node LLM call"):
            grade: AnswerGrade = await model.ainvoke(prompt)

        accepted = grade.grounded and grade.complete
        if accepted or state["reflection_attempts"] >= max_reflection_retries:
            updates: dict = {}
            if not accepted:
                draft = state["draft_answer"]
                updates["draft_answer"] = draft.model_copy(
                    update={
                        "caveats": [
                            *draft.caveats,
                            "This answer could not be fully verified against the retrieved "
                            "evidence within the reasoning budget.",
                        ]
                    }
                )
            return {**updates, "grade_verdict": "finalize"}

        feedback = grade.feedback or "The draft answer was judged incomplete or ungrounded."
        return {
            "messages": [HumanMessage(content=f"Reflection feedback: {feedback}")],
            "reflection_attempts": state["reflection_attempts"] + 1,
            "grade_verdict": "retry",
        }

    def route_after_grade(state: AgentState) -> str:
        return "agent" if state.get("grade_verdict") == "retry" else "finalize"

    async def finalize_node(state: AgentState) -> dict:
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("draft_answer", draft_answer_node)
    graph.add_node("grade", grade_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "draft_answer": "draft_answer"})
    graph.add_edge("tools", "agent")
    graph.add_edge("draft_answer", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"agent": "agent", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()
