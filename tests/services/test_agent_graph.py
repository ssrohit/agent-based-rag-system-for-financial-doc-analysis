"""
Graph control-flow tests for the agent (`agent_graph.py`), using hand-rolled fake chat models
injected via `build_agent_graph`'s factory parameters - no real LLM calls, no Langfuse prompt
fetches, and (since this module never imports `agent_tools.py`/`retrieval_service`) no real
embedding/cross-encoder model loads either.
"""
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.models.agent_models import AnswerGrade
from src.models.chat_models import FinancialAnswer
from src.services.agent_graph import build_agent_graph, initial_state


class ScriptedModel:
    """Fake chat model: `bind_tools`/`with_structured_output` return self (chainable, like the
    real LangChain API); `ainvoke` pops the next response off a fixed script in call order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, method="json_schema"):
        return self

    async def ainvoke(self, prompt):
        response = self._responses[self._index]
        self._index += 1
        return response


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id}


async def _fake_search_filings(query: str, tickers=None) -> tuple:
    doc = Document(page_content=f"evidence for {query}", metadata={"ticker": "AAPL"})
    return f"evidence for {query}", [doc]


def _build_graph(
    *,
    agent_responses: List[AIMessage],
    draft_responses: List[FinancialAnswer],
    grade_responses: List[AnswerGrade],
    max_tool_iterations: int = 4,
    max_reflection_retries: int = 1,
    max_tool_iterations_per_retry: int = 2,
):
    agent_model = ScriptedModel(agent_responses)
    draft_model = ScriptedModel(draft_responses)
    grade_model = ScriptedModel(grade_responses)
    return build_agent_graph(
        agent_model_factory=lambda: agent_model,
        draft_model_factory=lambda: draft_model,
        grade_model_factory=lambda: grade_model,
        tool_schemas=[],
        tool_functions={"search_filings": _fake_search_filings},
        get_agent_system_prompt=lambda: [],
        get_draft_prompt=lambda context, question: "draft-prompt",
        get_grade_prompt=lambda question, context, draft_text: "grade-prompt",
        max_tool_iterations=max_tool_iterations,
        max_reflection_retries=max_reflection_retries,
        max_tool_iterations_per_retry=max_tool_iterations_per_retry,
    )


@pytest.mark.asyncio
async def test_no_tool_call_skips_straight_to_draft_and_grade():
    app = _build_graph(
        agent_responses=[AIMessage(content="I have enough info.", tool_calls=[])],
        draft_responses=[FinancialAnswer(answer="Revenue was $100B", key_metrics=[], caveats=[])],
        grade_responses=[AnswerGrade(grounded=True, complete=True)],
    )

    final = await app.ainvoke(initial_state("What was revenue?"))

    assert final["draft_answer"].answer == "Revenue was $100B"
    assert final["loop_count"] == 0
    assert final["tool_call_log"] == []
    assert final["reflection_attempts"] == 0


@pytest.mark.asyncio
async def test_tool_call_routes_through_tools_and_back_to_agent():
    app = _build_graph(
        agent_responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("search_filings", {"query": "AAPL revenue"}, "call_1")],
            ),
            AIMessage(content="Done.", tool_calls=[]),
        ],
        draft_responses=[FinancialAnswer(answer="Answer", key_metrics=[], caveats=[])],
        grade_responses=[AnswerGrade(grounded=True, complete=True)],
    )

    final = await app.ainvoke(initial_state("What was Apple's revenue?"))

    assert final["loop_count"] == 1
    assert final["tool_call_log"] == ["search_filings(query='AAPL revenue')"]
    assert len(final["retrieved_chunks"]) == 1


@pytest.mark.asyncio
async def test_tool_iteration_cap_forces_draft_answer():
    def _wants_tools(call_id: str) -> AIMessage:
        # A fresh AIMessage per turn - reusing one object across turns would let LangGraph's
        # `add_messages` reducer (which assigns/mutates `.id` in place) treat a later "turn" as
        # an in-place update of the earlier one instead of a new appended message.
        return AIMessage(
            content="",
            tool_calls=[_tool_call("search_filings", {"query": "more"}, call_id)],
        )

    app = _build_graph(
        agent_responses=[_wants_tools("call_1"), _wants_tools("call_2"), _wants_tools("call_3")],
        draft_responses=[FinancialAnswer(answer="Best effort", key_metrics=[], caveats=[])],
        grade_responses=[AnswerGrade(grounded=True, complete=True)],
        max_tool_iterations=2,
    )

    final = await app.ainvoke(initial_state("An open-ended question"))

    assert final["loop_count"] == 2
    assert len(final["tool_call_log"]) == 2
    assert final["draft_answer"].answer == "Best effort"


@pytest.mark.asyncio
async def test_grade_retry_routes_back_once_then_finalizes_with_caveat():
    app = _build_graph(
        agent_responses=[
            AIMessage(content="First pass.", tool_calls=[]),
            AIMessage(content="Second pass.", tool_calls=[]),
        ],
        draft_responses=[
            FinancialAnswer(answer="Draft 1", key_metrics=[], caveats=[]),
            FinancialAnswer(answer="Draft 2", key_metrics=[], caveats=[]),
        ],
        grade_responses=[
            AnswerGrade(grounded=False, complete=False, feedback="Missing net income."),
            AnswerGrade(grounded=False, complete=False, feedback="Still missing net income."),
        ],
        max_reflection_retries=1,
    )

    final = await app.ainvoke(initial_state("What was net income?"))

    assert final["reflection_attempts"] == 1
    assert final["draft_answer"].answer == "Draft 2"
    assert any(
        "could not be fully verified" in caveat for caveat in final["draft_answer"].caveats
    )
    feedback_messages = [
        m for m in final["messages"] if "Reflection feedback" in getattr(m, "content", "")
    ]
    assert len(feedback_messages) == 1


@pytest.mark.asyncio
async def test_reflection_retry_gets_its_own_tool_budget_when_base_budget_is_exhausted():
    """A retry triggered after the base tool-call budget is already spent must still be able to
    call tools - otherwise the retry can never gather the extra evidence `grade` asked for and is
    just a wasted round-trip back to the same draft (see max_tool_iterations_per_retry)."""

    def _wants_tools(call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[_tool_call("search_filings", {"query": "q"}, call_id)],
        )

    app = _build_graph(
        agent_responses=[
            _wants_tools("call_1"),  # loop_count 0 -> 1, exhausts base budget of 1
            AIMessage(content="No more for now.", tool_calls=[]),  # -> first draft
            _wants_tools("call_2"),  # retry: extra per-retry budget lets this through
            AIMessage(content="Done.", tool_calls=[]),  # -> second draft
        ],
        draft_responses=[
            FinancialAnswer(answer="Draft 1", key_metrics=[], caveats=[]),
            FinancialAnswer(answer="Draft 2", key_metrics=[], caveats=[]),
        ],
        grade_responses=[
            AnswerGrade(grounded=False, complete=False, feedback="Need more evidence."),
            AnswerGrade(grounded=True, complete=True),
        ],
        max_tool_iterations=1,
        max_reflection_retries=1,
        max_tool_iterations_per_retry=1,
    )

    final = await app.ainvoke(initial_state("An open-ended question"))

    assert final["loop_count"] == 2
    assert final["tool_call_log"] == [
        "search_filings(query='q')",
        "search_filings(query='q')",
    ]
    assert final["reflection_attempts"] == 1
    assert final["draft_answer"].answer == "Draft 2"
    assert final["draft_answer"].caveats == []


@pytest.mark.asyncio
async def test_zero_per_retry_budget_starves_retry_of_new_evidence():
    """With no per-retry budget, a retry after the base budget is exhausted can't call tools at
    all - documents the old (pre-fix) behavior as an explicit, opt-in configuration rather than
    the default."""

    def _wants_tools(call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[_tool_call("search_filings", {"query": "q"}, call_id)],
        )

    app = _build_graph(
        agent_responses=[
            _wants_tools("call_1"),
            AIMessage(content="No more for now.", tool_calls=[]),
            _wants_tools("call_2"),  # requested, but no budget left -> forced to draft_answer
        ],
        draft_responses=[
            FinancialAnswer(answer="Draft 1", key_metrics=[], caveats=[]),
            FinancialAnswer(answer="Draft 2", key_metrics=[], caveats=[]),
        ],
        grade_responses=[
            AnswerGrade(grounded=False, complete=False, feedback="Need more evidence."),
            AnswerGrade(grounded=True, complete=True),
        ],
        max_tool_iterations=1,
        max_reflection_retries=1,
        max_tool_iterations_per_retry=0,
    )

    final = await app.ainvoke(initial_state("An open-ended question"))

    assert final["loop_count"] == 1
    assert final["tool_call_log"] == ["search_filings(query='q')"]
    assert final["draft_answer"].answer == "Draft 2"
