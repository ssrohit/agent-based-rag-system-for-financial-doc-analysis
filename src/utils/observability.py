"""
`observe_child`: like Langfuse's `@observe()`, but only takes effect when already running inside
an active Langfuse trace.

Plain `@observe()` starts a brand-new root trace whenever there's no active parent span in the
current OpenTelemetry context - including when the decorated function is called directly (e.g. a
LangGraph node invoked by a test that builds the graph itself, bypassing `agent_service.run_agent`,
which is the only place that opens the real request-level trace). That produced dozens of
orphaned, un-parented `agent_node`/`tools_node`/`draft_answer_node`/`grade_node`/tool/retriever
traces in Langfuse instead of the single trace a real request produces - "many traces instead of
one" for what was conceptually a single call. `observe_child` closes that gap: outside an active
trace it just calls the plain function (no tracing side effect at all), so instrumentation code can
be called directly - by tests, scripts, future callers - without polluting Langfuse.
"""
import functools
from typing import Any, Awaitable, Callable, TypeVar

from langfuse import get_client, observe

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def observe_child(**observe_kwargs: Any) -> Callable[[F], F]:
    """
    Decorator factory for async functions that should only be traced as a *child* span of an
    already-open Langfuse trace - never as a trace of their own.

    Use this instead of `@observe()` on any function that is meant to always run inside a larger
    traced call (a LangGraph node, a tool, a retrieval helper) but may also be called directly by
    something with no trace open (tests, scripts, a REPL). `@observe()` itself can't tell the
    difference: with no active trace it just starts a new root one, which is how a single
    conceptual call can end up producing several unrelated traces in Langfuse.

    Args:
        **observe_kwargs: Forwarded verbatim to `langfuse.observe()` when there is an active trace
            to attach to (e.g. `as_type="tool"`, `capture_input=False`). Ignored otherwise, since
            no span is created in that case.

    Returns:
        A decorator for an `async def` function. The wrapped function:
        - runs exactly as `@observe(**observe_kwargs)`-wrapped would (created as a child span of
          the current trace) when `langfuse.get_client().get_current_trace_id()` is not `None`;
        - otherwise runs the original, undecorated function directly, with no span created and no
          data sent to Langfuse.
        Either way the return value is unchanged from calling the plain function.
    """

    def decorator(fn: F) -> F:
        observed_fn = observe(**observe_kwargs)(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if get_client().get_current_trace_id() is None:
                return await fn(*args, **kwargs)
            return await observed_fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
