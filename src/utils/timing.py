"""
Shared helper for logging how long a time-consuming step took (LLM calls, retrieval, SEC
downloads, embedding, ...), so every call site logs duration the same way instead of each
service hand-rolling its own `time.perf_counter()` bookkeeping.
"""
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def log_duration(logger: logging.Logger, label: str, *, level: int = logging.INFO) -> Generator[None]:
    """
    Log how long the wrapped block took, on success or failure.

    Args:
        logger: The caller's module logger (each service already has one via
            `logging.getLogger(__name__)`).
        label: Human-readable description of the step, e.g. "LLM call (gemini-2.5-flash)".
        level: Log level for the success case; failures always log at ERROR regardless, so a
            slow failing step isn't silently missed just because callers picked a quieter level.

    Yields:
        None. Wrap the timed code in the `with` block.
    """
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.error("%s failed after %.3fs", label, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s took %.3fs", label, elapsed)
