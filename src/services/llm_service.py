import logging
from typing import Any, List, Optional, Type, TypeVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.config import settings
from src.utils.singleton import Singleton
from src.utils.timing import log_duration

from langfuse.langchain import CallbackHandler

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class LLMService(metaclass=Singleton):
    """
    Centralized LLM Service that handles structured and unstructured
    model invocations with automatic Langfuse tracing integration.
    """

    def __init__(self) -> None:
        # Defaults can be configured from settings if defined, otherwise fall back to gemini-2.5-flash
        self.default_model = getattr(settings, "CHAT_MODEL", "gemini-2.5-flash")

    def _get_callbacks(self, extra_callbacks: Optional[List[BaseCallbackHandler]] = None) -> List[BaseCallbackHandler]:
        """
        Retrieves/creates the LangChain CallbackHandler which automatically integrates
        with the current OpenTelemetry/Langfuse trace context.
        """
        callbacks = extra_callbacks or []
        try:
            # The Langfuse CallbackHandler automatically binds to the active @observe trace
            # context in the current execution thread/task via OpenTelemetry context propagation.
            handler = CallbackHandler()
            callbacks.append(handler)
        except Exception as e:
            logger.warning("Failed to initialize Langfuse CallbackHandler: %s", e)
        return callbacks

    def _resolve_model_name(self, model_name: Optional[str], prompt: Any, extra_kwargs: Optional[dict] = None) -> str:
        """
        Resolves the target model name with the following precedence:
        1. Explicitly provided `model_name` argument.
        2. Model name extracted from prompt config/metadata (e.g., Langfuse prompt client).
        3. Fallback to `self.default_model`.

        Strips provider prefixes (e.g. 'google_genai:gemini-2.5-flash' -> 'gemini-2.5-flash').
        """
        resolved = model_name

        if not resolved and prompt is not None:
            # Check candidate objects for model config (prompt object itself or extra_kwargs)
            candidates = [prompt]
            if extra_kwargs:
                for key in ("langfuse_prompt", "prompt_object", "prompt_client"):
                    if key in extra_kwargs and extra_kwargs[key] is not None:
                        candidates.insert(0, extra_kwargs[key])

            for target in candidates:
                # Check target.config (e.g. Langfuse TextPromptClient / ChatPromptClient)
                config = getattr(target, "config", None)
                if isinstance(config, dict) and config.get("model"):
                    resolved = config.get("model")
                    break

                # Check target.metadata
                metadata = getattr(target, "metadata", None)
                if isinstance(metadata, dict) and metadata.get("model"):
                    resolved = metadata.get("model")
                    break

                # Check direct target.model attribute
                model_attr = getattr(target, "model", None)
                if isinstance(model_attr, str) and model_attr:
                    resolved = model_attr
                    break

        if not resolved:
            resolved = self.default_model

        # Clean provider prefix if present (e.g. 'google_genai:gemini-2.5-flash' -> 'gemini-2.5-flash')
        if resolved and ":" in resolved:
            resolved = resolved.split(":")[-1]

        return resolved

    def get_chat_model(
        self,
        prompt: Any = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        extra_callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs: Any
    ) -> ChatGoogleGenerativeAI:
        """
        Build a `ChatGoogleGenerativeAI` instance through the same model-resolution and
        Langfuse-callback-attachment path `ainvoke`/`ainvoke_structured` use internally, exposed
        publicly for callers that need the raw model object (e.g. to `.bind_tools(...)` for an
        agent loop) rather than one of this service's narrower invoke helpers.

        Args:
            prompt: A Langfuse prompt object (or None) used for model-name resolution via
                `_resolve_model_name`; not sent to the model itself.
            model_name: Explicit model name override; see `_resolve_model_name` precedence.
            temperature: Sampling temperature.
            extra_callbacks: Additional LangChain callbacks beyond the auto-attached Langfuse one.
            **kwargs: Forwarded to `ChatGoogleGenerativeAI(...)`; also inspected for a
                `langfuse_prompt`/`prompt_object`/`prompt_client` key during model-name resolution.

        Returns:
            A configured, not-yet-invoked `ChatGoogleGenerativeAI` instance.
        """
        selected_model = self._resolve_model_name(model_name, prompt, kwargs)
        callbacks = self._get_callbacks(extra_callbacks)
        return ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=temperature,
            callbacks=callbacks,
            **kwargs
        )

    async def ainvoke_structured(
        self,
        prompt: Any,
        schema: Type[T],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        extra_callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs: Any
    ) -> T:
        """
        Asynchronously invokes the LLM with structured output mapping to the specified Pydantic schema.
        """
        selected_model = self._resolve_model_name(model_name, prompt, kwargs)
        try:
            model = self.get_chat_model(
                prompt=prompt,
                model_name=selected_model,
                temperature=temperature,
                extra_callbacks=extra_callbacks,
                **kwargs
            )
            structured_model = model.with_structured_output(
                schema=schema,
                method="json_schema"
            )

            # We cast to Type T since the returned value is guaranteed to match the schema
            with log_duration(logger, f"LLM structured call (model={selected_model}, schema={schema.__name__})"):
                response = await structured_model.ainvoke(prompt)
            return response
        except Exception as error:
            logger.error(
                "Error during structured LLM invocation (model: %s): %s",
                selected_model,
                str(error),
                exc_info=True
            )
            raise error

    async def ainvoke(
        self,
        prompt: Any,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        extra_callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs: Any
    ) -> str:
        """
        Asynchronously invokes the LLM and returns the raw string content response.
        """
        selected_model = self._resolve_model_name(model_name, prompt, kwargs)
        try:
            model = self.get_chat_model(
                prompt=prompt,
                model_name=selected_model,
                temperature=temperature,
                extra_callbacks=extra_callbacks,
                **kwargs
            )

            with log_duration(logger, f"LLM call (model={selected_model})"):
                response = await model.ainvoke(prompt)
            return str(response.content)
        except Exception as error:
            logger.error(
                "Error during unstructured LLM invocation (model: %s): %s",
                selected_model,
                str(error),
                exc_info=True
            )
            raise error


# Expose a singleton instance for standard imports
llm_service = LLMService()
