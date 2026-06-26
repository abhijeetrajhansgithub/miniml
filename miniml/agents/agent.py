import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, 
from miniml.inference.engines.engine_frame import LLMResponse, LLMSingleResponse


class AgentRunnable(ABC):
    """
    Abstract base for all agent runnables.

    Enforcement levels:
      1. @abstractmethod       — Python ABC: blocks instantiation if not overridden.
      2. __init_subclass__     — fires at *class definition* time, before any object
                                 is ever created. Catches missing or non-callable
                                 overrides immediately when the subclass module loads.
    """

    # All methods every subclass MUST concretely implement.
    _REQUIRED_METHODS = (
        "run",
        "_run_agent_loop",
        "_generate",
        "_validate",
        "_call_llm",
        "_execute_tool",
        "_load_prompt",
        "_record",
        "_record_error",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Skip enforcement on intermediate abstract classes
        # (i.e. classes that are themselves still abstract)
        if inspect.isabstract(cls):
            return

        missing: list[str] = []
        not_overridden: list[str] = []

        for method_name in cls._REQUIRED_METHODS:
            member = getattr(cls, method_name, None)

            # 1. Must exist and be callable
            if member is None or not callable(member):
                missing.append(method_name)
                continue

            # 2. Must be overridden — not just inherited from AgentRunnable
            if method_name in AgentRunnable.__dict__:
                for klass in cls.__mro__:
                    if method_name in klass.__dict__:
                        if klass is AgentRunnable:
                            not_overridden.append(method_name)
                        break

        errors: list[str] = []

        if missing:
            errors.append(
                f"  Missing methods (not defined at all): {missing}"
            )
        if not_overridden:
            errors.append(
                f"  Inherited but not overridden (still points to base): {not_overridden}"
            )

        if errors:
            raise TypeError(
                f"\n[AgentRunnable] Class '{cls.__name__}' does not fully implement "
                f"the required interface:\n" + "\n".join(errors)
            )

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> Any:
        """Public entry point. Returns the final result or a fallback message."""
        raise NotImplementedError

    @abstractmethod
    def _run_agent_loop(self) -> Optional[str] | Any:
        """Core generate -> validate loop. Returns the result or None on exhaustion."""
        raise NotImplementedError

    @abstractmethod
    def _generate(self, prompt: str) -> LLMSingleResponse:
        """
        Call the LLM for generation and parse the response.
        Returns a typed dict with '_instance' key, or None on parse failure.
        """
        raise NotImplementedError

    @abstractmethod
    def _validate(self, prompt: str) -> LLMSingleResponse:
        """
        Call the LLM for validation and parse the response.
        Returns a typed dict with '_instance' key, or None on parse failure.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _call_llm(
        self,
        prompt: str,
        stage: str,
        use_tools: bool | None = None,
    ) -> LLMResponse:
        """
        Dispatch a prompt to the configured LLM provider.
        """
        raise NotImplementedError

    # TODO: Refactor agent code to incorporate LLMResponse dataclass

    @abstractmethod
    def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
        """Look up and invoke a tool from the registry, then record the result."""
        raise NotImplementedError

    @abstractmethod
    def _load_prompt(
        self,
        yaml_key_path: str,
        prompt_key: str,
        inject_tools: bool = False,
    ) -> str:
        """Load a prompt string from a YAML file by key."""
        raise NotImplementedError

    @abstractmethod
    def _record(self, stage: str, task: str, data: Any) -> None:
        """Write an entry to both active memory and message history."""
        raise NotImplementedError

    @abstractmethod
    def _record_error(self, stage: str, task: str, exc: Exception) -> None:
        """Record a caught exception to memory and message history."""
        raise NotImplementedError
    
    @abstractmethod
    def _get_most_approximate_tool(self, tool_name: str) -> tuple[str, str]:
        """Get the most appropriate tool if there is no exact tool name match"""
        raise NotImplementedError