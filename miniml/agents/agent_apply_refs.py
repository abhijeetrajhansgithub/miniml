from typing import Optional, Dict, Any, List, Callable
from difflib import SequenceMatcher
import inspect

from miniml.agents.agent import AgentRunnable
from miniml._messages.messages import MessageHistory, TaggedMessage
from miniml.parsers.AgentApplyRefsParser import AgentApplyRefsValidationParser
from miniml.inference.engines.engine_frame import GetRefsEngineFrame
from miniml.utils.utilities import unpack_arguments, is_required_parameter, get_callable_args

from miniml.inference.engines.ollama_engine import get_response_ollama
from miniml.inference.engines.openrouter_engine import get_response_openrouter

from miniml.tools.toolreg import ToolRegistry, Tool, get_builtin_tools

_builtin_tools_names = get_builtin_tools()

import yaml
import json
from pathlib import Path
import pandas as pd

TOOL_REGISTRY = ToolRegistry()

# ── Debug helpers ──────────────────────────────────────────────────────────── #

_DBG_SEP = "─" * 60

def _dbg(section: str, label: str, value: Any = "") -> None:
    print(f"\n{_DBG_SEP}")
    print(f"[{section}] {label}")
    if value != "":
        print(value)
    print(_DBG_SEP)


# ── Agent class ────────────────────────────────────────────────────────────── #
class AgentApplyRefs(AgentRunnable):
    INFERENCE_TRIES_LIMIT: int = 3

    PROMPT_PATHS = {
        "prompts_validation": "miniml/prompts/validation.yaml",
        "prompts_generation":  "miniml/prompts/generation.yaml",
    }

    def __init__(self,
        column_inference: GetRefsEngineFrame,
        main_dataframe: pd.DataFrame,
        _parent_base_dir_path: str = None,
        tools: Optional[List[Tool]] = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:

        assert column_inference is not None, "Column inference is required"
        self.column_inference = column_inference

        assert main_dataframe is not None, "Main Dataframe can never be None"
        self.main_dataframe = main_dataframe

        assert _parent_base_dir_path is not None, "Parent base directory path is required"
        self._parent_base_dir_path = _parent_base_dir_path

        assert provider is not None, "Provider is required"
        self.provider = provider

        assert model is not None, "Model is required"
        self.model = model

        self.column_context = (
            f"Column name: {self.column_inference.column_name}; "
            f"Column type: {self.column_inference.column_type}; "
            f"Imputation strategy: {self.column_inference.imputation_strategy}; "
            f"Imputation reasoning: {self.column_inference.imputation_reasoning}; "
            f"Outlier Strategy: {self.column_inference.outlier_strategy}; "
            f"Outlier reasoning: {self.column_inference.outlier_reasoning}"
        )

        # tools — builtin and user-defined
        self._builtin_tools: List[Tool] = [
            TOOL_REGISTRY.get_tool(name=tool_name) for tool_name in _builtin_tools_names
        ]

        if tools is None:
            tools = self._builtin_tools

        self._user_tools: List[Tool] = [
            tool for tool in tools if tool not in self._builtin_tools
        ]

        print("=" * 50)
        print("PARAMS")
        print(f"parent_base_dir_path: {self._parent_base_dir_path}")
        print(f"provider: {self.provider}")
        print(f"model: {self.model}")
        print("=" * 50)

        self.message_history = MessageHistory()

        self._base_generation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "base_generation_prompt_for_apply_refs",
            inject_tools=True,
        )
        self._retry_generation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "retry_generation_prompt_for_apply_refs",
            inject_tools=True,
        )

        # validation 1 — tool-exec check
        self._base_validation_prompt_for_tool_exec = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "base_validation_prompt_for_apply_refs_for_tool_exec",
            inject_tools=True,
        )
        self._retry_validation_prompt_for_tool_exec = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "retry_validation_prompt_for_apply_refs_for_tool_exec",
            inject_tools=True,
        )

        # validation 2 — proc-end check
        self._base_validation_prompt_for_proc_end = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "base_validation_prompt_for_apply_refs_for_proc_end",
            inject_tools=False,
        )
        self._retry_validation_prompt_for_proc_end = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "retry_validation_prompt_for_apply_refs_for_proc_end",
            inject_tools=False,
        )

        print("=" * 50)
        print("Base generation prompt length:", len(self._base_generation_prompt))
        print("Retry generation prompt length:", len(self._retry_generation_prompt))
        print("Base validation (tool-exec) prompt length:", len(self._base_validation_prompt_for_tool_exec))
        print("Retry validation (tool-exec) prompt length:", len(self._retry_validation_prompt_for_tool_exec))
        print("=" * 50)

    # ──────────────────────────────────────────────────────────────────────── #
    # PRIMITIVE: call the LLM and return (content, raw_tool_calls).
    # ──────────────────────────────────────────────────────────────────────── #
    def _call_llm(
        self,
        prompt: str,
        stage: str,
        use_tools: bool = False,
    ) -> tuple[str, list]:
        """
        Call the LLM and return (content, raw_tool_calls).

        `use_tools` — forward the tool manifest to the provider.
        Generation calls set this True; validation calls set it False.
        """
        options: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 300,
        }

        _tools = []
        if use_tools:
            for tool in self._builtin_tools:
                if tool.type.lower() in ["imputer", "outlier"]:
                    _tools.append(tool.json)
            for tool in self._user_tools:
                if tool.type.lower() in ["imputer", "outlier"]:
                    _tools.append(tool.json)

        _dbg("LLM", f"Calling provider '{self.provider}' stage='{stage}' use_tools={use_tools}")

        if self.provider == "ollama":
            result = get_response_ollama(
                prompt=prompt,
                model=self.model,
                options_dict=options,
                tools=_tools if use_tools else [],
            )
            message = result.get("message", {})
            content = message.get("content", "")
            raw_tool_calls = message.get("tool_calls") or []

        elif self.provider == "openrouter":
            result = get_response_openrouter(
                prompt=prompt,
                model=self.model,
                options_dict=options,
                tools=_tools if use_tools else None,
            )
            content = result.get("choices", [])[0].get("message", {}).get("content", "")
            raw_tool_calls = result.get("choices", [])[0].get("message", {}).get("tool_calls", [])

        else:
            raise ValueError(f"Invalid provider: {self.provider}")

        _dbg("LLM", f"Content: {content}")
        _dbg("LLM", f"Raw tool calls: {raw_tool_calls}")

        return content, raw_tool_calls

    # ──────────────────────────────────────────────────────────────────────── #
    # GENERATION: build prompt → call LLM (with tools) →
    #             validate tool selection → execute tools.
    # ──────────────────────────────────────────────────────────────────────── #
    def _generate(self, prompt: str, is_retry: bool) -> Optional[str]:
        assert prompt is not None, "Prompt cannot be None"

        content, raw_tool_calls = self._call_llm(
            prompt=prompt,
            stage="generation",
            use_tools=True,
        )

        _dbg("GENERATE", f"content={content!r}  tool_calls={raw_tool_calls}")

        # ── Tool-exec validation ──────────────────────────────────────────── #
        # Only feed tool-exec validation failures into the retry context —
        # not execution records or proc-end failures, which caused the
        # poisoned-history bug.
        if is_retry:
            retry_ctx = self._retry_validation_prompt_for_tool_exec.replace(
                "[FEEDBACK]",
                self.message_history.get_tagged_string(
                    unique=True,
                    task="tool_exec_validation_failure"
                ),
            )
        else:
            retry_ctx = ""

        val_prompt_tool = self._base_validation_prompt_for_tool_exec.replace(
            "[AGENT_MEMORY]", retry_ctx
        )
        val_prompt_tool = val_prompt_tool.replace(
            "[CONTEXT]", self.column_context
        )
        val_prompt_tool = val_prompt_tool.replace(
            "[OUTPUT]",
            f"Tools selected: {', '.join([f.get('function').get('name') for f in raw_tool_calls])}",
        )

        _dbg("VALIDATION TOOL CHECK PROMPT", val_prompt_tool)

        val_content, _ = self._call_llm(
            prompt=val_prompt_tool,
            stage="validation",
            use_tools=False,
        )

        _dbg("VALIDATION TOOL CHECK RESPONSE", val_content)

        val_parser = AgentApplyRefsValidationParser(response=val_content)
        val_parsed = val_parser.parse()

        _dbg("VALIDATION TOOL CHECK PARSED", val_parsed)

        if val_parsed.get("_instance") == "error":
            self._record_tool_exec_validation_failure(
                content=f"[ERROR] Parser failed: {val_parsed.get('error')} "
                        f"| Tools selected: {', '.join([f.get('function').get('name') for f in raw_tool_calls])}",
            )
            return None

        if val_parsed.get("_instance") == "success":
            if val_parsed.get("valid") is False:
                self._record_tool_exec_validation_failure(
                    content=(
                        val_parsed.get("reasoning", "")
                        + f" | Tools selected: {', '.join([f.get('function').get('name') for f in raw_tool_calls])}"
                    ),
                )
                return None

        # ── Execute tools ─────────────────────────────────────────────────── #
        if raw_tool_calls:
            for function in raw_tool_calls:
                name = function.get("function", {}).get("name", "")
                arguments = function.get("function", {}).get("arguments", {})
                _dbg("LLM", f"Function call: {name}  arguments: {arguments}")

                self._execute_tool(tool_response={"name": name, "arguments": arguments})
                # Confirmed dispatch — recorded inside _execute_tool as
                # stage="tool_execution", task="tool_execution".
                # Record the lightweight "tool call dispatched" note separately.
                self._record_tool_call(tool_name=name)

        return content

    # ──────────────────────────────────────────────────────────────────────── #
    # VALIDATION (proc-end only): pure LLM text call, no tool execution.
    # ──────────────────────────────────────────────────────────────────────── #
    def _validate(self, prompt: str) -> Optional[str]:
        assert prompt is not None, "Prompt cannot be None"

        content, _ = self._call_llm(
            prompt=prompt,
            stage="validation",
            use_tools=False,
        )

        if not content:
            _dbg("VALIDATE", "LLM returned empty content")
            return None

        return content

    # ──────────────────────────────────────────────────────────────────────── #
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────────────────── #
    def _run_agent_loop(self) -> Optional[str] | Any:
        tries = 0

        while tries < self.INFERENCE_TRIES_LIMIT:
            tries += 1
            is_retry = tries > 1

            _dbg("LOOP", f"Iteration {tries}/{self.INFERENCE_TRIES_LIMIT}", f"is_retry={is_retry}")

            # ── Generation ───────────────────────────────────────────────── #
            # Feed all failure reasons (both tool-exec and proc-end) so the
            # generator knows what went wrong across all previous attempts.
            if is_retry:
                all_failures = self.message_history.get_tagged_string(
                    unique=True,
                    task="tool_exec_validation_failure"
                )
                proc_failures = self.message_history.get_tagged_string(
                    unique=True,
                    task="proc_end_validation_failure"
                )
                combined_failures = "\n".join(
                    filter(None, [all_failures, proc_failures])
                )
                retry_ctx = self._retry_generation_prompt.replace(
                    "[HISTORY]", combined_failures
                )
            else:
                retry_ctx = ""

            gen_prompt = self._base_generation_prompt.replace(
                "[AGENT_MEMORY]", retry_ctx
            )
            gen_prompt = gen_prompt.replace(
                "[CONTEXT]", self.column_context
            )

            _dbg("GENERATION PROMPT", gen_prompt)

            response_generated = self._generate(prompt=gen_prompt, is_retry=is_retry)

            _dbg("GENERATION RESPONSE", response_generated)

            # Generation + tool validation failed — retry.
            if response_generated is None:
                continue

            # ── Proc-end validation ───────────────────────────────────────── #
            # Retry context: only proc-end failures from previous iterations.
            if is_retry:
                retry_ctx = self._retry_validation_prompt_for_proc_end.replace(
                    "[FEEDBACK]",
                    self.message_history.get_tagged_string(
                        unique=True,
                        task="proc_end_validation_failure"
                    ),
                )
            else:
                retry_ctx = ""

            val_prompt = self._base_validation_prompt_for_proc_end.replace(
                "[AGENT_MEMORY]", retry_ctx
            )
            val_prompt = val_prompt.replace(
                "[CONTEXT]", self.column_context
            )

            # Give the proc-end validator only confirmed execution records —
            # clean, unambiguous evidence that tools actually ran.
            val_prompt = val_prompt.replace(
                "[EXECUTION_HISTORY]",
                self.message_history.get_tagged_string(
                    unique=True,
                    stage="tool_execution"),
            )

            _dbg("PROC-END VALIDATION PROMPT", val_prompt)

            response_validated = self._validate(prompt=val_prompt)

            _dbg("PROC-END VALIDATION RESPONSE", response_validated)

            val_result = AgentApplyRefsValidationParser(response=response_validated)
            val_result = val_result.parse()

            _dbg("PROC-END VALIDATION PARSED", val_result)

            if val_result.get("_instance") == "error":
                self._record_proc_end_validation_failure(
                    content=(
                        f"[ERROR] Parser failed: {val_result.get('error')} "
                        f"| imputation={self.column_inference.imputation_strategy}"
                        f", outlier={self.column_inference.outlier_strategy}"
                    ),
                )
                continue

            if val_result.get("_instance") == "success":
                if val_result.get("valid") is False:
                    self._record_proc_end_validation_failure(
                        content=(
                            val_result.get("reasoning", "")
                            + f" | imputation={self.column_inference.imputation_strategy}"
                            + f", outlier={self.column_inference.outlier_strategy}"
                        ),
                    )
                    continue
                else:
                    return self.main_dataframe

        # Exhausted retries — fall back to deterministic execution.
        return self._run_manual_tool_exec()

    def run(self) -> Any:
        _dbg("RUN", "Starting ApplyRefs agent run")
        return self._run_agent_loop()

    # ──────────────────────────────────────────────────────────────────────── #
    # FALLBACK
    # ──────────────────────────────────────────────────────────────────────── #
    def _run_manual_tool_exec(self) -> pd.DataFrame:
        _tool_reqs = [
            self.column_inference.imputation_strategy,
            self.column_inference.outlier_strategy,
        ]

        for strategy in _tool_reqs:
            tool_name, _ = self._get_most_approximate_tool(strategy)
            tool = TOOL_REGISTRY.get_tool(tool_name)
            self.main_dataframe = tool.func(
                df=self.main_dataframe,
                cols=[self.column_inference.column_name],
            )

        return self.main_dataframe

    # ──────────────────────────────────────────────────────────────────────── #
    # TOOL EXECUTION
    # ──────────────────────────────────────────────────────────────────────── #
    def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
        name = tool_response.get("name", "")
        arguments = tool_response.get("arguments", {})

        _dbg("Tool", f"Executing: {name}  arguments: {arguments}")

        if name.lower() not in TOOL_REGISTRY.get_tool_names():
            name, _ = self._get_most_approximate_tool(tool_name=name)

        arguments = unpack_arguments(data=arguments)
        _dbg("ARGUMENTS", str(arguments))

        all_args = dict(arguments)

        if "cols" not in all_args:
            all_args["cols"] = [self.column_inference.column_name]

        registered_tool: Tool = TOOL_REGISTRY.get_tool(name=name)

        if is_required_parameter(registered_tool.func, "df"):
            all_args["df"] = self.main_dataframe

        filtered_args = get_callable_args(registered_tool.func, all_args)
        tool_result: pd.DataFrame | Any | None = registered_tool.func(**filtered_args)

        # Tagged as stage="tool_execution", task="tool_execution" so that
        # proc-end validator can retrieve clean execution evidence via
        # message_history.get_tagged_string(stage="tool_execution").
        self._record_tool_execution(
            tool_name=name,
            arguments=arguments,
            result=(
                tool_result
                if isinstance(tool_result, str)
                else f"Successfully used tool: {name} with args: {arguments}"
            ),
        )

        if isinstance(tool_result, pd.DataFrame):
            self.main_dataframe = tool_result

    # ──────────────────────────────────────────────────────────────────────── #
    # TYPED RECORD HELPERS
    # Each method maps to exactly one (stage, task) pair so call sites are
    # self-documenting and the history is trivially filterable.
    # ──────────────────────────────────────────────────────────────────────── #

    def _record_tool_exec_validation_failure(self, content: str) -> None:
        """Record a tool-exec validator rejection."""
        _dbg("Recorder", "tool_exec_validation_failure", content)
        self.message_history.add_message(
            TaggedMessage(
                role="assistant",
                content=content,
                stage="tool_exec_validation",
                task="tool_exec_validation_failure",
            )
        )

    def _record_tool_execution(
        self,
        tool_name: str,
        arguments: Any,
        result: str,
    ) -> None:
        """Record a confirmed, successfully dispatched tool execution."""
        content = str({
            "tool_name": tool_name,
            "tool_arguments": arguments,
            "tool_result": result,
        })
        _dbg("Recorder", "tool_execution", content)
        self.message_history.add_message(
            TaggedMessage(
                role="assistant",
                content=content,
                stage="tool_execution",
                task="tool_execution",
            )
        )

    def _record_tool_call(self, tool_name: str) -> None:
        """Record that a tool was dispatched (lightweight breadcrumb)."""
        content = f"Tool used: {tool_name}"
        _dbg("Recorder", "tool_call", content)
        self.message_history.add_message(
            TaggedMessage(
                role="assistant",
                content=content,
                stage="tool_execution",
                task="tool_call",
            )
        )

    def _record_proc_end_validation_failure(self, content: str) -> None:
        """Record a proc-end validator rejection."""
        _dbg("Recorder", "proc_end_validation_failure", content)
        self.message_history.add_message(
            TaggedMessage(
                role="assistant",
                content=content,
                stage="proc_end_validation",
                task="proc_end_validation_failure",
            )
        )

    def _record_error(self, stage: str, task: str, exc: Exception) -> None:
        """
        Generic error recorder kept for unexpected / uncategorised errors.
        Prefer the typed helpers above for known failure paths.
        """
        labeled = f"[ERROR][stage={stage}][task={task}][type={type(exc).__name__}] {exc}"
        _dbg("Recorder", f"ERROR stage='{stage}' task='{task}'", labeled)
        self.message_history.add_message(
            TaggedMessage(
                role="assistant",
                content=labeled,
                stage=stage,       # type: ignore[arg-type]
                task="error",
            )
        )

    def _record(self, stage: str, task: str, data: Any) -> None:
        return super()._record(stage, task, data)

    # ──────────────────────────────────────────────────────────────────────── #
    # PROMPT LOADING
    # ──────────────────────────────────────────────────────────────────────── #
    def _load_prompt(
        self,
        yaml_key_path: str,
        prompt_key: str,
        inject_tools: bool = False,
    ) -> str:
        prompt_path = Path(self._parent_base_dir_path) / yaml_key_path

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)

        prompt = prompts[prompt_key]

        if inject_tools:
            prompt = prompt.replace(
                "[USER_TOOLS]",
                "--\n".join([f"TOOL: {tool.name}" for tool in self._user_tools if tool.type.lower() in ["imputer", "outlier"]]),
            )
            prompt = prompt.replace(
                "[BUILTIN_TOOLS]",
                "--\n".join([f"TOOL: {tool.name}" for tool in self._builtin_tools if tool.type.lower() in ["imputer", "outlier"]]),
            )

        return prompt

    # ──────────────────────────────────────────────────────────────────────── #
    # FUZZY TOOL MATCHING
    # ──────────────────────────────────────────────────────────────────────── #
    def _get_most_approximate_tool(self, tool_name: str) -> tuple[str, str]:
        _user_tools: list[str] = [tool.name for tool in self._user_tools]
        _builtin_tools: list[str] = [tool.name for tool in self._builtin_tools]

        best_match = ""
        best_list_name = ""
        best_score = 0.0

        for value in _user_tools:
            score = SequenceMatcher(None, tool_name.lower(), value.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = value
                best_list_name = "_user_tools"

        for value in _builtin_tools:
            score = SequenceMatcher(None, tool_name.lower(), value.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = value
                best_list_name = "_builtin_tools"

        return best_match, best_list_name