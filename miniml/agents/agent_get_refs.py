from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

from miniml.agents.agent import AgentRunnable
from miniml._messages.messages import MessageHistory, TaggedMessage, AgentMessage
from miniml.inference.engines.engine_frame import GetRefsEngineFrame, LLMResponse, LLMSingleResponse, AgentGetRefsIOResponse, AgentParserResponse

from miniml.parsers.AgentGetRefsParser import AgentGetRefsGenerationParser, AgentGetRefsValidationParser

from miniml.inference.engines.ollama_engine import get_response_ollama
from miniml.inference.engines.openrouter_engine import get_response_openrouter

from miniml.tools.toolreg import ToolRegistry, Tool, get_builtin_tools
from miniml.utils.utilities import format_messages, unpack_arguments, is_required_parameter, get_callable_args      # type: ignore

_builtin_tools_names = get_builtin_tools()

import yaml
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
class AgentGetRefs(AgentRunnable):
    INFERENCE_TRIES_LIMIT: int = 10

    PROMPT_PATHS = {
        "prompts_validation":    "miniml/prompts/validation.yaml",
        "prompts_generation":    "miniml/prompts/generation.yaml",
    }

    def __init__(self,
        _parent_base_dir_path: str | None = None,
        column_data: pd.DataFrame | None = None,
        column: str | None = None,
        feature_type: str | None = None,
        use_validator: bool = True,
        provider: str | None = None,
        model: str | None = None,
        tools: Optional[List[Tool]] = None
    ):
        assert _parent_base_dir_path is not None, "Parent base directory path is required"
        self._parent_base_dir_path = _parent_base_dir_path

        assert column_data is not None, "Column data is required"
        self.column_data = column_data

        assert column is not None, "Column is required"
        self.column = column

        assert feature_type is not None, "Feature type is required"
        self.feature_type = feature_type

        self.use_validator = use_validator

        assert provider is not None, "Provider is required"
        self.provider = provider

        assert model is not None, "Model is required"
        self.model = model

        # build data context
        self.data_context = f"Column: {self.column}\nFeature type: {self.feature_type}"

        # tools - builtin and user defined
        self._builtin_tools: List[Tool] = [
            TOOL_REGISTRY.get_tool(name=tool_name) for tool_name in _builtin_tools_names
        ]

        if tools is None:
            tools = self._builtin_tools

        self._user_tools: List[Tool] = [
            tool for tool in tools if tool not in self._builtin_tools
        ]

        # print everything in params
        print("="*50)
        print("PARAMS")
        print(f"parent_base_dir_path: {self._parent_base_dir_path}")
        print(f"column_data: {self.column_data}")
        print(f"column: {self.column}")
        print(f"feature_type: {self.feature_type}")
        print(f"use_validator: {self.use_validator}")
        print(f"provider: {self.provider}")
        print(f"model: {self.model}")
        print("="*50)

        self.message_history = MessageHistory()

        self._base_generation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "base_generation_prompt_for_get_refs",
            inject_tools=True
        )

        self._retry_generation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "retry_generation_prompt_for_get_refs",
            inject_tools=True
        )

        self._base_validation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "base_validation_prompt_for_get_refs",
            inject_tools=True
        )

        self._retry_validation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "retry_validation_prompt_for_get_refs",
            inject_tools=True
        )

        # print the lengths of each
        print("="*50)
        print("Base generation prompt length:", len(self._base_generation_prompt))
        print("Retry generation prompt length:", len(self._retry_generation_prompt))
        print("Base validation prompt length:", len(self._base_validation_prompt))
        print("Retry validation prompt length:", len(self._retry_validation_prompt))
        print("="*50)

    def _run_agent_loop(self) -> Optional[str] | Any:
        tries = 0
        _first_instance: Dict[str, Any] | None = None

        while tries < self.INFERENCE_TRIES_LIMIT:
            tries += 1
            is_retry = tries > 1

            _dbg("LOOP", f"Iteration {tries}/{self.INFERENCE_TRIES_LIMIT}", f"is_retry={is_retry}")

            # ############################################################################
            # GENERATION
            # ############################################################################

            if is_retry:
                retry_context = self._retry_generation_prompt.replace(
                    "[HISTORY]",
                    self.message_history.get_history_string()
                )
            else:
                retry_context = ""

            current_gen_prompt = self._base_generation_prompt.replace(
                "[AGENT_MEMORY]",
                retry_context
            )

            current_gen_prompt = current_gen_prompt.replace(
                "[DATA]",
                str(self.column_data)
            )

            current_gen_prompt = current_gen_prompt.replace(
                "[CONTEXT]",
                self.data_context
            )

            _dbg("[GENERATED PROMPT FINAL]", current_gen_prompt)

            response_generated: LLMSingleResponse = self._generate(
                prompt=current_gen_prompt
            )

            _dbg("RESPONSE", str(response_generated))

            _gen_parser: AgentGetRefsGenerationParser = AgentGetRefsGenerationParser(
                response=response_generated.content if type(response_generated.content) is str else ""
            )

            _parsed_data: AgentParserResponse = _gen_parser.parse()

            if _parsed_data.instance_ == "error":
                assert _parsed_data.error is not None

                self._record_error(
                    stage="get-refs-generation",
                    task="generation",
                    exc=_parsed_data.error if type(_parsed_data.error) is Exception else Exception(_parsed_data.error)  # type: ignore
                )
                _dbg("ERROR", _parsed_data.error)
                continue

            _dbg("PARSED_DATA", str(_parsed_data))

            # FIX 1: capture _first_instance as soon as we have a valid parse,
            # regardless of whether the validator is on
            if _first_instance is None:
                _first_instance = {
                    "_instance": "GetRefsEngineFrame",
                    "data": GetRefsEngineFrame(
                        column_name=self.column,
                        column_type=self.feature_type,
                        imputation_strategy=(
                            _parsed_data.data.imputation_strategy or ""
                            if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                            else ""
                        ),
                        imputation_reasoning=(
                            _parsed_data.data.imputation_reasoning or ""
                            if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                            else ""
                        ),
                        outlier_strategy=(_parsed_data.data.outlier_strategy or ""
                            if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                            else ""
                        ),
                        outlier_reasoning=(_parsed_data.data.outlier_reasoning or ""
                            if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                            else ""
                        )
                    )
                }

            if not self.use_validator:
                return _first_instance

            # ############################################################################
            # VALIDATION
            # ############################################################################

            if is_retry:
                retry_context = self._retry_validation_prompt.replace(
                    "[FEEDBACK]",
                    self.message_history.get_history_string()
                )
            else:
                retry_context = ""

            current_val_prompt = self._base_validation_prompt.replace(
                "[AGENT_MEMORY]",
                retry_context
            )

            current_val_prompt = current_val_prompt.replace(
               "[DATA]",
               str(self.column_data)
            )

            current_val_prompt = current_val_prompt.replace(
                "[CONTEXT]",
                self.data_context
            )

            current_val_prompt = current_val_prompt.replace(
                "[OUTPUT]", response_generated.content if type(response_generated.content) is str else ""
            )

            _dbg("[VALIDATION PROMPT OUTPUT]", current_val_prompt)

            response_validated: LLMSingleResponse = self._validate(
                prompt=current_val_prompt
            )

            _dbg("VALIDATION", str(response_validated))

            _val_parser: AgentGetRefsValidationParser = AgentGetRefsValidationParser(
                response=response_validated.content if type(response_validated.content) is str else ""
            )
            _val_result: AgentParserResponse = _val_parser.parse()

            _dbg("VALIDATION_RESULT [PARSED]", str(_val_result))

            if _val_result.instance_ == "error":
                prev_imputation = (
                    _parsed_data.data.imputation_strategy or ""
                    if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                    else ""
                )

                prev_outlier = (
                    _parsed_data.data.outlier_strategy or ""
                    if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                    else ""
                )

                self._record_error(
                    stage="validation",
                    task="validation",
                    exc=Exception(
                        f"{_val_result.error or ''}. "
                        f"Previous Response: {prev_imputation} and {prev_outlier}"
                    )
                )
                _dbg("ERROR", _val_result.error or "")
                continue

            if _val_result.instance_ == "success":
                if _val_result.valid is False:
                    prev_imputation = (
                        _parsed_data.data.imputation_strategy or ""
                        if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                        else ""
                    )

                    prev_outlier = (
                        _parsed_data.data.outlier_strategy or ""
                        if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                        else ""
                    )

                    self._record(
                        stage="get-refs-validation",
                        task="validation",
                        data=(
                            f"{_val_result.reasoning or ''} "
                            f"Previous Response: {prev_imputation} and {prev_outlier}"
                        )
                    )
                else:
                    return {
                        "_instance": "GetRefsEngineFrame",
                        "data": GetRefsEngineFrame(
                            column_name=self.column,
                            column_type=self.feature_type,
                            imputation_strategy=(
                                _parsed_data.data.imputation_strategy or ""
                                if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                                else ""
                            ),
                            imputation_reasoning=(
                                _parsed_data.data.imputation_reasoning or ""
                                if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                                else ""
                            ),
                            outlier_strategy=(_parsed_data.data.outlier_strategy or ""
                                if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                                else ""
                            ),
                            outlier_reasoning=(_parsed_data.data.outlier_reasoning or ""
                                if isinstance(_parsed_data.data, AgentGetRefsIOResponse)
                                else ""
                            )
                        )
                    }

        # retries exhausted — return the first valid result we captured, else None
        return _first_instance


    def run(self) -> Any:
        _dbg("RUN", "Starting GetRefs agent run")
        return self._run_agent_loop()

    def _call_llm(self, prompt: str, stage: str, use_tools: bool | None = None) -> LLMResponse:
        options: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 300,
        }

        _dbg("LLM", f"Calling provider '{self.provider}' for stage '{stage}'")

        if stage == "generation":
            if self.provider == "ollama":
                result = get_response_ollama(prompt=prompt, model=self.model, options_dict=options)
                message = result.get("message", {})
                content = message.get("content", "")
                raw_tool_calls: List[Any] = message.get("tool_calls") or []

            elif self.provider == "openrouter":
                result = get_response_openrouter(prompt=prompt, model=self.model, options_dict=options)
                content = result.get("choices", [])[0].get("message", {}).get("content", "")
                raw_tool_calls = result.get("choices", [])[0].get("message", {}).get("tool_calls", [])

            else:
                raise ValueError(f"Invalid provider: {self.provider}")

        elif stage == "validation":
            if self.provider == "ollama":
                result = get_response_ollama(prompt=prompt, model=self.model, options_dict=options)
                message = result.get("message", {})
                content = message.get("content", "")
                raw_tool_calls = message.get("tool_calls") or []

            elif self.provider == "openrouter":
                result = get_response_openrouter(prompt=prompt, model=self.model, options_dict=options)
                content = result.get("choices", [])[0].get("message", {}).get("content", "")
                raw_tool_calls = result.get("choices", [])[0].get("message", {}).get("tool_calls", [])

            else:
                raise ValueError(f"Invalid provider: {self.provider}")

        else:
            raise ValueError(f"Invalid stage: {stage}")

        _dbg("LLM", f"Content: {content}")
        _dbg("LLM", f"Raw tool calls: {raw_tool_calls}")

        # FIX 2: removed _record call here — raw LLM dicts were polluting the
        # feedback history fed back to the validator, causing it to loop forever

        if raw_tool_calls:
            for function in raw_tool_calls:
                name = function.get("function", {}).get("name", "")
                arguments = function.get("function", {}).get("arguments", {})
                _dbg("LLM", f"Function call: {name} with arguments: {arguments}")

                self._execute_tool(
                    tool_response={
                        "name": name,
                        "arguments": arguments,
                    }
                )

        return LLMResponse(content=content, tool_calls=raw_tool_calls)


    def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
        # FIX 3: was using undefined variable `tool_name` instead of `name`
        name = tool_response.get("name", "")
        arguments = tool_response.get("arguments", {})

        _dbg("Tool", f"Executing tool: {name} with arguments: {arguments}")

        if name.lower() not in TOOL_REGISTRY.get_tool_names():
            name, _ = self._get_most_approximate_tool(tool_name=name)
        
        arguments = unpack_arguments(data=arguments)
        _dbg("ARGUMENTS", str(arguments))

        all_args = dict(arguments)

        registered_tool: Tool = TOOL_REGISTRY.get_tool(name=name)

        filtered_args = get_callable_args(registered_tool.func, all_args)    # type: ignore

        try:
            tool_result: pd.DataFrame | Any | None = registered_tool.func(**filtered_args)    # type: ignore

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
        except:
            self._record_tool_execution(
                tool_name=name,
                arguments=arguments,
                result=f"Failed to execute tool: {name} with args: {arguments}",
            )


    def _generate(self, prompt: str, is_retry: bool = False) -> LLMSingleResponse:
        assert prompt is not None, "Prompt cannot be None"

        _llm_response: LLMResponse = self._call_llm(
            prompt=prompt,
            stage="generation"
        )

        raw_response, _ = _llm_response.content, _llm_response.tool_calls

        if raw_response is None:
            _dbg("GENERATE", "LLM returned None")
            return LLMSingleResponse(
                content=None
            )

        return LLMSingleResponse(
            content=raw_response
        )


    def _validate(self, prompt: str) -> LLMSingleResponse:
        assert prompt is not None, "Prompt cannot be None"

        _llm_val_response: LLMResponse = self._call_llm(
            prompt=prompt,
            stage="validation"
        )

        raw_response, _ = _llm_val_response.content, _llm_val_response.tool_calls

        if raw_response is None:
            _dbg("GENERATE", "LLM returned None")
            return LLMSingleResponse(
                content=None
            )

        return LLMSingleResponse(
            content=raw_response
        )

    def _record(self, stage: str, task: str, data: Any) -> None:
        _dbg("Recorder", f"Recording stage '{stage}', task '{task}', data: {data}")

        payload = str(data)

        self.message_history.add_message(
            AgentMessage(
                role="assistant",
                content=payload,
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


    def _record_error(self, stage: str, task: str, exc: Exception) -> None:
        _dbg("Recorder", f"Recording error for stage '{stage}', task '{task}', error: {exc}")
        labeled = f"[ERROR][stage={stage}][task={task}][type={type(exc).__name__}] {exc}"
        self._record(stage=stage, task=task, data=labeled)

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
            prompt = prompt.replace("[USER_TOOLS]", "--\n".join([f"TOOL: {tool.name} FUNCTION_DESCRIPTION: {str(tool.json).replace(fr"\n", " ").replace("  ", " ")}" for tool in self._user_tools]))

        return prompt


    def _get_most_approximate_tool(self, tool_name: str) -> tuple[str, str]:
        _user_tools: list[str] = [tool.name for tool in self._user_tools]
        _builtin_tools: list[str] = [tool.name for tool in self._builtin_tools]

        best_match = ""
        best_list_name = ""
        best_score = 0.0

        for value in _user_tools:
            score = SequenceMatcher(
                None,
                tool_name.lower(),
                value.lower(),
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = value
                best_list_name = "_user_tools"

        for value in _builtin_tools:
            score = SequenceMatcher(
                None,
                tool_name.lower(),
                value.lower(),
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = value
                best_list_name = "_builtin_tools"

        return best_match, best_list_name