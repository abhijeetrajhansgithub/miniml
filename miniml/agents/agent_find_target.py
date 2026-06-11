from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

from miniml.agents.agent import AgentRunnable
from miniml._messages.messages import MessageHistory, MessageRole, MessageType, BaseMessage, AgentMessage
from miniml.inference.engines.engine_frame import FindTargetEngineFrame

from miniml.parsers.AgentFindTargetParser import AgentFindTargetGenerationParser, AgentFindTargetValidationParser

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
class AgentFindTarget(AgentRunnable):
    INFERENCE_TRIES_LIMIT: int = 10

    PROMPT_PATHS = {
        "prompts_validation":    "miniml/prompts/validation.yaml",
        "prompts_generation":    "miniml/prompts/generation.yaml",
    }
    
    def __init__(self,
        column_inferences: List[FindTargetEngineFrame],  # fix: was GetRefsEngineFrame (undefined)
        provider: str | None = None,
        model: str | None = None,
        tools: List[Tool] | None = None,
        use_validator: bool = True,
        _parent_base_dir_path: str = None
    ):
        self.column_inferences: Dict[str, Dict[str, str]] = dict()

        for elem in column_inferences:
            self.column_inferences[elem.column_name] = {
                "column_name": elem.column_name,
                "column_type": elem.column_type,
                "imputation_strategy": elem.imputation_strategy,
                "imputation_reasoning": elem.imputation_reasoning,
                "outlier_strategy": elem.outlier_strategy,
                "outlier_reasoning": elem.outlier_reasoning
            }
        
        assert provider is not None, "Provider is required"
        self.provider = provider

        assert model is not None, "Model is required"
        self.model = model

        assert _parent_base_dir_path is not None, "Parent base directory path is required"
        self._parent_base_dir_path = _parent_base_dir_path

        self.use_validator = use_validator  # fix: was never assigned, caused AttributeError

        self.column_inferences_json_string = json.dumps(self.column_inferences, indent=2)

        # print everything in params
        print("="*50)
        print("PARAMS")
        print(f"parent_base_dir_path: {self._parent_base_dir_path}")
        print(f"column_inferences: {self.column_inferences}")
        print(f"provider: {self.provider}")
        print(f"model: {self.model}")
        print("="*50)

        self.message_history = MessageHistory()

        # tools - builtin and user defined
        self._builtin_tools: List[Tool] = [
            TOOL_REGISTRY.get_tool(name=tool_name) for tool_name in _builtin_tools_names
        ]

        if tools is None:
            tools = self._builtin_tools

        self._user_tools: List[Tool] = [
            tool for tool in tools if tool not in self._builtin_tools
        ]

        # Load the prompts
        self._base_generation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "base_generation_prompt_for_target_finder",
            inject_tools=True
        )

        self._retry_generation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "retry_generation_prompt_for_target_finder",
            inject_tools=True
        )

        self._base_validation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "base_validation_prompt_for_target_finder",
            inject_tools=True
        )

        self._retry_validation_prompt = self._load_prompt(
            self.PROMPT_PATHS["prompts_validation"],
            "retry_validation_prompt_for_target_finder",
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

        TARGET_FREQ_COUNT = dict()

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
                "[CONTEXT]",
                self.column_inferences_json_string
            )

            _dbg("[GENERATED PROMPT FINAL]", current_gen_prompt)

            response_generated = self._generate(
                prompt=current_gen_prompt
            )

            _dbg("RESPONSE", response_generated)

            _gen_parser = AgentFindTargetGenerationParser(
                response=response_generated
            )

            _parsed_data: Dict[str, Any] = _gen_parser.parse()

            if _parsed_data.get("_instance") == "error":
                self._record_error(
                    stage="find-target-generation",
                    task="generation",
                    exc=_parsed_data.get("error")
                )
                _dbg("ERROR", _parsed_data.get("error"))
                continue

            _dbg("PARSED_DATA", _parsed_data)

            if not self.use_validator:
                return {
                    "_instance": "FindTargetEngineFrame",
                    "data": FindTargetEngineFrame(
                        target_column=_parsed_data.get("target_column")
                    )
                }
            
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
                "[CONTEXT]",
                self.column_inferences_json_string
            )

            current_val_prompt = current_val_prompt.replace(
                "[OUTPUT]",
                response_generated
            )

            _dbg("[VALIDATION PROMPT OUTPUT]", current_val_prompt)

            response_validated = self._validate(  # fix: was passing prompt= but param was named generated_response
                prompt=current_val_prompt
            )

            _dbg("VALIDATION", response_validated)

            _val_parser = AgentFindTargetValidationParser(
                response=response_validated
            )
            _val_result = _val_parser.parse()

            _dbg("VALIDATION_RESULT [PARSED]", _val_result)

            if _val_result.get("_instance") == "error":
                self._record_error(
                    stage="validation",
                    task="validation",
                    exc=_val_result.get("error")
                )
                _dbg("ERROR", _val_result.get("error"))
                continue

            if _val_result.get("_instance") == "success":
                
                if _parsed_data.get("target_column") not in TARGET_FREQ_COUNT:
                    TARGET_FREQ_COUNT[_parsed_data.get("target_column")] = 1
                else:
                    TARGET_FREQ_COUNT[_parsed_data.get("target_column")] += 1
                    

                if _val_result.get("valid") is False:
                    self._record(
                        stage="get-refs-validation",
                        task="validation",
                        data=_val_result.get("reasoning")
                    )
                else:
                    return {
                        "_instance": "FindTargetEngineFrame",
                        "data": FindTargetEngineFrame(
                            target_column=_parsed_data.get("target_column")
                        )
                    }
        else:
            return {
                "_instance": "FindTargetEngineFrame",
                "data": FindTargetEngineFrame(
                    target_column=max(TARGET_FREQ_COUNT, key=TARGET_FREQ_COUNT.get)
                )
            }

    def run(self) -> Any:
        _dbg("RUN", "Starting FindTarget agent run")
        return self._run_agent_loop()

    def _call_llm(self, prompt: str, stage: str) -> Optional[str] | Dict[str, Any]:
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
                raw_tool_calls = message.get("tool_calls") or []

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

        return content

    def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
        name = tool_response.get("name", "")
        arguments = tool_response.get("arguments", {})

        _dbg("Tool", f"Executing tool: {name} with arguments: {arguments}")

        if name.lower() not in TOOL_REGISTRY.get_tool_names():
            name, _ = self._get_most_approximate_tool(tool_name=name)

        registered_tool: Tool = TOOL_REGISTRY.get_tool(name=name)

        tool_result: pd.DataFrame | Any | None = registered_tool.func(**arguments)

        self._record(
            stage="tool_execution",
            task=name,
            data={
                "tool_name": name,
                "tool_arguments": arguments,
                "tool_result": tool_result if isinstance(tool_result, str) else None,
            }
        )

    def _generate(self, prompt: str) -> Optional[Dict[str, Any]] | str:
        assert prompt is not None, "Prompt cannot be None"

        raw_response = self._call_llm(
            prompt=prompt,
            stage="generation"
        )

        if raw_response is None:
            _dbg("GENERATE", "LLM returned None")
            return None

        return raw_response

    def _validate(self, prompt: str) -> Optional[Dict[str, Any]] | str:  # fix: param was named generated_response but caller passed prompt=; also fix: body used undefined `prompt` instead of the parameter
        assert prompt is not None, "Prompt cannot be None"

        raw_response = self._call_llm(
            prompt=prompt,
            stage="validation"
        )

        if raw_response is None:
            _dbg("GENERATE", "LLM returned None")
            return None

        return raw_response

    def _record(self, stage: str, task: str, data: Any) -> None:
        _dbg("Recorder", f"Recording stage '{stage}', task '{task}', data: {data}")

        payload = str(data)

        self.message_history.add_message(
            AgentMessage(
                role="assistant",
                content=payload,
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