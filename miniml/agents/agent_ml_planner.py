from typing import Optional, Dict, Any, List, override, Tuple
from difflib import SequenceMatcher
import json

from miniml.agents.agent import AgentRunnable
from miniml._messages.messages import MessageHistory, TaggedMessage
from miniml.inference.engines.engine_frame import DatasetContext, LLMResponse, LLMSingleResponse

from miniml.parsers.AgentMLPlannerParser import SubAgentModelSelectionGenerationParser
from miniml.inference.engines.ollama_engine import get_response_ollama
from miniml.inference.engines.openrouter_engine import get_response_openrouter

from miniml.tools.toolreg import ToolRegistry, Tool, get_builtin_tools
from miniml.ml.models import MODELS             # type: ignore
from miniml.utils.utilities import format_messages

_models_globals: Dict[str, Dict[str, Any]] | Dict[Any, Any] = MODELS

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
class AgentMLPlanner(AgentRunnable):
    INFERENCE_TRIES_LIMIT: int = 5

    PROMPT_PATHS = {
        "prompts_validation":    "miniml/prompts/validation.yaml",
        "prompts_generation":    "miniml/prompts/generation.yaml",
    }

    def __init__(
        self,
        _parent_base_dir_path: str | None = None,
        data_context: DatasetContext | None = None,
        usp_models: List[str] | None = None,
        usp_metrics: List[str] | None = None,
        iters: int | None = None,
        use_cross_validation: bool = False,
        use_llm: bool = True,
        tools: Optional[List[Tool]] = None,
        model: str | None = None,
        provider: str | None = None,
        **kwargs    # type: ignore
    ):
        _dbg("INSIDE", "ML Planner")

        assert _parent_base_dir_path is not None, "Parent base directory path is required"
        self._parent_base_dir_path = _parent_base_dir_path

        assert data_context is not None, "Data Context is required"
        self.data_context: DatasetContext = data_context


        # extract details from data context
        self.n_rows: int = self.data_context.n_rows
        self.n_cols: int = self.data_context.n_cols
        self.target: str = self.data_context.target
        self.problem_type: str = self.data_context.problem_type
        self.numeric_cols: List[str] = self.data_context.numeric_cols
        self.categorical_cols: List[str] = self.data_context.categorical_cols
        self.class_imbalance: int | float | None = self.data_context.class_imbalance
        self.transformations_applied: List[str] = self.data_context.transformations_applied
        self.tts: str = self.data_context.tts
        self.ml_models: List[str] = self.data_context.ml_models

        # message history
        self.message_history = MessageHistory()

        # workflow tracking
        self.models_used = [] or usp_models
        self.split = self.tts.split(":")
        
        # tts
        self.train_split_pcnt: int | None = None
        self.test_split_pcnt: int | None = None 
        self.val_split_pcnt: int | None = None

        if len(self.split) == 2:
            self.train_split_pcnt = int(self.split[0])
            self.test_split_pcnt = int(self.split[1])
        elif len(self.split) == 3:
            self.train_split_pcnt = int(self.split[0])
            self.test_split_pcnt = int(self.split[1])
            self.val_split_pcnt = int(self.split[2])
        else:
            raise ValueError("Invalid split format")

        if self.problem_type == "regression":
            self.metrics = ["rmse", "mae"] or usp_metrics
        elif self.problem_type == "classification":
            self.metrics = ["a", "r", "f1"] or usp_metrics

        # Initialize prompts
        self._base_generation_prompt_for_ml_planner_model_selection = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "base_generation_prompt_for_ml_planner_model_selection",
            inject_tools=False
        )

        self._retry_generation_prompt_for_ml_planner_model_selection = self._load_prompt(
            self.PROMPT_PATHS["prompts_generation"],
            "retry_generation_prompt_for_ml_planner_model_selection",
            inject_tools=False
        )
        
        _models_default = ["random-forest", "xgboost"]
        if self.ml_models is None or (    # type: ignore
            (isinstance(self.ml_models, list) and len(self.ml_models) == 0)    # type: ignore
        ):
            self.ml_models = _models_default

        # json build the data context
        self.data_context_json = json.dumps(
            {
                "n_rows": self.n_rows,
                "n_cols": self.n_cols,
                "target": self.target,
                "problem_type": self.problem_type,
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
                "class_imbalance": self.class_imbalance,
                "transformations_applied": self.transformations_applied,
                "tts": self.tts,
                "ml_models": self.ml_models,
            },
            indent=4
        )

        # oad the params to class init
        self.iters = iters
        self.use_cross_validation = use_cross_validation
        self.use_llm = use_llm

        # print all the params for debug
        _dbg("PARAMS", f"iters={self.iters}")
        _dbg("PARAMS", f"use_cross_validation={self.use_cross_validation}")
        _dbg("PARAMS", f"use_llm={self.use_llm}")

        # tools — builtin and user-defined
        self._builtin_tools: List[Tool] = [
            TOOL_REGISTRY.get_tool(name=tool_name) for tool_name in _builtin_tools_names
        ]

        if tools is None:
            tools = self._builtin_tools

        self._user_tools: List[Tool] = [
            tool for tool in tools if tool not in self._builtin_tools
        ]

        self.model = model
        self.provider = provider


            

    
    def _run_sub_agent(self, sub_agent_task: str, validation: bool) -> Dict[str, Any] | None:
        tries = 0

        _base_generation_prompt: str | None = None
        _retry_generation_prompt: str | None = None
        _stage: str | None = None

        if sub_agent_task == "model-selection":
            _base_generation_prompt = self._base_generation_prompt_for_ml_planner_model_selection
            _retry_generation_prompt = self._retry_generation_prompt_for_ml_planner_model_selection

            PARSER_CLASS = SubAgentModelSelectionGenerationParser
            _stage = "ml-model-selection"
        

        assert _base_generation_prompt is not None
        assert _retry_generation_prompt is not None
        assert _stage is not None

        
        # common loop
        while tries < self.INFERENCE_TRIES_LIMIT:
            tries += 1
            is_retry = tries > 1

            _dbg("SUB-AGENT LOOP", f"Iteration {tries}/{self.INFERENCE_TRIES_LIMIT}", f"is_retry={is_retry}")

            if is_retry:
                history: str = format_messages(
                    self.message_history.get_by_stage_and_task(
                        stage="ml_modelling",
                        task="ml_model_selection",
                    )
                )

                retry_context = self._retry_generation_prompt_for_ml_planner_model_selection.replace(
                    "HISTORY",
                    history,
                )
            else:
                retry_context = "None"

            _current_gen_prompt = _base_generation_prompt.replace(
                "[AGENT_MEMORY]",
                retry_context
            )

            _current_gen_prompt = _current_gen_prompt.replace(
                "[DATA]",
                self.data_context_json
            )

            if sub_agent_task == "model-selection":
                _current_gen_prompt = _current_gen_prompt.replace(
                    "[MODEL_OPTIONS]",
                    ",".join(self.ml_models)
                ) if self.ml_models is not None and isinstance(self.ml_models, list) and len(self.ml_models) > 0 else ",".join(self.ml_models)   # type: ignore 

            _dbg("[SA - GENERATED FINAL PROMPT]", _current_gen_prompt)

            response_generated = self._generate(
                prompt=_current_gen_prompt,
                sub_agent_task="model-selection"
            )

            _dbg(f" {sub_agent_task} RESPONSE", response_generated)

            if response_generated is None:
                continue

            _gen_parser = PARSER_CLASS(
                response=response_generated
            )

            _parsed_data: Dict[str, Any] = _gen_parser.parse()

            _dbg(f" {sub_agent_task} PARSED DATA", _parsed_data)

            if _parsed_data.get("_instance") == "error":
                self._record_error(
                    stage=_stage,
                    task="generation",
                    exc=_parsed_data.get("error")
                )
                _dbg("ERROR", _parsed_data.get("error"))
                continue

            _dbg(f"{_stage}: PARSED_DATA", _parsed_data)

            if not validation:
                return {
                    "_instance": "success",
                    "stage": _stage,
                    "data": _parsed_data
                }
  

        

    
    def _run_agent_loop(self) -> Optional[str] | Any:
        _USE_LLM: bool = self.use_llm

        tries = 0
        while tries < self.INFERENCE_TRIES_LIMIT:

            tries += 1

            if _USE_LLM:
                # SUB-AGENT 1: Run ML model selector
                sub_agent_data = self._run_sub_agent(
                    sub_agent_task="model-selection",
                    validation=False
                )

                _dbg("SUB-AGENT DATA", str(sub_agent_data))
                break
        pass 

    def run(self) -> Any:
        _dbg("RUN", "Starting MLPlanner agent run")
        return self._run_agent_loop() or None

    def _call_llm(self, prompt: str, stage: str, use_tools: bool | None = False) -> LLMResponse:

        assert self.model is not None 
        assert self.provider is not None

        assert prompt is not None
        assert stage is not None
        assert use_tools is not None


        options: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 300,
        }

        _tools: List[Any] = []
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
            raw_tool_calls: List[Any] = message.get("tool_calls") or []

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

        return LLMResponse(
            content=content,
            tool_calls=raw_tool_calls
        )

        # return content, raw_tool_calls

    def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
        pass 

    @override
    def _generate(self, prompt: str) -> LLMSingleResponse:
        assert prompt is not None, "Prompt cannot be None"

        _llm_response: LLMResponse = self._call_llm(
            prompt=prompt,
            stage="generation"
        )

        raw_response, raw_tool_calls = _llm_response.content, _llm_response.tool_calls

        if raw_response is None:
            _dbg("GENERATE", "LLM returned None")

            return LLMSingleResponse(
                content=None
            )
    
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

        return LLMSingleResponse(
            content=raw_response
        )
            


    def _validate(self, generated_response: str) -> LLMSingleResponse:
        pass
    
    def _get_available_models(self) -> List[str]:
        return list(_models_globals.keys())
    
    def _get_model(problem_type: str, model_name: str):
        return _models_globals[problem_type][model_name]

    def _record(self, stage: str, task: str, data: Any) -> None:
        return super()._record(stage, task, data)
    
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
    
    def _get_most_approximate_tool(self, tool_name: str) -> tuple[str, str]:
        pass
    