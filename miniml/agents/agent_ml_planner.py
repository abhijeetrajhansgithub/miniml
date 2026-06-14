from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

from pandas.core.common import not_none

from miniml.agents.agent import AgentRunnable
from miniml._messages.messages import MessageHistory, MessageRole, MessageType, BaseMessage, AgentMessage
from miniml.inference.engines.engine_frame import DatasetContext

from miniml.inference.engines.ollama_engine import get_response_ollama
from miniml.inference.engines.openrouter_engine import get_response_openrouter

from miniml.tools.toolreg import ToolRegistry, Tool, get_builtin_tools
from miniml.ml.models import _MODELS

_models_globals: dict[str, dict[str, Any]] = _MODELS

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
        use_llm: bool = True
        **kwargs
    ):
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
        self.models_used = []
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
            self.metrics = ["rmse", "mae"]
        elif self.problem_type == "classification":
            self.metrics = ["a", "r", "f1"]

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
        if self.ml_models is None or (isinstance(self.ml_models, list) and len(self.ml_models) == 0):
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


            

    
    def _run_sub_agent(self, sub_agent_task):
        tries = 0

        if sub_agent_task == "model-selection":
            _base_generation_prompt = self._base_generation_prompt_for_ml_planner_model_selection
            _retry_generation_prompt = self._retry_generation_prompt_for_ml_planner_model_selection

        
        # common loop
        while tries < self.INFERENCE_TRIES_LIMIT:
            tries += 1
            is_retry = tries > 1

            _dbg("SUB-AGENT LOOP", f"Iteration {tries}/{self.INFERENCE_TRIES_LIMIT}", f"is_retry={is_retry}")

            if is_retry:
                retry_context = self._retry_generation_prompt_for_ml_planner_model_selection.replace(
                    "HISTORY",
                    self.message_history.get_by_stage_and_task(
                        stage="ml_modelling",
                        task="ml_model_selection"
                    )
                )
            else:
                retry_context = ""

            _current_gen_prompt = _base_generation_prompt.replace(
                "[AGENT_MEMORY]",
                retry_context
            )

            _current_gen_prompt = _current_gen_prompt.replace(
                "[DATA]",
                self.data_context_json
            )

            _dbg("[SA - GENERATED FINAL PROMPT]", _current_gen_prompt)

            response_generated = self._generate(
                prompt=_current_gen_prompt,
                sub_agent_task="model-selection"
            )

            _dbg("RESPONSE", response_generated)
        
    
    def _choose_models(self):
        pass 

    def _choose_hyperparams(self):
        pass 


        

    
    def _run_agent_loop(self) -> Optional[str] | Any:
        tries = 0
        while tries < self.INFERENCE_TRIES_LIMIT:

            tries += 1

            # SUB-AGENT 1: Run ML model selector
            self._run_sub_agent("model-selection")
        pass 

    def run(self) -> Any:
        pass 

    def _call_llm(self, prompt: str, stage: str) -> Optional[str] | Dict[str, Any]:
        pass 

    def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
        pass 

    def _generate(self, prompt: str, sub_agent_task: str) -> Optional[Dict[str, Any]] | str:
        pass 

    def _validate(self, generated_response: str) -> Optional[Dict[str, Any]] | str:
        pass
    
    def _get_available_models(self) -> List[str]:
        return list(_models_globals.keys())
    
    def _get_model(problem_type: str, model_name: str):
        return _models_globals[problem_type][model_name]

    def _record(self, stage: str, task: str, data: Any) -> None:
        pass 

    def _record_error(self, stage: str, task: str, exc: Exception) -> None:
        pass
    
    def _get_most_approximate_tool(self, tool_name: str) -> tuple[str, str]:
        pass
    