import os
from typing import Any, Dict, List, Optional, Literal, Callable   # type: ignore
import configparser
import pandas as pd                             # type: ignore
import yaml

from miniml.errors.errors import ConfigFileError, ToolError
from miniml.utils.utilities import get_encoding, get_delimiter
from miniml._memory._memory import MemoryProfile, ActiveMemory
from miniml.tools.toolreg import Tool, ToolRegistry
from miniml.inference.engines.engine_frame import DatasetContext

# load builtins 
from miniml.utils._load_builtin_tools import load_tools

load_tools()

from miniml.inference.engines.engine_frame import GetRefsEngineFrame
from miniml.preprocessing.preprocessing import DataPreprocessing
from miniml.agents.agent_ml_planner import AgentMLPlanner

# Base directory
PARENT_BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("PARENT_BASE_DIR_PATH:", PARENT_BASE_DIR_PATH)

# ----- initialize active memory -------
_active_memory = ActiveMemory()

# ----- Tool Registry ------
TOOL_REGISTRY = ToolRegistry()

def get_list(value: str):
    
    return [v.strip() for v in value.split(",")]


def get_tuple(value: str):
    return tuple(map(int, value.split(":")))


def stringify(data: str | Dict[str, Any]) -> str:
    return str(data)


class MiniML:
    def __init__(self, 
                 config_file_path: str, 
                 auto: bool = False,
                 target_column: str | None = None,
                 verbose: bool = False,
                 hard_verbose: bool = False,
                 save_csv: bool = False,
                 tools: Optional[List[Callable[..., Any]]] = None,  
    ) -> None:
        self.verbose = verbose
        self.hard_verbose = hard_verbose
        self.save_csv = save_csv
        self.target_column = target_column
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.ToolList: List[Tool] = []

        if tools is not None:
            for tool in tools:
                tool_obj = TOOL_REGISTRY.get_tool(name=tool.__name__)
                print(tool_obj)

                self.tools[tool_obj.name] = tool
                self.ToolList.append(tool_obj)
        

        for tool in self.ToolList:
            self._validate_tool(tool)

        
        self.auto = auto

        if self.auto:
            self.fn_auto()
        else:
            ext = os.path.splitext(config_file_path)[1].lower().lstrip(".")

            if ext not in {"yaml", "yml", "cfg"}:
                raise ValueError(f"Unsupported config type: {ext}")

            self.file_type: Literal["yaml", "yml", "cfg"] | str = ext

            self.config_file_path = os.path.abspath(config_file_path)

            if not os.path.isfile(self.config_file_path):
                raise ConfigFileError(f"Config file {self.config_file_path} does not exist")

            print(f"Config file path: {self.config_file_path}")

            self.config = self.load_configs(_type=self.file_type)
            self.parsed_config = self.parse_config(_type=self.file_type)

            print(f"Parsed Config: {self.parsed_config}")

            # ----------------- extract data -------------------
            self.data_path: str = self.parsed_config.get("data", {}).get("dataset", "")
            self.use_llm_for_data_inference: bool = self.parsed_config.get("data", {}).get("use_llm_for_data_inference", False)
            self.delimiter: str | None = self.parsed_config.get("data", {}).get("delimiter", None)
            self.encoding: str | None = self.parsed_config.get("data", {}).get("encoding", None)
            self.data_modelling_type: str = self.parsed_config.get("data", {}).get("type", "")
            self.do_eda: bool = self.parsed_config.get("data", {}).get("eda", False)
            self.sheet_name: str | None = self.parsed_config.get("data", {}).get("sheet_name", None)

            self.output_path: str = self.parsed_config.get("output", {}).get("path", "")

            self.llm_provider: str = self.parsed_config.get("llm", {}).get("provider", "")
            self.llm_model: str = self.parsed_config.get("llm", {}).get("model", "")
            self.use_llm: bool = self.parsed_config.get("llm", {}).get("use_llm", False)

            self.tts_split: str = self.parsed_config.get("tts", {}).get("split", "80:20")

            self.ml_models: list[Any] = self.parsed_config.get("ml", {}).get("models", [])

            self.metrics: list[Any] = self.parsed_config.get("evaluation", {}).get("metrics", [])

            self.iters: int = self.parsed_config.get("training", {}).get("iters", 10)
            self.use_cv: bool = self.parsed_config.get("training", {}).get("use_cross_validation", False)
            self.cv_params: dict[str, Any] = self.parsed_config.get("training", {}).get("cv_params", {})


            # ----------------- fetch full data path -------------------
            if not os.path.isabs(self.data_path):
                self.data_path = os.path.abspath(self.data_path)


            # ----------------- fetch delimiter -------------------
            if self.delimiter is None:
                self.delimiter = get_delimiter(self.data_path)

            # ----------------- fetch encoding -------------------
            if self.encoding is None:
                self.encoding = get_encoding(self.data_path)


            assert self.data_modelling_type in ["classification", "regression"], "Data modelling type must be either classification or regression"
            assert self.use_llm_for_data_inference is True or self.use_llm_for_data_inference is False, "Use LLM for data inference must be either True or False"
            assert self.delimiter is not None, "Delimiter cannot be None"
            assert self.encoding is not None, "Encoding cannot be None"

            # print all config params extracted
            print("=" * 100)
            print("All config params extracted:")
            print(f"  - Data path: {self.data_path}")
            print(f"  - Use LLM for data inference: {self.use_llm_for_data_inference}")
            print(f"  - Delimiter: {self.delimiter}")
            print(f"  - Encoding: {self.encoding}")
            print(f"  - Data modelling type: {self.data_modelling_type}")
            print(f"  - Do EDA: {self.do_eda}")
            print(f"  - Sheet name: {self.sheet_name}")
            print(f"  - Output path: {self.output_path}")
            print(f"  - LLM provider: {self.llm_provider}")
            print(f"  - LLM model: {self.llm_model}")
            print(f"  - Use LLM: {self.use_llm}")
            print(f"  - TTS split: {self.tts_split}")
            print(f"  - ML models: {self.ml_models}")
            print(f"  - Metrics: {self.metrics}")
            print(f"  - Iters: {self.iters}")
            print(f"  - Use CV: {self.use_cv}")
            print(f"  - CV params: {self.cv_params}")
            print("=" * 100)

            # initialize preprocessing
            self.preprocessing = DataPreprocessing(
                data_file_path=self.data_path,
                delimiter=self.delimiter,
                encoding=self.encoding,
                sheet_name=self.sheet_name if self.sheet_name else 1,
                provider=self.llm_provider,
                model=self.llm_model,
                use_llm=self.use_llm,
                target_column=self.target_column if self.target_column else None,
                output_path=self.output_path if self.output_path else None,
                tools=self.ToolList,
                _parent_base_dir_path=PARENT_BASE_DIR_PATH,
            )

            _preprocessed_data: pd.DataFrame = self.preprocessing.get_modified_data()

            dataContx: DatasetContext = DatasetContext(
                n_rows=_preprocessed_data.shape[0],
                n_cols=_preprocessed_data.shape[1],
                target=self.preprocessing.get_target(),
                problem_type=self.data_modelling_type,
                numeric_cols=self.preprocessing.get_column_types()["numeric"],
                categorical_cols=self.preprocessing.get_column_types()["categorical"],
                class_imbalance=None,
                transformations_applied=self.preprocessing.get_transformations_applied(),
                tts=self.tts_split,
                ml_models=self.ml_models,
            )

            ml_planner_agent = AgentMLPlanner(
                _parent_base_dir_path=PARENT_BASE_DIR_PATH,
                data_context=dataContx,
                usp_models=self.ml_models,
                usp_metrics=self.metrics,
                iters=self.iters,
                use_cross_validation=self.use_cv,
                use_llm=self.use_llm,
                model=self.llm_model,
                provider=self.llm_provider,
            )

            ml_planner_agent.run()
    
    def _validate_tool(self, tool: Tool):
        if tool.name is None or len(tool.name) == 0:               # type: ignore
            raise ToolError("Tool name is required")

        if tool.docstring is None or len(tool.docstring) == 0:     # type: ignore 
            raise ToolError("Tool docstring is required")

        if tool.signature is None or len(tool.signature) == 0:     # type: ignore
            raise ToolError("Tool signature is required")

        if tool.func is None or not callable(tool.func):           # type: ignore
            raise ToolError("Tool function is required")
        
        if not tool.name.startswith("fn_tool_"):
            raise ToolError("Tool name must start with 'fn_tool_'")


    def load_configs(
        self,
        _type: Literal["cfg", "yaml", "yml"] | str
    ) -> configparser.ConfigParser | Dict[str, Any] | None:

        if _type == "cfg":
            config = configparser.ConfigParser()
            config.read(self.config_file_path)
            return config

        elif _type in {"yaml", "yml"}:
            with open(self.config_file_path, "r") as f:
                config = yaml.safe_load(f)

            return config


    def parse_config(self, 
        _type: str
    ) -> Dict[str, Any]:
        if self.config is None:
            raise ConfigFileError("Config file is not valid")
        
        config: configparser.ConfigParser | Dict[str, Any] = self.config

        print("Type of config:", type(config))
        if _type.lower() in {"yaml", "yml"} and type(config) == dict:
            parsed: Dict[str, Any] = config

            return parsed

        if _type == 'cfg' and type(config) == configparser.ConfigParser:
            try:
                parsed: Dict[str, Any] = {
                    "data": {
                        "dataset": config["data"]["dataset"],
                        "use_llm_for_data_inference": config.getboolean(
                            "data", "use_llm_for_data_inference", fallback=False
                        ),
                        "delimiter": config.get("data", "delimiter", fallback=None),
                        "encoding": config.get("data", "encoding", fallback=None),
                        "eda": config.getboolean("data", "eda", fallback=False),
                        "sheet_name": config.get("data", "sheet_name", fallback=None),
                        "type": config["data"]["type"],
                    },
                    "output": {
                        "path": config["output"]["path"],
                    },
                    "llm": {
                        "provider": config.get("llm", "provider", fallback=None),
                        "model": config.get("llm", "model", fallback=None),
                        "use_llm": config.getboolean("llm", "use_llm", fallback=False),
                    },
                    "tts": {
                        "split": get_tuple(config.get("tts", "split", fallback="80:20")),
                    },
                    "ml": {
                        "models": get_list(config["ml"]["models"]),
                    },
                    "evaluation": {
                        "metrics": get_list(config["evaluation"]["metrics"]),
                    },
                    "training": {
                        "iters": config.getint("training", "iters"),
                        "use_cross_validation": config.getboolean("training", "use_cross_validation"),
                        "cv_params": {
                            "k_folds": config.getint(
                                "training.cv_params", "k_folds", fallback=5
                            )
                        },
                    },
                }

                print("Parsed Config:", parsed)
                return parsed

            except KeyError as e:
                raise ConfigFileError(f"Missing config section/key: {e}")

            except ValueError as e:
                raise ConfigFileError(f"Invalid value in config: {e}")

        else:
            raise ValueError("Invalid config type")
    

    def fn_auto(self):
        pass

