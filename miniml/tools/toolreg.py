from dataclasses import dataclass
from typing import Dict, Any, Callable, List


_builtin_tools_map_: List[str] = [
    "fn_tool_mean_imputer",
    "fn_tool_median_imputer",
    "fn_tool_mode_imputer",
    "fn_tool_ffill_imputer",
    "fn_tool_bfill_imputer",
    "fn_tool_knn_imputer",
    "fn_tool_iterative_imputer",
    "fn_tool_constant_imputer",
    "fn_tool_zero_imputer",
    "fn_tool_drop_missing",
    "fn_tool_zscore_outlier",
    "fn_tool_iqr_outlier",
    "fn_tool_log_transform_outlier",
    "fn_tool_drop_outlier",
    "fn_tool_drop_duplicates",
    "fn_tool_standardizer",
    "fn_tool_normalizer",
    "fn_tool_robust_scaler",
    "fn_tool_one_hot_encoder",
    "fn_tool_label_encoder",
    "fn_tool_target_encoder",
]

def get_builtin_tools() -> List[str]:
    return _builtin_tools_map_

@dataclass
class Tool:
    name: str
    docstring: str
    signature: str
    params: Dict[Any, Any]
    json: Dict[str, Any]
    func: Callable                # type: ignore


class ToolRegistry:
    _instance = None
    _initialized = False

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance
    
    def __init__(self) -> None:

        if not self._initialized:

            self._tools: Dict[str, Tool] = {}

            self._initialized = True
    
    def register_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def add_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Tool:
        return self._tools[name]
    
    def rm_tool(self, name: str) -> None:
        del self._tools[name]
    
    def inspect_tools(self) -> Dict[str, Tool]:
        return self._tools
    
    def clear(self) -> None:
        self._tools = {}

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())