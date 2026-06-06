import os
from typing import Any, Dict, List, Optional, Literal, Callable   # type: ignore
import configparser
import pandas as pd                             # type: ignore
import yaml

from miniml.errors.errors import ConfigFileError, ToolError
from miniml.utils.utilities import get_encoding, get_delimiter
from miniml._memory._memory import MemoryProfile, ActiveMemory
from miniml.tools.toolreg import Tool, ToolRegistry

# load builtins 
from miniml.utils._load_builtin_tools import load_tools

load_tools()

from miniml.inference.engines.engine_frame import (
    ProviderEngineFrame
)

# Base directory
PARENT_BASE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PARENT_BASE_DIR_PATH)

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
    pass