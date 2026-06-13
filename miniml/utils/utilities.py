from charset_normalizer import from_path                              # type: ignore
import csv
import ast
import json
import re
import math
import inspect
from typing import Any, Dict, List, Tuple, cast, Optional             # type: ignore

from miniml._memory._memory import MemoryProfile
from miniml.errors.errors import DatasetInferencingError

def get_encoding(file_path: str) -> str:
    return from_path(file_path).best().encoding                       # type: ignore


def get_delimiter(file_path: str, sample_size: int = 5000) -> str:
    with open(file_path, "r") as f:
        dialect = csv.Sniffer().sniff(f.read(sample_size))
        return dialect.delimiter

def unpack(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    if isinstance(parsed, dict):
        return {k: unpack(v) for k, v in parsed.items()}

    if isinstance(parsed, list):
        return [unpack(v) for v in parsed]

    return parsed


def unpack_arguments(data: dict[str, Any]) -> dict[str, Any]:
    return {k: unpack(v) for k, v in data.items()}

def is_required_parameter(func, param_name) -> bool:
    sig = inspect.signature(func)
    param = sig.parameters[param_name]
    return param.default is inspect.Parameter.empty


def get_callable_args(func, all_args) -> Dict[Any, Any]:
    sig = inspect.signature(func)

    return {
        name: value
        for name, value in all_args.items()
        if name in sig.parameters
    }