from typing import Dict, Any, Callable, get_origin, get_args  # type: ignore
from miniml.errors.errors import ToolError
from miniml.tools.toolreg import ToolRegistry, Tool
import inspect

TOOL_REGISTRY = ToolRegistry()


def _python_type_to_json_type(annotation: Any) -> str:
    """Convert Python type annotations to JSON schema types."""

    if annotation is str:
        return "string"

    if annotation is int:
        return "integer"

    if annotation is float:
        return "number"

    if annotation is bool:
        return "boolean"

    if annotation is list:
        return "array"

    if annotation is dict:
        return "object"

    origin = get_origin(annotation)

    if origin is list:
        return "array"

    if origin is dict:
        return "object"

    return "string"  # fallback


import inspect
from typing import Any, Callable, Dict

def tool(func: Callable[..., Any]) -> Callable[..., Any]:
    _name = func.__name__
    _docstring = func.__doc__
    _signature = str(inspect.signature(func))

    if _docstring is None:
        raise ToolError("Docstring is required")

    EXCLUDED_PARAMS = {"df", "cols"}

    _params: Dict[str, Any] = {}
    properties: Dict[str, Any] = {}
    required = []

    for param in inspect.signature(func).parameters.values():
        if param.name.lower() in EXCLUDED_PARAMS:
            continue

        _params[param.name] = param.annotation

        properties[param.name] = {
            "type": _python_type_to_json_type(param.annotation),
            "description": f"Parameter '{param.name}'"
        }

        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    _json: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": _name,
            "description": _docstring.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    _type = None
    if "imputer" in _name.lower():
        _type = "imputer" 
    elif "outlier" in _name.lower():
        _type = "outlier"
    elif "scaler" in _name.lower() or "normalizer" in _name.lower() or "encoder" in _name.lower():
        _type = "ml-prep"
    else:
        _type = "other"

    _tool = Tool(
        name=_name,
        docstring=_docstring,
        signature=_signature,
        type=_type,
        params=_params,
        json=_json,
        func=func
    )

    TOOL_REGISTRY.register_tool(_tool)

    return func