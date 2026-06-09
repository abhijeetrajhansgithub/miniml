from typing import Optional, Dict, Any, List
from difflib import SequenceMatcher

from miniml.agents.agent import AgentRunnable
from miniml._messages.messages import MessageHistory, MessageRole, MessageType, BaseMessage, AgentMessage

from miniml.inference.engines.ollama_engine import get_response_ollama
from miniml.inference.engines.openrouter_engine import get_response_openrouter

from miniml.tools.toolreg import ToolRegistry, Tool, get_builtin_tools

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
# class AgentFindTarget(AgentRunnable):
#     INFERENCE_TRIES_LIMIT: int = 10

#     PROMPT_PATHS = {
#         "prompts_validation":    "miniml/prompts/validation.yaml",
#         "prompts_generation":    "miniml/prompts/generation.yaml",
#     }
#     pass