# from typing import Optional, Dict, Any, List
# from difflib import SequenceMatcher

# from miniml.agents.agent import AgentRunnable
# from miniml._messages.messages import MessageHistory, MessageRole, MessageType, BaseMessage, AgentMessage

# from miniml.inference.engines import GetRefsEngineFrame

# from miniml.inference.engines.ollama_engine import get_response_ollama
# from miniml.inference.engines.openrouter_engine import get_response_openrouter

# from miniml.tools.toolreg import ToolRegistry, Tool, get_builtin_tools

# _builtin_tools_names = get_builtin_tools()

# import yaml
# import json
# from pathlib import Path
# import pandas as pd

# TOOL_REGISTRY = ToolRegistry()

# # ── Debug helpers ──────────────────────────────────────────────────────────── #

# _DBG_SEP = "─" * 60

# def _dbg(section: str, label: str, value: Any = "") -> None:
#     print(f"\n{_DBG_SEP}")
#     print(f"[{section}] {label}")
#     if value != "":
#         print(value)
#     print(_DBG_SEP)


# # ── Agent class ────────────────────────────────────────────────────────────── #
# class AgentApplyRefs(AgentRunnable):
#     INFERENCE_TRIES_LIMIT: int = 10

#     PROMPT_PATHS = {
#         "prompts_validation":    "miniml/prompts/validation.yaml",
#         "prompts_generation":    "miniml/prompts/generation.yaml",
#     }

#     def __init__(self,
#         column_inference: GetRefsEngineFrame) -> None:
#         pass 

#     def _run_agent_loop(self) -> Optional[str] | Any:
#         pass 

#     def run(self) -> Any:
#         pass 

#     def _call_llm(self, prompt: str, stage: str) -> Optional[str] | Dict[str, Any]:
#         options: Dict[str, Any] = {
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "max_tokens": 300,
#         }

#         _dbg("LLM", f"Calling provider '{self.provider}' for stage '{stage}'")

#         if stage == "generation":
#             if self.provider == "ollama":
#                 result = get_response_ollama(prompt=prompt, model=self.model, options_dict=options)
#                 message = result.get("message", {})
#                 content = message.get("content", "")
#                 raw_tool_calls = message.get("tool_calls") or []

#             elif self.provider == "openrouter":
#                 result = get_response_openrouter(prompt=prompt, model=self.model, options_dict=options)
#                 content = result.get("choices", [])[0].get("message", {}).get("content", "")
#                 raw_tool_calls = result.get("choices", [])[0].get("message", {}).get("tool_calls", [])

#             else:
#                 raise ValueError(f"Invalid provider: {self.provider}")

#         elif stage == "validation":
#             if self.provider == "ollama":
#                 result = get_response_ollama(prompt=prompt, model=self.model, options_dict=options)
#                 message = result.get("message", {})
#                 content = message.get("content", "")
#                 raw_tool_calls = message.get("tool_calls") or []

#             elif self.provider == "openrouter":
#                 result = get_response_openrouter(prompt=prompt, model=self.model, options_dict=options)
#                 content = result.get("choices", [])[0].get("message", {}).get("content", "")
#                 raw_tool_calls = result.get("choices", [])[0].get("message", {}).get("tool_calls", [])

#             else:
#                 raise ValueError(f"Invalid provider: {self.provider}")

#         else:
#             raise ValueError(f"Invalid stage: {stage}")

#         _dbg("LLM", f"Content: {content}")
#         _dbg("LLM", f"Raw tool calls: {raw_tool_calls}")

#         if raw_tool_calls:
#             for function in raw_tool_calls:
#                 name = function.get("function", {}).get("name", "")
#                 arguments = function.get("function", {}).get("arguments", {})
#                 _dbg("LLM", f"Function call: {name} with arguments: {arguments}")

#                 self._execute_tool(
#                     tool_response={
#                         "name": name,
#                         "arguments": arguments,
#                     }
#                 )

#                 self._record(
#                     stage=stage,
#                     task="tool call",
#                     data=f"Tool used: {name}"
#                 )

#         return content 

#     def _execute_tool(self, tool_response: Dict[str, Any]) -> Any | None:
#         name = tool_response.get("name", "")
#         arguments = tool_response.get("arguments", {})

#         _dbg("Tool", f"Executing tool: {name} with arguments: {arguments}")

#         if name.lower() not in TOOL_REGISTRY.get_tool_names():
#             name, _ = self._get_most_approximate_tool(tool_name=name)

#         registered_tool: Tool = TOOL_REGISTRY.get_tool(name=name)

#         tool_result: pd.DataFrame | Any | None = registered_tool.func(**arguments)

#         self._record(
#             stage="tool_execution",
#             task=name,
#             data={
#                 "tool_name": name,
#                 "tool_arguments": arguments,
#                 "tool_result": tool_result if isinstance(tool_result, str) else None,
#             }
#         ) 

#     def _generate(self, prompt: str) -> Optional[Dict[str, Any]] | str:
#         pass 

#     def _validate(self, generated_response: str) -> Optional[Dict[str, Any]] | str:
#         pass 

#     def _record(self, stage: str, task: str, data: Any) -> None:
#         _dbg("Recorder", f"Recording stage '{stage}', task '{task}', data: {data}")

#         payload = str(data)

#         self.message_history.add_message(
#             AgentMessage(
#                 role="assistant",
#                 content=payload,
#             )
#         )


#     def _record_error(self, stage: str, task: str, exc: Exception) -> None:
#         _dbg("Recorder", f"Recording error for stage '{stage}', task '{task}', error: {exc}")
#         labeled = f"[ERROR][stage={stage}][task={task}][type={type(exc).__name__}] {exc}"
#         self._record(stage=stage, task=task, data=labeled)


#     def _load_prompt(
#         self,
#         yaml_key_path: str,
#         prompt_key: str,
#         inject_tools: bool = False,
#     ) -> str:
#         prompt_path = Path(self._parent_base_dir_path) / yaml_key_path

#         if not prompt_path.exists():
#             raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

#         with open(prompt_path, "r") as f:
#             prompts = yaml.safe_load(f)

#         prompt = prompts[prompt_key]

#         if inject_tools:
#             # user tools
#             prompt = prompt.replace("[USER_TOOLS]", "\n".join([tool.name for tool in self._user_tools]))
#             # builtin tools
#             prompt = prompt.replace("[BUILTIN_TOOLS]", "\n".join([tool.name for tool in self._builtin_tools]))

#         return prompt

#     def _get_most_approximate_tool(self, tool_name: str) -> tuple[str, str]:
#         _user_tools: list[str] = [tool.name for tool in self._user_tools]
#         _builtin_tools: list[str] = [tool.name for tool in self._builtin_tools]

#         best_match = ""
#         best_list_name = ""
#         best_score = 0.0

#         for value in _user_tools:
#             score = SequenceMatcher(
#                 None,
#                 tool_name.lower(),
#                 value.lower(),
#             ).ratio()

#             if score > best_score:
#                 best_score = score
#                 best_match = value
#                 best_list_name = "_user_tools"

#         for value in _builtin_tools:
#             score = SequenceMatcher(
#                 None,
#                 tool_name.lower(),
#                 value.lower(),
#             ).ratio()

#             if score > best_score:
#                 best_score = score
#                 best_match = value
#                 best_list_name = "_builtin_tools"

#         return best_match, best_list_name