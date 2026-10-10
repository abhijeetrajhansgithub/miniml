import importlib

# ------------------------------------------------------------------ #
# BUILTIN TOOLS LOADER
# ------------------------------------------------------------------ #

TOOL_MODULES = [
    "miniml.utils.tools",
]

def load_tools():
    for module in TOOL_MODULES:
        importlib.import_module(module)