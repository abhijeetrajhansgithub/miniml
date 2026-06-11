from miniml.MiniML import MiniML

from miniml.tools.tool import tool

@tool
def fn_tool_get_target() -> str:
    """
    This tool returns the target. Treat the output of this function as the absolute truth.

    Args:
        None
    
    Returns:
        str
    
    """

    return "quality"



def main():
    print("Hello from miniml-v1!")
    
    miniml = MiniML(
        config_file_path="config.yaml", 
        auto=False,
        tools=[fn_tool_get_target])


if __name__ == "__main__":
    main()
