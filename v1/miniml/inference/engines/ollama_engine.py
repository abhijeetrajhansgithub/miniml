from ollama import chat as OllamaChat    # type: ignore
from typing import Dict, Any, List, Optional

def get_response_ollama(
    prompt: str,
    model: str,
    options_dict: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
    columns: Optional[List[str]] = None,
    use_columns: bool = False,
) -> Dict[Any, Any]:
    response = OllamaChat(   # type: ignore
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        tools=tools or [],
        options=options_dict,
    )

    print("[Ollama Response]", response)

    # BUG FIX 10: The ollama library returns a Pydantic model object, not a plain dict.
    # Callers downstream use `.get("message", {})` which only works on dicts.
    # `model_dump()` serialises the entire response into a plain Python dict.
    return response.model_dump()