import json
from typing import Any, Dict, List, Optional

import requests

def get_response_openrouter(
    prompt: str,
    model: str,
    options_dict: Dict[str, Any],
    openrouter_api_key: str | None = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:

    if not openrouter_api_key:
        raise ValueError("OpenRouter API key is not set")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    response_json = response.json()

    print("[OPENROUTER RESPONSE]")
    print(json.dumps(response_json, indent=2))

    return response_json