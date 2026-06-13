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

    # BUG FIX 11: `headers` was used below but never defined — NameError at runtime.
    # The Authorization header is required by the OpenRouter API.
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # BUG FIX 12: `options_dict` was accepted as a parameter but silently ignored —
        # temperature, top_p, and max_tokens were never forwarded to the API payload.
        "temperature": options_dict.get("temperature", 0.7),
        "top_p": options_dict.get("top_p", 0.9),
        "max_tokens": options_dict.get("max_tokens", 300),
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