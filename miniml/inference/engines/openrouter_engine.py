import json
import time
from typing import Any, Dict

import requests


def get_response_openrouter(
    prompt: str,
    model: str,
    options_dict: Dict[str, Any],
    openrouter_api_key: str | None = None,
) -> str:
    """
    Generate a response using OpenRouter Chat Completions API.

    Features:
    - Automatic retries
    - Exponential backoff
    - Proper 429 handling
    - Timeout handling
    """

    if not openrouter_api_key:
        raise ValueError("OpenRouter API key is not set")

    MAX_RETRIES = 5
    BASE_DELAY = 5

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
    }

    site_url = options_dict.get("site_url")
    site_name = options_dict.get("site_name")

    if site_url:
        headers["HTTP-Referer"] = site_url

    if site_name:
        headers["X-OpenRouter-Title"] = site_name

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        response = None

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )

            # ----------------------------
            # Success
            # ----------------------------
            if response.status_code == 200:
                response_json = response.json()

                print("[OPENROUTER RESPONSE]")
                print(json.dumps(response_json, indent=2))

                return response_json["choices"][0]["message"]["content"]

            # ----------------------------
            # Rate limit handling
            # ----------------------------
            if response.status_code == 429:
                try:
                    error_json = response.json()

                    retry_after = (
                        error_json.get("error", {})
                        .get("metadata", {})
                        .get("retry_after_seconds")
                    )

                    if retry_after is None:
                        retry_after = response.headers.get("Retry-After")

                    if retry_after is not None:
                        retry_after = int(float(retry_after))
                    else:
                        retry_after = BASE_DELAY * (2 ** (attempt - 1))

                    print(
                        f"[OPENROUTER] Rate limited (429). "
                        f"Retrying in {retry_after} seconds..."
                    )

                    if attempt < MAX_RETRIES:
                        time.sleep(retry_after)
                        continue

                except Exception:
                    pass

            # ----------------------------
            # Retryable server errors
            # ----------------------------
            if response.status_code >= 500:
                delay = BASE_DELAY * (2 ** (attempt - 1))

                print(
                    f"[OPENROUTER] Server error "
                    f"{response.status_code}. "
                    f"Retrying in {delay} seconds..."
                )

                if attempt < MAX_RETRIES:
                    time.sleep(delay)
                    continue

            # ----------------------------
            # Non-retryable errors
            # ----------------------------
            print(
                f"[OPENROUTER] HTTP {response.status_code}"
            )

            print(response.text)

            response.raise_for_status()

        except requests.exceptions.ConnectTimeout as error:
            last_error = error

            delay = BASE_DELAY * (2 ** (attempt - 1))

            print(
                f"[OPENROUTER] Connection timeout "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:
                time.sleep(delay)

        except requests.exceptions.ReadTimeout as error:
            last_error = error

            delay = BASE_DELAY * (2 ** (attempt - 1))

            print(
                f"[OPENROUTER] Read timeout "
                f"({attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:
                time.sleep(delay)

        except requests.exceptions.HTTPError as error:
            last_error = error

            print(f"[OPENROUTER] HTTP Error: {error}")

            if response is not None:
                print(response.text)

            break

        except requests.exceptions.RequestException as error:
            last_error = error

            delay = BASE_DELAY * (2 ** (attempt - 1))

            print(
                f"[OPENROUTER] Request failed: {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(delay)

        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

            print(
                f"[OPENROUTER] Failed to parse response: "
                f"{error}"
            )

            break

    raise RuntimeError(
        f"[OPENROUTER] Failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )