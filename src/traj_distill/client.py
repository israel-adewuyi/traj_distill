from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from .config import RequestConfig, ServerConfig


class VLLMClientError(RuntimeError):
    pass


def create_chat_completion(
    server_config: ServerConfig,
    request_config: RequestConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request_config.model,
        "messages": request_config.messages,
    }
    payload.update(request_config.options)

    endpoint = f"{server_config.base_url}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if server_config.api_key:
        headers["Authorization"] = f"Bearer {server_config.api_key}"

    req = request.Request(
        url=endpoint,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=server_config.timeout_seconds) as response:
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VLLMClientError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise VLLMClientError(
            f"Unable to reach vLLM server at {endpoint}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise VLLMClientError("Request to vLLM timed out.") from exc

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise VLLMClientError("vLLM response was not valid JSON.") from exc


def extract_assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VLLMClientError("No `choices` field found in response.")

    first = choices[0]
    if not isinstance(first, dict):
        raise VLLMClientError("Malformed choice in response.")

    message = first.get("message")
    if not isinstance(message, dict):
        raise VLLMClientError("No assistant `message` found in first choice.")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                text_parts.append(item["text"])
        if text_parts:
            return "".join(text_parts)

    raise VLLMClientError("Assistant text content was not found in response.")

