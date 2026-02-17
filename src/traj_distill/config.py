from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class ServerConfig:
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class RequestConfig:
    model: str
    messages: list[dict[str, str]]
    options: dict[str, Any] = field(default_factory=dict)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_server_config(path: Path) -> ServerConfig:
    data = _read_toml(path)

    base_url = str(data.get("base_url", "")).strip().rstrip("/")
    if not base_url:
        raise ValueError("server config must define `base_url`.")

    api_key_value = data.get("api_key")
    api_key = None
    if isinstance(api_key_value, str) and api_key_value.strip():
        api_key = api_key_value.strip()

    timeout_seconds = float(data.get("timeout_seconds", 60.0))
    if timeout_seconds <= 0:
        raise ValueError("`timeout_seconds` must be > 0.")

    return ServerConfig(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def load_request_config(path: Path) -> RequestConfig:
    data = _read_toml(path)

    model = str(data.get("model", "")).strip()
    if not model:
        raise ValueError("request config must define `model`.")

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("request config must include a non-empty `messages` array.")

    messages: list[dict[str, str]] = []
    for index, item in enumerate(raw_messages, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"`messages[{index}]` must be a table.")

        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"`messages[{index}].role` must be a non-empty string.")
        if not isinstance(content, str):
            raise ValueError(f"`messages[{index}].content` must be a string.")

        messages.append({"role": role.strip(), "content": content})

    options = data.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("`options` must be a table when provided.")

    return RequestConfig(model=model, messages=messages, options=options)

