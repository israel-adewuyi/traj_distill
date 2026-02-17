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


@dataclass(frozen=True)
class DistillConfig:
    input_glob: str
    output_dir: Path
    prompt_file: Path
    model: str
    token_cap: int
    options: dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False
    continue_on_error: bool = True
    retry_count: int = 1
    max_compression_passes: int = 2


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


def load_distill_config(path: Path) -> DistillConfig:
    data = _read_toml(path)
    base_dir = path.parent.resolve()

    input_glob = str(data.get("input_glob", "")).strip()
    if not input_glob:
        raise ValueError("distill config must define `input_glob`.")
    input_glob_path = Path(input_glob)
    if not input_glob_path.is_absolute():
        input_glob = str((base_dir / input_glob).resolve())

    output_dir_raw = str(data.get("output_dir", "")).strip()
    if not output_dir_raw:
        raise ValueError("distill config must define `output_dir`.")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    prompt_file_raw = str(data.get("prompt_file", "")).strip()
    if not prompt_file_raw:
        raise ValueError("distill config must define `prompt_file`.")
    prompt_file = Path(prompt_file_raw)
    if not prompt_file.is_absolute():
        prompt_file = (base_dir / prompt_file).resolve()
    if not prompt_file.is_file():
        raise ValueError(f"prompt file was not found: {prompt_file}")

    model = str(data.get("model", "")).strip()
    if not model:
        raise ValueError("distill config must define `model`.")

    token_cap = int(data.get("token_cap", 0))
    if token_cap <= 0:
        raise ValueError("`token_cap` must be a positive integer.")

    options = data.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("`options` must be a table when provided.")

    overwrite = bool(data.get("overwrite", False))
    continue_on_error = bool(data.get("continue_on_error", True))
    retry_count = int(data.get("retry_count", 1))
    if retry_count < 0:
        raise ValueError("`retry_count` must be >= 0.")

    max_compression_passes = int(data.get("max_compression_passes", 2))
    if max_compression_passes < 1:
        raise ValueError("`max_compression_passes` must be >= 1.")

    return DistillConfig(
        input_glob=input_glob,
        output_dir=output_dir,
        prompt_file=prompt_file,
        model=model,
        token_cap=token_cap,
        options=options,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
        retry_count=retry_count,
        max_compression_passes=max_compression_passes,
    )

