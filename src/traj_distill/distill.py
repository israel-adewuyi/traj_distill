from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

from .client import VLLMClientError, create_chat_completion, extract_assistant_text
from .config import (
    DistillConfig,
    RequestConfig,
    ServerConfig,
    load_distill_config,
    load_server_config,
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
ROLE_KEYS = {"system", "user", "assistant", "tool"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-distill trajectories from JSON files using vLLM."
    )
    parser.add_argument(
        "--server-config",
        type=Path,
        required=True,
        help="Path to server config TOML.",
    )
    parser.add_argument(
        "--distill-config",
        type=Path,
        required=True,
        help="Path to distillation config TOML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and run model calls, but do not write output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files even if they already exist.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first file that fails.",
    )
    return parser


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if isinstance(raw_messages, list):
        return [
            _normalize_message(item, f"messages[{index}]")
            for index, item in enumerate(raw_messages)
        ]

    if isinstance(raw_messages, dict):
        if "role" in raw_messages and "content" in raw_messages:
            return [_normalize_message(raw_messages, "messages")]

        messages: list[dict[str, str]] = []
        for key, value in _ordered_mapping_items(raw_messages):
            key_text = str(key)

            # Common compact shape: {"system": "...", "user": "...", ...}
            if isinstance(value, str) and key_text in ROLE_KEYS:
                messages.append({"role": key_text, "content": value})
                continue

            if isinstance(value, dict):
                messages.append(_normalize_message(value, f"messages[{key_text!r}]"))
                continue

            raise ValueError(
                f"messages mapping entry {key_text!r} must be a message table "
                "or role->string pair."
            )

        if not messages:
            raise ValueError("messages must not be empty.")
        return messages

    raise ValueError("messages must be a list or mapping of messages.")


def _ordered_mapping_items(mapping: dict[Any, Any]) -> list[tuple[Any, Any]]:
    items = list(mapping.items())
    if all(_is_int_like_key(key) for key, _ in items):
        return sorted(items, key=lambda pair: int(pair[0]))
    return items


def _is_int_like_key(value: Any) -> bool:
    if isinstance(value, int):
        return True
    return isinstance(value, str) and value.isdigit()


def _normalize_message(raw_message: Any, location: str) -> dict[str, str]:
    if not isinstance(raw_message, dict):
        raise ValueError(f"{location} must be a table with `role` and `content`.")

    role = raw_message.get("role")
    content = raw_message.get("content")

    if not isinstance(role, str) or not role.strip():
        raise ValueError(f"{location}.role must be a non-empty string.")
    if not isinstance(content, str):
        raise ValueError(f"{location}.content must be a string.")

    return {"role": role.strip(), "content": content}


def estimate_token_count(messages: list[dict[str, str]]) -> int:
    # Approximate count used for local guardrails; exact tokenization is model-specific.
    return sum(len(TOKEN_PATTERN.findall(message["content"])) for message in messages)


def discover_input_files(input_glob: str) -> list[Path]:
    matches = [Path(path).resolve() for path in glob.glob(input_glob, recursive=True)]
    json_files = [
        path for path in matches if path.is_file() and path.suffix.lower() == ".json"
    ]
    return sorted(set(json_files))


def infer_input_root(input_glob: str) -> Path:
    wildcard_match = re.search(r"[*?\[]", input_glob)
    static_prefix = (
        input_glob if wildcard_match is None else input_glob[: wildcard_match.start()]
    )
    root = Path(static_prefix)
    if root.suffix:
        root = root.parent
    if not root.as_posix().strip():
        return Path.cwd().resolve()
    return root.resolve()


def build_distill_prompt(
    prompt_instructions: str,
    token_cap: int,
    source_messages: list[dict[str, str]],
    previous_estimate: int | None,
) -> list[dict[str, str]]:
    request_lines = [
        "Distill the trajectory while preserving all invariants from the system instructions.",
        f"Target cap: <= {token_cap} tokens across all `content` values in output `messages`.",
        "Return only JSON in this shape: {\"messages\": [{\"role\": \"...\", \"content\": \"...\"}]}",
        "Do not include markdown fences and do not include extra keys.",
    ]
    if previous_estimate is not None:
        request_lines.append(
            f"Your previous output was about {previous_estimate} tokens; compress it further."
        )

    source_block = json.dumps({"messages": source_messages}, indent=2, ensure_ascii=True)
    request_lines.append("")
    request_lines.append("Trajectory to distill:")
    request_lines.append(source_block)

    return [
        {"role": "system", "content": prompt_instructions},
        {"role": "user", "content": "\n".join(request_lines)},
    ]


def parse_distilled_messages(assistant_text: str) -> list[dict[str, str]]:
    decoded = _load_json_from_text(assistant_text)
    if not isinstance(decoded, dict):
        raise ValueError("distillation output must be a JSON object.")
    if "messages" not in decoded:
        raise ValueError("distillation output must include `messages`.")
    return normalize_messages(decoded["messages"])


def _load_json_from_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fences(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("model response did not contain valid JSON.")
        snippet = cleaned[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise ValueError("model response contained malformed JSON.") from exc


def _strip_code_fences(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def create_chat_completion_with_retries(
    server_config: ServerConfig,
    request_config: RequestConfig,
    retry_count: int,
) -> dict[str, Any]:
    last_exc: VLLMClientError | None = None
    for _ in range(retry_count + 1):
        try:
            return create_chat_completion(server_config, request_config)
        except VLLMClientError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable retry state")


def distill_messages_once(
    source_messages: list[dict[str, str]],
    server_config: ServerConfig,
    distill_config: DistillConfig,
    prompt_instructions: str,
    previous_estimate: int | None,
) -> list[dict[str, str]]:
    options = dict(distill_config.options)
    # Server-side cap for completion length; prompt still carries the trajectory cap request.
    options.setdefault("max_tokens", distill_config.token_cap)

    request_config = RequestConfig(
        model=distill_config.model,
        messages=build_distill_prompt(
            prompt_instructions=prompt_instructions,
            token_cap=distill_config.token_cap,
            source_messages=source_messages,
            previous_estimate=previous_estimate,
        ),
        options=options,
    )

    response = create_chat_completion_with_retries(
        server_config=server_config,
        request_config=request_config,
        retry_count=distill_config.retry_count,
    )
    assistant_text = extract_assistant_text(response)
    return parse_distilled_messages(assistant_text)


def distill_with_cap(
    source_messages: list[dict[str, str]],
    server_config: ServerConfig,
    distill_config: DistillConfig,
    prompt_instructions: str,
) -> list[dict[str, str]]:
    working_messages = source_messages
    previous_estimate: int | None = None

    # Extra compression passes let the model tighten output when first pass misses cap.
    for _ in range(distill_config.max_compression_passes):
        distilled = distill_messages_once(
            source_messages=working_messages,
            server_config=server_config,
            distill_config=distill_config,
            prompt_instructions=prompt_instructions,
            previous_estimate=previous_estimate,
        )
        estimate = estimate_token_count(distilled)
        if estimate <= distill_config.token_cap:
            return distilled
        working_messages = distilled
        previous_estimate = estimate

    raise ValueError(
        f"distilled trajectory exceeded token cap after "
        f"{distill_config.max_compression_passes} pass(es)."
    )


def distill_payload(
    payload: Any,
    server_config: ServerConfig,
    distill_config: DistillConfig,
    prompt_instructions: str,
) -> Any:
    if isinstance(payload, dict):
        if "messages" not in payload:
            raise ValueError("input JSON object must include `messages`.")
        output = dict(payload)
        source_messages = normalize_messages(payload["messages"])
        output["messages"] = distill_with_cap(
            source_messages=source_messages,
            server_config=server_config,
            distill_config=distill_config,
            prompt_instructions=prompt_instructions,
        )
        return output

    if isinstance(payload, list):
        output_items: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"input JSON list item {index} must be an object.")
            if "messages" not in item:
                raise ValueError(f"input JSON list item {index} is missing `messages`.")

            output_item = dict(item)
            source_messages = normalize_messages(item["messages"])
            output_item["messages"] = distill_with_cap(
                source_messages=source_messages,
                server_config=server_config,
                distill_config=distill_config,
                prompt_instructions=prompt_instructions,
            )
            output_items.append(output_item)
        return output_items

    raise ValueError("input JSON must be an object or a list of objects.")


def run(
    server_config: ServerConfig,
    distill_config: DistillConfig,
    dry_run: bool,
    overwrite: bool,
    fail_fast: bool,
) -> int:
    prompt_instructions = distill_config.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_instructions:
        raise ValueError("prompt file must not be empty.")

    files = discover_input_files(distill_config.input_glob)
    if not files:
        raise ValueError("no JSON files matched `input_glob`.")

    input_root = infer_input_root(distill_config.input_glob)
    overwrite_enabled = overwrite or distill_config.overwrite

    processed = 0
    succeeded = 0
    failed = 0
    skipped = 0

    print(f"found {len(files)} input file(s)")
    for index, input_path in enumerate(files, start=1):
        processed += 1
        try:
            relative_path = input_path.relative_to(input_root)
        except ValueError:
            relative_path = Path(input_path.name)

        output_path = distill_config.output_dir / relative_path
        if output_path.exists() and not overwrite_enabled:
            skipped += 1
            print(f"[{index}/{len(files)}] skipped {input_path} (output exists)")
            continue

        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            distilled_payload = distill_payload(
                payload=payload,
                server_config=server_config,
                distill_config=distill_config,
                prompt_instructions=prompt_instructions,
            )
            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(distilled_payload, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
            succeeded += 1
            print(f"[{index}/{len(files)}] ok {input_path} -> {output_path}")
        except Exception as exc:  # broad by design for per-file batch isolation
            failed += 1
            print(f"[{index}/{len(files)}] error {input_path}: {exc}")
            should_stop = fail_fast or not distill_config.continue_on_error
            if should_stop:
                break

    print(
        "summary: "
        f"processed={processed} succeeded={succeeded} failed={failed} skipped={skipped}"
    )
    return 1 if failed else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        server_config = load_server_config(args.server_config)
        distill_config = load_distill_config(args.distill_config)
        return run(
            server_config=server_config,
            distill_config=distill_config,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
        )
    except (OSError, ValueError, VLLMClientError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
