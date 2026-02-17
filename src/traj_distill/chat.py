from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import VLLMClientError, create_chat_completion, extract_assistant_text
from .config import load_request_config, load_server_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one chat request to a vLLM server using TOML configs."
    )
    parser.add_argument(
        "--server-config",
        type=Path,
        required=True,
        help="Path to server config TOML.",
    )
    parser.add_argument(
        "--request-config",
        type=Path,
        required=True,
        help="Path to request config TOML.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON response instead of assistant text.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        server = load_server_config(args.server_config)
        req = load_request_config(args.request_config)
        response = create_chat_completion(server, req)
    except (OSError, ValueError, VLLMClientError) as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    if args.raw:
        print(json.dumps(response, indent=2, ensure_ascii=True))
        return 0

    try:
        assistant_text = extract_assistant_text(response)
    except VLLMClientError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")

    print(assistant_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

