# traj_distill

Minimal scaffold for:
- sending one chat request to a vLLM OpenAI-compatible server, and
- batch-distilling trajectory `messages` from `.json` files.

## Run One Chat Request

1. Start a vLLM OpenAI-compatible server (example):

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --host 127.0.0.1 --port 8000
```

2. Send a request using TOML config files:

```bash
uv run traj-distill-chat \
  --server-config configs/server.example.toml \
  --request-config configs/request.example.toml
```

3. Print raw JSON if needed:

```bash
uv run traj-distill-chat \
  --server-config configs/server.example.toml \
  --request-config configs/request.example.toml \
  --raw
```

## Batch Distill JSON Trajectories

1. Create/update:
- `configs/server.toml` (based on `configs/server.example.toml`)
- `configs/distill.toml` (based on `configs/distill.example.toml`)
- your prompt file referenced by `prompt_file` in `distill.toml`

2. Run batch distillation:

```bash
uv run traj-distill-run \
  --server-config configs/server.toml \
  --distill-config configs/distill.toml
```

Useful flags:
- `--dry-run` validates/processes without writing output files.
- `--overwrite` overwrites existing output files.
- `--fail-fast` stops on the first failing file.
