<<<<<<< HEAD
# traj_distill
=======
# traj_distill

Minimal scaffold for sending chat messages to a vLLM server and getting the response back.

## Run

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
>>>>>>> b0793be (initial commit)
