# ALICE Deployment Guide

## Option 1 — Deploy to Hugging Face Spaces (Recommended)

### Prerequisites
- HuggingFace account with a valid `HF_TOKEN`
- `git` and `git-lfs` installed
- `openenv` CLI: `pip install openenv-core`

### Steps

#### 1. Login to HuggingFace
```bash
huggingface-cli login
```

#### 2. Push with OpenEnv (from the alice/ directory)
```bash
cd alice
openenv push --repo-id rohanjain1648/alice-rl-env
```

This command:
- Validates `openenv.yaml`
- Builds a Docker image optimised for HF Spaces
- Uploads everything to `huggingface.co/spaces/rohanjain1648/alice-rl-env`

#### 3. Set Space Secrets
In your Space → Settings → Repository secrets, add:

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Your HuggingFace API token |
| `API_BASE_URL` | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` |

#### 4. Verify the Space
Once built, visit `https://huggingface.co/spaces/rohanjain1648/alice-rl-env`.
The OpenEnv web UI is at `/web`, health check at `/health`.

---

## Option 2 — Manual Docker Build (HF Spaces)

If `openenv push` is unavailable, deploy manually:

```bash
# 1. Clone a fresh HF Space repo
git clone https://huggingface.co/spaces/rohanjain1648/alice-rl-env
cd alice-rl-env

# 2. Copy project files
cp -r /path/to/ALICE/alice .
cp /path/to/ALICE/Dockerfile .
cp /path/to/ALICE/README.md .
cp /path/to/ALICE/train.py .
cp /path/to/ALICE/train.ipynb .
cp /path/to/ALICE/inference.py .
cp /path/to/ALICE/eval.py .
cp -r /path/to/ALICE/tests .
cp -r /path/to/ALICE/plots .   # if available

# 3. Push
git add .
git commit -m "Deploy ALICE"
git push
```

Set Secrets (same as Option 1 above).

---

## Option 3 — Local Docker

```bash
# Build
docker build -t alice-env .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_..." \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  alice-env
```

Test:
```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

## Option 4 — Local Python (development)

```bash
# Install
cd alice
pip install -e ".[dev]"

# Set env vars
export HF_TOKEN="hf_..."
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# In a second terminal — run inference
python inference.py

# Run evaluation on held-out set
python eval.py

# Run training (requires GPU)
python train.py --num-episodes 300
```

---

## Endpoints

Once running (any option), the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/schema` | GET | Action/observation schemas |
| `/ws` | WebSocket | Persistent session |
| `/web` | GET | OpenEnv web UI |
| `/docs` | GET | Swagger API docs |

---

## Troubleshooting

**`ImportError: openenv is required`**
→ Run `pip install openenv-core` or `uv sync` from the `alice/` directory.

**`HF_TOKEN and API_BASE_URL required`**
→ ALICE runs in degraded mode (template tasks, no Oracle) without these.
Set them as environment variables or Space secrets.

**`uvicorn server.app:app` fails with module not found**
→ Make sure you are running from the `alice/` directory (not the project root).
The correct command from project root is: `uvicorn alice.server.app:app`

**Docker build fails on `uv sync`**
→ Ensure `pyproject.toml` exists in `alice/`. The Dockerfile expects to run
`uv sync` from `alice/` inside the container.

**Port conflict on HF Spaces**
→ The `app_port: 8000` in README.md front matter must match the port in the
Dockerfile CMD. Both are set to 8000.
