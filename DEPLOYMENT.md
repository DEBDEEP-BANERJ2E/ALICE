# ALICE RL Environment - Deployment

OpenEnv handles the server, API, and deployment. Just push to HF Spaces with Docker.

## Quick Start

### Step 1: Create Space
```bash
# Go to https://huggingface.co/new-space
# Select Docker SDK
# Name: alice-rl-env
```

### Step 2: Push Code
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/alice-rl-env
cd alice-rl-env

# Copy ALICE files
cp -r ../alice .
cp ../Dockerfile .
cp ../README.md .
cp ../train.py .
cp ../inference.py .
cp ../eval.py .
cp -r ../tests .

# Push
git add .
git commit -m "Deploy ALICE"
git push
```

### Step 3: Set Secrets
In Space Settings → Repository secrets:
- `HF_TOKEN`: Your HF API token
- `API_BASE_URL`: `https://api-inference.huggingface.co/models/`
- `MODEL_NAME`: `Qwen/Qwen2.5-7B-Instruct`

### Step 4: Train
OpenEnv will start the server automatically. Run training:
```bash
python train.py --num-episodes 300
```

---

## Local Development

```bash
uv sync
python train.py --num-episodes 300
```

---

## Hardware

- **CPU Basic (free):** 2-4 hours
- **Your laptop (free):** 30-60 min with GPU
- **Custom Hardware (paid):** 15-30 min

---

## Expected Results

**Before:** Mean reward -0.30, Success rate 0%
**After:** Mean reward +0.20, Success rate 25%
**Improvement:** +0.50 reward delta (67%)
