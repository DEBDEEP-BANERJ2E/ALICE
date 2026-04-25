# ALICE Quick Start

Everything is ready. Pick your path:

---

## 🚀 Path 1: Test Locally (5 minutes)

```bash
# Terminal 1: Start server
cd alice
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Run inference
python ../inference.py
```

Expected output:
```
[START] ALICE inference — model=Qwen/Qwen2.5-7B-Instruct tasks=3
[STEP] task=0 tier=easy response="..." r_final=0.75
[END] mean_r_final=0.5000 duration=45.2s
```

---

## 🐳 Path 2: Docker (10 minutes)

```bash
# Build
docker build -t alice-env .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN="your_token" \
  -e API_BASE_URL="https://api-inference.huggingface.co/models/" \
  alice-env

# Test (in another terminal)
curl -s http://localhost:8000/health
```

---

## 🎓 Path 3: Train with HF Credits (Tomorrow)

### Step 1: Get HF Token
- Go to https://huggingface.co/settings/tokens
- Create new token with write access

### Step 2: Set Environment
```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://api-inference.huggingface.co/models/"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
```

### Step 3: Run Training
```bash
python train.py \
  --num-episodes 300 \
  --output-dir ./output/alice-trained \
  --hardware t4
```

### Step 4: Evaluate
```bash
python eval.py
```

---

## 📊 What You Have

✅ Full ALICE environment (11 components)
✅ Docker deployment ready
✅ Training scripts (TRL + Unsloth)
✅ Inference script
✅ Evaluation script
✅ 13 unit tests + 12 property tests
✅ Complete documentation

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `README.md` | Full architecture and design |
| `train.py` | GRPO training script |
| `train.ipynb` | Colab notebook |
| `inference.py` | Standalone evaluation |
| `eval.py` | Held-out test set evaluation |
| `Dockerfile` | Docker build |
| `alice/server/app.py` | FastAPI server |
| `alice/models.py` | Data models |

---

## 🔧 Troubleshooting

### Server won't start
```bash
# Check port 8000 is free
lsof -i :8000

# Or use different port
python -m uvicorn server.app:app --port 8001
```

### Inference fails
```bash
# Check HF_TOKEN is set
echo $HF_TOKEN

# Check API_BASE_URL is correct
echo $API_BASE_URL
```

### Docker build fails
```bash
# Clean build
docker build --no-cache -t alice-env .
```

---

## 📖 Next Steps

1. **Test locally** — Run `python inference.py` to verify everything works
2. **Review README.md** — Understand the architecture
3. **Tomorrow: Train** — Add HF credits and run `python train.py`
4. **Evaluate** — Run `python eval.py` to measure improvement

---

**Everything is ready. Just pick a path above and start!** 🚀
