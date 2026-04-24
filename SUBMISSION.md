# ALICE RL Environment - Submission

**Status:** ✅ Complete and verified

## What You're Getting

A real RL environment for negation arithmetic with:
- Curriculum learning (auto difficulty adjustment)
- 3-tier verification (programmatic + LLM judge + regression)
- Chain-of-thought scaffolding
- Failure bank for targeted training
- Discrimination rewards for zone of proximal development

## Quick Start

### 1. Deploy to HF Spaces (Docker)
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/alice-rl-env
cd alice-rl-env

# Copy files
cp -r ../alice .
cp ../Dockerfile .
cp ../README.md .
cp ../train.py .
cp ../inference.py .
cp ../eval.py .
cp -r ../tests .

# Push
git add . && git commit -m "Deploy ALICE" && git push
```

### 2. Set Secrets
In Space Settings → Repository secrets:
- `HF_TOKEN`: Your HF API token
- `API_BASE_URL`: `https://api-inference.huggingface.co/models/`
- `MODEL_NAME`: `Qwen/Qwen2.5-7B-Instruct`

### 3. Train
```bash
python train.py --num-episodes 300
```

OpenEnv handles the server automatically.

---

## What's Included

### Core Environment
- `alice/models.py` - Data models
- `alice/server/alice_environment.py` - Main environment
- `alice/server/failure_bank.py` - Failure persistence
- `alice/server/curriculum_manager.py` - Curriculum learning
- `alice/server/reward.py` - Reward calculation
- `alice/server/oracle.py` - HF API integration
- `alice/server/task_generator.py` - Task generation
- `alice/server/verifier.py` - 3-tier verification
- `alice/server/episode_handler.py` - Episode FSM
- `alice/server/app.py` - FastAPI server

### Training & Evaluation
- `train.py` - GRPO training with TRL/Unsloth
- `train.ipynb` - Colab notebook
- `inference.py` - Inference script
- `eval.py` - Evaluation on held-out test set

### Testing
- `tests/test_unit.py` - 13 unit tests
- `tests/test_properties.py` - 12 property-based tests

### Documentation
- `README.md` - Full documentation
- `DEPLOYMENT.md` - Deployment guide
- `IMPLEMENTATION_SUMMARY.md` - Architecture details

---

## Expected Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Mean reward | -0.30 | +0.20 | +0.50 (67%) |
| Success rate | 0% | 25% | +25% |

---

## Verification

```bash
# Check imports
python -c "from alice.server.alice_environment import AliceEnvironment; print('OK')"

# Run tests
pytest tests/ -v

# Run inference
python inference.py

# Run evaluation
python eval.py
```

---

## Key Features

✅ Real RL environment (not simulation)
✅ Curriculum learning
✅ 3-tier verification
✅ Chain-of-thought scaffolding
✅ Failure bank
✅ Discrimination rewards
✅ Concurrent sessions
✅ Degraded mode (works without API)
✅ Comprehensive testing
✅ OpenEnv-compatible

---

## Files

```
alice/
├── models.py
├── client.py
├── openenv.yaml
└── server/
    ├── app.py
    ├── alice_environment.py
    ├── failure_bank.py
    ├── curriculum_manager.py
    ├── reward.py
    ├── oracle.py
    ├── task_generator.py
    ├── verifier.py
    ├── episode_handler.py
    └── gradio_dashboard.py

tests/
├── conftest.py
├── test_unit.py
└── test_properties.py

train.py
train.ipynb
inference.py
eval.py
Dockerfile
README.md
DEPLOYMENT.md
IMPLEMENTATION_SUMMARY.md
```

---

## Next Steps

1. Deploy to HF Spaces (see DEPLOYMENT.md)
2. Set environment variables
3. Run training: `python train.py --num-episodes 300`
4. Evaluate: `python eval.py`
5. Download trained model
6. Submit Space link

---

🚀 Ready to submit!
