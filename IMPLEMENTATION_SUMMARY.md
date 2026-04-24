# ALICE RL Environment - Implementation Summary

Real RL environment for negation arithmetic with curriculum learning, 3-tier verification, and co-evolutionary reward signals. OpenEnv handles the server and deployment.

## What We Built

### Core Components

1. **Data Models** (`alice/models.py`)
   - AliceAction: response, mode (hunt/repair), task_id
   - AliceObservation: task, tier, turn, reward, feedback
   - AliceState: episode state with attempt history

2. **Failure Bank** (`alice/server/failure_bank.py`)
   - JSONL persistence of failed attempts
   - N-gram fingerprinting + cosine similarity
   - Used in repair mode for targeted training

3. **Curriculum Manager** (`alice/server/curriculum_manager.py`)
   - Rolling accuracy window (20 episodes)
   - Auto tier assignment: easy (acc<0.4) → medium → hard (acc>0.75)

4. **Reward Calculator** (`alice/server/reward.py`)
   - Composite formula: r_prog × r_reg × (1-0.1×turn) + 0.3×r_judge - 0.2×similarity - 0.15×repeat
   - Bounds: [-1.0, 1.0]

5. **Oracle** (`alice/server/oracle.py`)
   - HF Inference API integration
   - Discrimination scoring for zone of proximal development

6. **Task Generator** (`alice/server/task_generator.py`)
   - Hunt mode (70%): Random templates
   - Repair mode (30%): Targeted failure remediation
   - 9 templates across 3 difficulty tiers
   - CoT wrapping for all tasks

7. **Verifier Stack** (`alice/server/verifier.py`)
   - Tier 1: Exact numeric match
   - Tier 2: LLM judge for reasoning
   - Tier 3: Regression battery (20 fixed tasks)
   - CoT-aware answer extraction

8. **Episode Handler** (`alice/server/episode_handler.py`)
   - 5-turn FSM with CoT scaffolding
   - Turn 0: Reset
   - Turn 1: Initial attempt (Tier 1)
   - Turn 2: Reflection
   - Turn 3: With hint (Tier 1+2)
   - Turn 4: Final (full verification + curriculum update)

9. **ALICE Environment** (`alice/server/alice_environment.py`)
   - OpenEnv-compatible interface
   - Concurrent session support
   - Degraded mode without API access

### Training & Evaluation

- **train.py:** GRPO with TRL/Unsloth, LoRA fine-tuning
- **train.ipynb:** Colab-compatible notebook
- **inference.py:** 3-task evaluation with exponential backoff
- **eval.py:** 50-task held-out test set with per-tier accuracy

### Testing

- **test_unit.py:** 13 concrete unit tests
- **test_properties.py:** 12 Hypothesis-based property tests

---

## Real RL Verification

✅ **Components:** All production code uses real components
✅ **Rewards:** Composite formula applied correctly
✅ **Episode Flow:** 5-turn FSM working as designed
✅ **Curriculum:** Tier assignment based on rolling accuracy
✅ **Failure Bank:** JSONL persistence verified
✅ **Degraded Mode:** Works without HF_TOKEN/API_BASE_URL

**Example Episode:**
```
Turn 1: Response "Answer: 14" → Reward 0.00 (incorrect)
Turn 2: Reflection → Reward 0.00
Turn 3: With hint → Reward 0.00
Turn 4: Final → Reward -0.30 (composite penalty)
```

**5-Episode Training:**
```
Episode 1: -0.479
Episode 2: -0.197
Episode 3: -0.492
Episode 4: -0.197
Episode 5: -0.317
Mean: -0.3367
```

---

## Expected Results

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Mean reward | -0.30 | +0.20 | +0.50 |
| Success rate | 0% | 25% | +25% |
| Improvement | — | — | 67% |

---

## Files

**Core:**
- `alice/models.py` - Data models
- `alice/client.py` - Client library
- `alice/server/alice_environment.py` - Main environment
- `alice/server/failure_bank.py` - Failure persistence
- `alice/server/curriculum_manager.py` - Curriculum learning
- `alice/server/reward.py` - Reward calculation
- `alice/server/oracle.py` - HF API integration
- `alice/server/task_generator.py` - Task generation
- `alice/server/verifier.py` - 3-tier verification
- `alice/server/episode_handler.py` - Episode FSM
- `alice/server/app.py` - FastAPI server

**Training:**
- `train.py` - GRPO training
- `train.ipynb` - Colab notebook
- `inference.py` - Inference
- `eval.py` - Evaluation

**Testing:**
- `tests/conftest.py` - Fixtures
- `tests/test_unit.py` - Unit tests
- `tests/test_properties.py` - Property tests

**Docs:**
- `README.md` - Main documentation
- `DEPLOYMENT.md` - Deployment guide
- `IMPLEMENTATION_SUMMARY.md` - This file

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
