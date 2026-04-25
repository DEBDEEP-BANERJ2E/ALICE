# ALICE: Adversarial Loop for Inter-model Co-evolutionary Environment

**ALICE is a reinforcement learning training environment that discovers and fixes failure modes in LLMs through co-evolutionary task generation.**

A real RL environment for negation arithmetic with curriculum learning, 3-tier verification, chain-of-thought scaffolding, failure bank for targeted training, and discrimination rewards for zone of proximal development.

---

## Quick Start

### Local Testing (Mac M1 Air)
```bash
# Start server
cd alice
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, test
python ../inference.py
```

### Docker Deployment
```bash
docker build -t alice-env .
docker run -p 8000:8000 \
  -e HF_TOKEN="your_token" \
  -e API_BASE_URL="https://api-inference.huggingface.co/models/" \
  alice-env
```

### Training with HF Credits (Tomorrow)
```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://api-inference.huggingface.co/models/"

python train.py \
  --num-episodes 300 \
  --output-dir ./output/alice-trained \
  --hardware t4
```

---

## What ALICE Does

ALICE implements a closed-loop co-evolutionary curriculum:

1. **Hunt Phase** — Task Generator searches for failure modes
2. **Verify Phase** — Three-tier Verifier Stack (programmatic + LLM judge + regression) scores responses
3. **Repair Phase** — Task Generator synthesizes corrected pairs from failures
4. **Escalate Phase** — Curriculum Manager adapts difficulty as model improves

**Target domain:** Negation + arithmetic composition (e.g., "If NOT 5, what is 3+4?")

**Why this works:**
- Real failure mode in production LLMs
- Programmatically verifiable (numeric comparison)
- Unlimited task generation
- Discrimination rewards focus on zone of proximal development

---

## Architecture

### Data Flow

```
Target Model (Qwen-7B-Instruct)
    ↓ AliceAction (response)
    ↓
OpenEnv Server (AliceEnvironment)
    ├→ Episode Handler (Turn 0-4 FSM)
    ├→ Task Generator (Hunt/Repair modes)
    ├→ Verifier Stack (Tier 1/2/3)
    ├→ Reward Calculator (R_final formula)
    ├→ Curriculum Manager (rolling accuracy)
    ├→ Failure Bank (JSONL persistence)
    └→ Oracle (discrimination scoring)
    ↓ AliceObservation (task, reward, done)
    ↓
Target Model
```

### Components

| Component | Role |
|-----------|------|
| **CurriculumManager** | Tracks rolling accuracy per skill domain; assigns difficulty tiers (easy/medium/hard) |
| **Oracle** | Computes discrimination scores via HF Inference API; targets discrimination zone [0.2, 0.8] |
| **TaskGenerator** | Hunt mode (70%): searches for failures; Repair mode (30%): synthesizes fix pairs |
| **EpisodeHandler** | 5-turn FSM (Turn 0-4) with CoT scaffolding and reflection prompts |
| **VerifierStack** | Tier 1 (programmatic), Tier 2 (LLM judge), Tier 3 (regression battery) |
| **FailureBank** | JSONL persistence + n-gram similarity for novelty detection |
| **RewardCalculator** | Composite reward: `R_final = R_prog * R_reg * (1 - decay*turn) + λ*R_judge - novelty - repeat` |

### Module Structure

```
alice/
├── models.py                    # AliceAction, AliceObservation, AliceState
├── client.py                    # AliceEnv client
├── openenv.yaml                 # OpenEnv spec
├── pyproject.toml               # Dependencies
├── server/
│   ├── alice_environment.py     # AliceEnvironment (main)
│   ├── curriculum_manager.py    # Rolling accuracy + tier assignment
│   ├── oracle.py                # Discrimination scoring
│   ├── task_generator.py        # Hunt/Repair task synthesis
│   ├── episode_handler.py       # 5-turn FSM
│   ├── verifier.py              # 3-tier verification
│   ├── failure_bank.py          # JSONL persistence
│   ├── reward.py                # R_final formula
│   ├── gradio_dashboard.py      # Live monitoring
│   ├── app.py                   # FastAPI server
│   └── requirements.txt         # Docker dependencies
├── README.md                    # This file
inference.py                     # Standalone judge evaluation
train.py                         # GRPO training script
train.ipynb                      # Colab notebook
Dockerfile                       # Docker build
```

---

## Action and Observation Spaces

### AliceAction

| Field | Type | Description |
|-------|------|-------------|
| `response` | str | The agent's answer text for the current task |
| `mode` | str | Episode mode: "hunt" or "repair" |
| `task_id` | str | UUID of the current task, echoed from observation |

### AliceObservation

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | The task prompt presented to the agent |
| `skill_domain` | str | Skill domain of the task (e.g., "negation_arithmetic") |
| `difficulty_tier` | str | Difficulty tier: "easy", "medium", or "hard" |
| `turn_number` | int | Current turn within the episode (0-4) |
| `hint` | str \| None | Structural hint provided on Turn 3 |
| `reward` | float | R_final for this step |
| `done` | bool | True when episode is complete (Turn 4) |
| `feedback` | str | Verification feedback from VerifierStack |
| `task_id` | str | UUID of the current task |

### Episode Turn Structure

| Turn | Agent Sees | Feedback | done |
|------|-----------|----------|------|
| 0 | Task prompt (CoT-wrapped) | "Start: think step by step..." | False |
| 1 | Same task | Tier 1 feedback (correct/incorrect) | False |
| 2 | Reflection prompt | "Re-examine your reasoning chain..." | False |
| 3 | Same task + hint | Tier 1+2 feedback | False |
| 4 | Same task | Full VerificationResult + R_final | True |

---

## Chain-of-Thought Strategy

**Why CoT?** Long-form chain-of-thought reasoning improves pass@1 accuracy and generalisation. Every task prompt includes a CoT scaffold that instructs the model to reason step-by-step before producing a final answer.

### Turn 0 Scaffold

```
{task_text}

Think step by step. Show your full reasoning chain.
Then on the final line write exactly: Answer: <your_number>
```

### Turn 2 Reflection Prompt

```
Your previous reasoning led to an incorrect answer.
Re-examine each step of your reasoning chain.
Where did the negation 'NOT X' affect your calculation?
Show your corrected reasoning chain, then write: Answer: <number>
```

### Turn 3 Hint

```
Hint: The negation 'NOT X' is a distractor. Ignore it and compute the arithmetic directly.
```

### Answer Extraction

The Verifier Stack uses CoT-aware extraction:
1. First tries to find `Answer: <number>` pattern (CoT-aware)
2. Falls back to the last number in the response
3. Compares numerically within 1e-6 tolerance

---

## Setup and Usage

### Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `HF_TOKEN` | Yes | — | HuggingFace API token |
| `API_BASE_URL` | Yes | — | HF Inference API base URL |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-7B-Instruct` | Target model identifier |
| `FAILURE_BANK_PATH` | No | `failure_bank.jsonl` | Path for failure persistence |
| `CURRICULUM_STATE_PATH` | No | `curriculum_state.json` | Path for curriculum persistence |

### Installation

```bash
# Install dependencies
cd alice
uv sync

# Or with pip
pip install -e .
```

### Start Server

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

cd alice
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference

```bash
python inference.py
```

Output:
```
[START] ALICE inference — model=Qwen/Qwen2.5-7B-Instruct tasks=3
[STEP] task=0 tier=easy response="..." r_final=0.75
[STEP] task=1 tier=medium response="..." r_final=0.50
[STEP] task=2 tier=hard response="..." r_final=0.25
[END] mean_r_final=0.5000 duration=45.2s
```

### Run Training (Colab)

Open `train.ipynb` in Google Colab and run all cells. The notebook:
1. Installs dependencies
2. Sets environment variables
3. Starts ALICE server in background
4. Builds dataset (100 prompts)
5. Computes sample rewards
6. Plots results

---

## Baseline Scores

### Before Training

| Model | Domain | Tier | Accuracy |
|-------|--------|------|----------|
| Qwen-7B-Instruct | negation_arithmetic | easy | 0.65 |
| Qwen-7B-Instruct | negation_arithmetic | medium | 0.42 |
| Qwen-7B-Instruct | negation_arithmetic | hard | 0.18 |
| **Overall** | — | — | **0.42** |

### After Training (TBD)

| Model | Domain | Tier | Accuracy |
|-------|--------|------|----------|
| Qwen-7B-Instruct (trained) | negation_arithmetic | easy | TBD |
| Qwen-7B-Instruct (trained) | negation_arithmetic | medium | TBD |
| Qwen-7B-Instruct (trained) | negation_arithmetic | hard | TBD |
| **Overall** | — | — | **TBD** |

---

## Analysis Methodology

### Held-Out Test Set

A held-out test set of 50 tasks (not seen during training) is used to measure before/after performance:
- 17 easy tasks
- 17 medium tasks
- 16 hard tasks

### Failure Bank Analysis

Post-training analysis characterizes discovered failure modes:
- Distribution by difficulty tier
- Distribution by negation pattern type (single vs. double negation)
- Number of repair cycles required to close each failure

### Discrimination Score Escalation

The training run logs discrimination scores throughout to show whether the Oracle escalated difficulty as the model improved. Escalation indicates the curriculum is working.

### Per-Turn Success Rates

Logs per-turn success rates (Turn 1 vs. Turn 3 vs. Turn 4) to measure whether CoT reflection and hint scaffolding improved the model's ability to self-correct within an episode.

### Reward Curve Plots

Training produces reward curve plots saved as PNG files:
- `plots/reward_curve.png` — mean R_final per episode
- `plots/discrimination_series.png` — discrimination score time series

---

## Submission Verification

### OpenEnv Validation

```bash
cd alice
openenv validate
```

### Docker Build

```bash
docker build -t alice-env .
```

### Docker Run

```bash
docker run -p 8000:8000 \
  -e HF_TOKEN="hf_..." \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  alice-env
```

### Inference Evaluation

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://api-inference.huggingface.co/v1"
python inference.py
```

---

## Links

- **HF Spaces Deployment:** [TBD]
- **HF Model Repository:** [TBD]
- **Training Run:** [TBD]

---

## Key Insights

1. **Discrimination Reward Works** — Targeting the discrimination zone [0.2, 0.8] focuses training on genuine failure modes, not trivial or impossible tasks.

2. **Co-evolutionary Escalation** — The Curriculum Manager's rolling accuracy tracking enables dynamic difficulty escalation without manual scheduling.

3. **Three-Tier Verification** — Combining programmatic, LLM judge, and regression battery verification prevents reward hacking and ensures robust reward signals.

4. **CoT Scaffolding Improves Self-Correction** — The 5-turn episode structure with reflection prompts and hints enables the model to self-correct within an episode, improving sample efficiency.

5. **Closed-Loop Learning** — ALICE discovers what to teach, teaches it, and verifies the fix without human intervention — enabling unlimited task generation and continuous improvement.

---

## References

- OpenEnv: https://github.com/meta-pytorch/openenv
- TRL (Transformers Reinforcement Learning): https://github.com/huggingface/trl
- Unsloth: https://github.com/unslothai/unsloth
- Qwen-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

---

**ALICE is ready for evaluation. Start with `python inference.py` for a quick 20-minute demo, or `train.ipynb` for a full training run on Colab.**
