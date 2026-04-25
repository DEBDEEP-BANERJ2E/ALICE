---
title: ALICE RL Environment
emoji: 🌀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - llm
---

# ALICE Environment

ALICE (Adversarial Loop for Inter-model Co-evolutionary Environment) is an OpenEnv
reinforcement-learning environment that trains LLMs to fix negation-arithmetic failure modes.

## Quick Start

```python
from alice.client import AliceEnv
from alice.models import AliceAction

with AliceEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation
    print(obs.task)          # negation-arithmetic task prompt
    print(obs.difficulty_tier)  # easy / medium / hard

    action = AliceAction(response="Answer: 7", mode="hunt", task_id=obs.task_id)
    result = env.step(action)
    print(result.observation.feedback)
    print(result.reward)
```

## Deploy to Hugging Face Spaces

```bash
# From the alice/ directory
openenv push --repo-id YOUR_USERNAME/alice-rl-env
```

The deployed Space provides:
- **Web UI** at `/web` — interactive OpenEnv interface
- **API docs** at `/docs` — Swagger/OpenAPI
- **Health check** at `/health`
- **WebSocket** at `/ws` — persistent low-latency sessions

## Environment Details

### Action — `AliceAction`

| Field | Type | Description |
|-------|------|-------------|
| `response` | str | The agent's answer to the current task |
| `mode` | str | `"hunt"` (find failures) or `"repair"` (fix them) |
| `task_id` | str | UUID echoed from the observation |

### Observation — `AliceObservation`

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | CoT-wrapped negation-arithmetic prompt |
| `skill_domain` | str | `"negation_arithmetic"` |
| `difficulty_tier` | str | `"easy"`, `"medium"`, or `"hard"` |
| `turn_number` | int | Current turn (0–4) |
| `hint` | str \| None | Structural hint injected on turn 3 |
| `reward` | float | R_final for this step |
| `done` | bool | True when episode ends (turn 4) |
| `feedback` | str | Verifier feedback text |
| `task_id` | str | UUID of current task |

### Episode Structure (5-turn FSM)

| Turn | What happens |
|------|--------------|
| 0 | Task presented with CoT scaffold |
| 1 | First attempt scored; Tier-1 feedback returned |
| 2 | Reflection prompt — model re-examines its chain |
| 3 | Hint injected; Tier 1+2 feedback |
| 4 | Final answer; full R_final computed; episode ends |

### Reward Formula

```
R_final = R_prog × R_reg × (1 − decay × turn) + λ × R_judge − novelty_penalty − repeat_penalty
```

- **R_prog** — programmatic exact-match (0 or 1)
- **R_reg** — regression battery pass rate
- **R_judge** — LLM oracle semantic score
- **novelty_penalty** — discourages n-gram similar failures
- **repeat_penalty** — penalises identical responses

## Running Locally

```bash
# Set credentials
export HF_TOKEN="hf_..."
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Start server (from alice/ directory)
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
alice/
├── __init__.py
├── client.py               # AliceEnv client
├── models.py               # AliceAction, AliceObservation, AliceState
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Dependencies
└── server/
    ├── app.py              # FastAPI application
    ├── alice_environment.py
    ├── curriculum_manager.py
    ├── episode_handler.py
    ├── failure_bank.py
    ├── gradio_dashboard.py
    ├── oracle.py
    ├── requirements.txt
    ├── reward.py
    ├── task_generator.py
    └── verifier.py
```
