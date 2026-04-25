# ALICE Agent-Environment Interaction Architecture

## Overview
ALICE (Adversarial Loop for Inter-model Co-evolutionary Environment) is a reinforcement learning training environment that uses a 5-turn episode structure with curriculum learning, failure tracking, and multi-tier verification.

---

## COMPONENT BREAKDOWN

### 1. AGENT (LLM Model)
- **Role**: Receives tasks and generates responses
- **Input**: AliceObservation (task, difficulty, turn number, hints, feedback)
- **Output**: AliceAction (response text, mode, task_id)
- **Training**: Uses DPO (Direct Preference Optimization) with TRL

### 2. ENVIRONMENT (AliceEnvironment)
- **Role**: Core RL loop orchestrator
- **Methods**:
  - `reset()`: Initializes new episode, returns initial observation
  - `step(action)`: Processes agent action, returns new observation
- **State Management**: Maintains AliceState internally

### 3. EPISODE HANDLER (EpisodeHandler)
- **Role**: Manages 5-turn episode flow
- **Turns**:
  - Turn 0: Initial task presentation
  - Turn 1: Agent attempts task
  - Turn 2: Verification feedback
  - Turn 3: Hint provided (if needed)
  - Turn 4: Final attempt + episode end
- **Methods**:
  - `start_episode()`: Creates new episode with task
  - `handle_turn()`: Processes each turn's logic

### 4. TASK GENERATOR (TaskGenerator)
- **Role**: Creates tasks from failure bank or curriculum
- **Modes**:
  - Hunt: New tasks from curriculum
  - Repair: Tasks from failure bank (agent learns from past failures)
- **Task Types**:
  - CoT (Chain-of-Thought): "Solve X step by step"
  - Repair: "Fix this broken attempt: Y"

### 5. CURRICULUM MANAGER (CurriculumManager)
- **Role**: Tracks skill mastery and difficulty progression
- **Tracks**:
  - Accuracy per skill domain
  - Tier progression (easy → medium → hard)
  - Success/failure counts
- **Decision**: Determines next task difficulty based on performance

### 6. FAILURE BANK (FailureBank)
- **Role**: Persistent storage of agent failures
- **Stores**: FailureRecord (task, agent_response, correct_answer, timestamp)
- **Use**: Repair mode tasks created from these failures
- **Format**: JSONL (one record per line)

### 7. ORACLE (Oracle)
- **Role**: Ground truth evaluator using HF API
- **Methods**:
  - `score_task()`: Evaluates if agent response is correct
  - `_compute_pass_rate()`: Calculates accuracy over multiple attempts
- **API**: Calls HuggingFace inference API with TinyLlama model

### 8. VERIFIER STACK (VerifierStack)
- **Role**: 3-tier verification system
- **Tier 1 (Exact Match)**: String comparison with correct answer
- **Tier 2 (Semantic)**: Oracle model evaluation
- **Tier 3 (Consistency)**: Multiple attempts consistency check
- **Output**: Verification feedback sent to agent

### 9. REWARD CALCULATOR (RewardCalculator)
- **Role**: Computes training signal
- **Components**:
  - R_task: Task completion reward
  - R_efficiency: Reward for solving in fewer turns
  - R_consistency: Reward for stable performance
  - R_final: Weighted sum of above
- **Formula**: R_final = w1*R_task + w2*R_efficiency + w3*R_consistency

### 10. GRADIO DASHBOARD (GradioDashboard)
- **Role**: Real-time monitoring and visualization
- **Displays**:
  - Episode statistics
  - Success rates by difficulty
  - Failure bank size
  - Training progress

---

## DATA FLOW ARROWS & INTERACTIONS

### Arrow 1: AGENT → ENVIRONMENT (Action Submission)
- **Type**: AliceAction
- **Contains**: response, mode, task_id
- **Trigger**: Agent generates response to task
- **Frequency**: Once per turn (up to 5 times per episode)

### Arrow 2: ENVIRONMENT → AGENT (Observation)
- **Type**: AliceObservation
- **Contains**: task, skill_domain, difficulty_tier, turn_number, hint, reward, done, feedback, task_id
- **Trigger**: After environment processes action
- **Frequency**: Once per turn

### Arrow 3: ENVIRONMENT → EPISODE_HANDLER (Turn Processing)
- **Type**: AliceAction
- **Purpose**: Delegate turn-specific logic
- **Methods**: `handle_turn(action, state)`
- **Returns**: Updated AliceState

### Arrow 4: EPISODE_HANDLER → TASK_GENERATOR (Task Request)
- **Type**: Task request with mode (hunt/repair)
- **Purpose**: Get next task for episode
- **Returns**: Task object with prompt and correct_answer

### Arrow 5: TASK_GENERATOR → FAILURE_BANK (Query)
- **Type**: Query for repair mode tasks
- **Purpose**: Retrieve failed tasks for agent to repair
- **Returns**: FailureRecord with original task and agent's failed attempt

### Arrow 6: TASK_GENERATOR → CURRICULUM_MANAGER (Difficulty Query)
- **Type**: Query for skill domain
- **Purpose**: Get current difficulty tier for hunt mode
- **Returns**: Difficulty tier (easy/medium/hard)

### Arrow 7: CURRICULUM_MANAGER → CURRICULUM_STATE (Persistence)
- **Type**: Read/Write JSON
- **Purpose**: Load and save accuracy/tier tracking
- **File**: curriculum_state.json

### Arrow 8: EPISODE_HANDLER → VERIFIER_STACK (Verification Request)
- **Type**: (task, agent_response, correct_answer)
- **Purpose**: Verify if agent's response is correct
- **Returns**: (is_correct: bool, feedback: str, confidence: float)

### Arrow 9: VERIFIER_STACK → ORACLE (Semantic Check)
- **Type**: (task, agent_response, correct_answer)
- **Purpose**: Use LLM to evaluate semantic correctness
- **Returns**: Correctness score (0.0-1.0)

### Arrow 10: ORACLE → HF_API (Model Query)
- **Type**: HTTP POST to HuggingFace inference API
- **Endpoint**: https://api-inference.huggingface.co/v1
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Returns**: Model's evaluation of correctness

### Arrow 11: EPISODE_HANDLER → REWARD_CALCULATOR (Reward Computation)
- **Type**: (is_correct, turn_number, consistency_score)
- **Purpose**: Calculate training reward
- **Returns**: R_final (float)

### Arrow 12: EPISODE_HANDLER → FAILURE_BANK (Failure Recording)
- **Type**: FailureRecord (task, response, correct_answer)
- **Purpose**: Store failed attempts for future repair mode
- **Trigger**: When agent fails verification
- **File**: failure_bank.jsonl

### Arrow 13: EPISODE_HANDLER → CURRICULUM_MANAGER (Outcome Recording)
- **Type**: (skill_domain, correct: bool)
- **Purpose**: Update accuracy tracking for curriculum progression
- **Trigger**: At episode end

### Arrow 14: EPISODE_HANDLER → GRADIO_DASHBOARD (Logging)
- **Type**: Episode summary (episode_id, success, reward, difficulty)
- **Purpose**: Real-time monitoring
- **Frequency**: Once per episode

### Arrow 15: TRAINING_SCRIPT → ENVIRONMENT (Episode Loop)
- **Type**: Multiple reset() + step() calls
- **Purpose**: Generate training data for DPO trainer
- **Frequency**: Once per episode in training

### Arrow 16: ENVIRONMENT → TRAINING_SCRIPT (Dataset)
- **Type**: (prompt, chosen_response, rejected_response)
- **Purpose**: Provide training pairs for DPO
- **Returns**: Dataset for TRL trainer

---

## EPISODE FLOW SEQUENCE

```
START EPISODE
    ↓
[1] reset() → AliceEnvironment
    ├─ TaskGenerator.generate_task() → Task
    ├─ CurriculumManager.get_tier() → difficulty
    └─ Return: AliceObservation (Turn 0)
    ↓
[2] Agent receives observation
    ├─ Reads: task, difficulty, skill_domain
    └─ Generates: response
    ↓
[3] Agent submits AliceAction
    ├─ response: agent's answer
    ├─ mode: "hunt" or "repair"
    └─ task_id: UUID
    ↓
[4] step(action) → AliceEnvironment
    ├─ EpisodeHandler.handle_turn()
    │   ├─ VerifierStack.verify() → (is_correct, feedback)
    │   │   ├─ Tier 1: Exact match
    │   │   ├─ Tier 2: Oracle semantic check
    │   │   └─ Tier 3: Consistency check
    │   ├─ RewardCalculator.compute() → R_final
    │   ├─ If failed: FailureBank.append(FailureRecord)
    │   ├─ If turn < 4: Generate hint (Turn 3)
    │   └─ If turn == 4: Episode ends
    ├─ CurriculumManager.record_outcome()
    └─ Return: AliceObservation (Turn N)
    ↓
[5] Agent receives new observation
    ├─ Reads: feedback, hint (if Turn 3), reward
    └─ Generates: next response
    ↓
[6] Repeat steps 3-5 until turn == 4 (done=True)
    ↓
[7] Episode ends
    ├─ GradioDashboard.log_episode()
    └─ Training data collected for DPO
    ↓
END EPISODE
```

---

## TURN-BY-TURN BREAKDOWN

### Turn 0 (Initial)
- **Agent receives**: Task prompt, difficulty, skill domain
- **Agent action**: None (observation only)
- **Environment**: Prepares episode state

### Turn 1 (First Attempt)
- **Agent receives**: Task prompt
- **Agent submits**: Response
- **Environment**:
  - Verifies response (3-tier)
  - Computes reward
  - Records if failed
  - Provides feedback

### Turn 2 (Feedback)
- **Agent receives**: Verification feedback, reward
- **Agent submits**: Revised response
- **Environment**: Re-verifies, updates reward

### Turn 3 (Hint)
- **Agent receives**: Structural hint (e.g., "Break into steps")
- **Agent submits**: Response with hint guidance
- **Environment**: Verifies with hint context

### Turn 4 (Final)
- **Agent receives**: Final feedback, cumulative reward
- **Agent submits**: Final response
- **Environment**: 
  - Final verification
  - Episode ends (done=True)
  - Updates curriculum
  - Logs to dashboard

---

## CURRICULUM PROGRESSION

```
Skill Domain: negation_arithmetic

Easy Tier:
  ├─ Accuracy threshold: 60%
  ├─ Task complexity: Simple negation + single operation
  └─ Example: "NOT(5 + 3) = ?"

Medium Tier:
  ├─ Accuracy threshold: 75%
  ├─ Task complexity: Multiple operations + negation
  └─ Example: "NOT((5 + 3) * 2) = ?"

Hard Tier:
  ├─ Accuracy threshold: 85%
  ├─ Task complexity: Complex nested operations
  └─ Example: "NOT(NOT(5 + 3) - 2) = ?"

Progression Logic:
  - If accuracy > threshold → Promote to next tier
  - If accuracy < 50% → Demote to previous tier
  - Repair mode: Use failures from current tier
```

---

## REWARD DECOMPOSITION

```
R_final = w1 * R_task + w2 * R_efficiency + w3 * R_consistency

R_task:
  - 1.0 if correct on first attempt
  - 0.5 if correct after hint
  - 0.0 if incorrect after all turns

R_efficiency:
  - Bonus for solving in fewer turns
  - Formula: 1.0 - (turn_number / 4)

R_consistency:
  - Reward for stable performance
  - Tracks entropy of recent attempts
  - Higher entropy = lower consistency reward

Weights (default):
  - w1 = 0.6 (task correctness)
  - w2 = 0.2 (efficiency)
  - w3 = 0.2 (consistency)
```

---

## FAILURE BANK STRUCTURE

```
FailureRecord:
  - task: str (original task prompt)
  - agent_response: str (what agent submitted)
  - correct_answer: str (ground truth)
  - timestamp: str (ISO format)
  - skill_domain: str (negation_arithmetic)
  - difficulty_tier: str (easy/medium/hard)

Storage: failure_bank.jsonl
  - One JSON object per line
  - Append-only (no overwrites)
  - Used for repair mode task generation
```

---

## VERIFICATION STACK TIERS

```
Tier 1: Exact Match
  - Compare agent_response with correct_answer
  - Fast, deterministic
  - Confidence: 1.0 if match, 0.0 if no match

Tier 2: Semantic (Oracle)
  - Use LLM to evaluate correctness
  - Prompt: "Is this answer correct? [task] [response] [correct_answer]"
  - Confidence: 0.0-1.0 based on model output

Tier 3: Consistency
  - Check if agent gives same answer on multiple attempts
  - Entropy-based scoring
  - Confidence: Based on consistency window

Final Decision:
  - If Tier 1 matches: Use Tier 1 result
  - Else if Tier 2 confidence > 0.7: Use Tier 2 result
  - Else use Tier 3 result
```

---

## TRAINING DATA GENERATION

```
For each episode:
  1. Agent completes 5-turn episode
  2. Collect all (task, response, reward) tuples
  3. Generate training pairs:
     - Chosen: Response with highest reward
     - Rejected: Response with lowest reward
  4. Format for DPO trainer:
     {
       "prompt": task,
       "chosen": best_response,
       "rejected": worst_response
     }
  5. Accumulate into dataset
  6. Pass to TRL DPOTrainer for optimization
```

---

## API ENDPOINTS (FastAPI)

```
POST /reset
  - Input: None
  - Output: AliceObservation
  - Purpose: Start new episode

POST /step
  - Input: AliceAction
  - Output: AliceObservation
  - Purpose: Process agent action

GET /dashboard
  - Input: None
  - Output: Dashboard data (JSON)
  - Purpose: Real-time monitoring

GET /health
  - Input: None
  - Output: {"status": "ok"}
  - Purpose: Health check
```

---

## KEY INTERACTIONS SUMMARY

| From | To | Data | Purpose |
|------|-----|------|---------|
| Agent | Environment | AliceAction | Submit response |
| Environment | Agent | AliceObservation | Provide task/feedback |
| Environment | EpisodeHandler | AliceAction | Delegate turn logic |
| EpisodeHandler | TaskGenerator | Query | Get next task |
| TaskGenerator | CurriculumManager | Query | Get difficulty |
| TaskGenerator | FailureBank | Query | Get repair tasks |
| EpisodeHandler | VerifierStack | (task, response, answer) | Verify correctness |
| VerifierStack | Oracle | (task, response, answer) | Semantic check |
| Oracle | HF API | Prompt | Model evaluation |
| EpisodeHandler | RewardCalculator | (correct, turn, consistency) | Compute reward |
| EpisodeHandler | FailureBank | FailureRecord | Store failures |
| EpisodeHandler | CurriculumManager | (domain, correct) | Update accuracy |
| EpisodeHandler | Dashboard | Episode summary | Log progress |
| Training | Environment | Multiple episodes | Generate dataset |
| Environment | Training | (prompt, chosen, rejected) | DPO training pairs |
