# ALICE System Architecture - Visual Diagram

## MAIN SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ALICE RL TRAINING SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   AGENT      │
                              │  (LLM Model) │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   ARROW 1    │  │   ARROW 2    │  │   ARROW 15   │
            │   (Action)   │  │(Observation) │  │  (Episodes)  │
            └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                   │                 │                 │
                   └─────────────────┼─────────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │  ALICE ENVIRONMENT     │
                        │  (Core RL Loop)        │
                        │  - reset()             │
                        │  - step(action)        │
                        └────────┬───────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │   ARROW 3    │ │   ARROW 16   │ │   ARROW 14   │
            │(Turn Logic)  │ │(Training)    │ │(Dashboard)   │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                   │                │                │
                   ▼                ▼                ▼
        ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
        │ EPISODE HANDLER  │ │ TRAINING     │ │   GRADIO     │
        │ (5-turn flow)    │ │ SCRIPT       │ │  DASHBOARD   │
        │ - handle_turn()  │ │ (DPO Trainer)│ │ (Monitoring) │
        └────────┬─────────┘ └──────────────┘ └──────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ ARROW 4  │ │ ARROW 8  │ │ ARROW 11 │
│(Task)    │ │(Verify)  │ │(Reward)  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ TASK         │ │ VERIFIER     │ │ REWARD       │
│ GENERATOR    │ │ STACK        │ │ CALCULATOR   │
│ (hunt/repair)│ │ (3-tier)     │ │ (R_final)    │
└────┬─────────┘ └────┬─────────┘ └──────────────┘
     │                │
  ┌──┴──┐          ┌──┴──┐
  │     │          │     │
  ▼     ▼          ▼     ▼
┌──────────────┐ ┌──────────────┐
│ ARROW 5      │ │ ARROW 9      │
│(Failures)    │ │(Semantic)    │
└────┬─────────┘ └────┬─────────┘
     │                │
     ▼                ▼
┌──────────────┐ ┌──────────────┐
│ FAILURE BANK │ │ ORACLE       │
│ (JSONL)      │ │ (LLM eval)   │
└──────────────┘ └────┬─────────┘
                      │
                      ▼
                 ┌──────────────┐
                 │ ARROW 10     │
                 │ (HF API)     │
                 └────┬─────────┘
                      │
                      ▼
                 ┌──────────────┐
                 │ HUGGINGFACE  │
                 │ INFERENCE    │
                 │ API          │
                 │ (TinyLlama)  │
                 └──────────────┘

     ┌─────────────────────────────────────┐
     │ ARROW 6 (Difficulty)                │
     │ ARROW 7 (Curriculum State)          │
     │ ARROW 12 (Failure Recording)        │
     │ ARROW 13 (Outcome Recording)        │
     └─────────────────────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │ CURRICULUM MANAGER   │
            │ (Tier progression)   │
            │ - easy → medium      │
            │ - medium → hard      │
            └──────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │ CURRICULUM STATE     │
            │ (curriculum_state    │
            │  .json)              │
            └──────────────────────┘
```


---

## EPISODE FLOW DIAGRAM (5-Turn Sequence)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EPISODE LIFECYCLE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

START
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TURN 0: INITIALIZATION                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Environment.reset()                                                         │
│    ├─ TaskGenerator.generate_task()                                          │
│    │   ├─ CurriculumManager.get_tier() ──→ difficulty (easy/medium/hard)    │
│    │   └─ Return: Task(prompt, correct_answer)                              │
│    │                                                                         │
│    └─ Return: AliceObservation                                               │
│        ├─ task: "NOT(5 + 3) = ?"                                             │
│        ├─ skill_domain: "negation_arithmetic"                                │
│        ├─ difficulty_tier: "easy"                                            │
│        ├─ turn_number: 0                                                     │
│        ├─ hint: None                                                         │
│        ├─ reward: 0.0                                                        │
│        ├─ done: False                                                        │
│        └─ task_id: "uuid-1234"                                               │
│                                                                              │
│  Agent receives observation                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TURN 1: FIRST ATTEMPT                                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent submits AliceAction:                                                  │
│    ├─ response: "-8"                                                         │
│    ├─ mode: "hunt"                                                           │
│    └─ task_id: "uuid-1234"                                                   │
│                                                                              │
│  Environment.step(action)                                                    │
│    └─ EpisodeHandler.handle_turn(action, state)                              │
│        ├─ VerifierStack.verify(task, response, correct_answer)              │
│        │   ├─ Tier 1: Exact match? "-8" == "-8" ✓ YES                       │
│        │   ├─ Confidence: 1.0                                                │
│        │   └─ is_correct: True, feedback: "Correct!"                         │
│        │                                                                     │
│        ├─ RewardCalculator.compute(is_correct=True, turn=1)                  │
│        │   ├─ R_task: 1.0 (correct on first attempt)                         │
│        │   ├─ R_efficiency: 0.75 (1.0 - 1/4)                                 │
│        │   ├─ R_consistency: 1.0 (no history yet)                            │
│        │   └─ R_final: 0.6*1.0 + 0.2*0.75 + 0.2*1.0 = 0.85                   │
│        │                                                                     │
│        ├─ CurriculumManager.record_outcome("negation_arithmetic", True)      │
│        │   └─ Update accuracy: 1/1 = 100%                                    │
│        │                                                                     │
│        └─ Return: AliceObservation                                            │
│            ├─ task: "NOT(5 + 3) = ?"                                         │
│            ├─ turn_number: 1                                                 │
│            ├─ reward: 0.85                                                   │
│            ├─ feedback: "Correct!"                                           │
│            ├─ done: False                                                    │
│            └─ task_id: "uuid-1234"                                           │
│                                                                              │
│  Agent receives observation                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TURN 2: FEEDBACK & REVISION                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent submits AliceAction:                                                  │
│    ├─ response: "-8"                                                         │
│    ├─ mode: "hunt"                                                           │
│    └─ task_id: "uuid-1234"                                                   │
│                                                                              │
│  Environment.step(action)                                                    │
│    └─ EpisodeHandler.handle_turn(action, state)                              │
│        ├─ VerifierStack.verify(task, response, correct_answer)              │
│        │   ├─ Tier 1: Exact match? "-8" == "-8" ✓ YES                       │
│        │   └─ is_correct: True                                               │
│        │                                                                     │
│        ├─ RewardCalculator.compute(is_correct=True, turn=2)                  │
│        │   ├─ R_task: 1.0                                                    │
│        │   ├─ R_efficiency: 0.5 (1.0 - 2/4)                                  │
│        │   ├─ R_consistency: 0.95 (high consistency)                         │
│        │   └─ R_final: 0.6*1.0 + 0.2*0.5 + 0.2*0.95 = 0.79                   │
│        │                                                                     │
│        └─ Return: AliceObservation (turn_number: 2, reward: 0.79)            │
│                                                                              │
│  Agent receives observation                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TURN 3: HINT PROVIDED                                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent submits AliceAction:                                                  │
│    ├─ response: "-8"                                                         │
│    ├─ mode: "hunt"                                                           │
│    └─ task_id: "uuid-1234"                                                   │
│                                                                              │
│  Environment.step(action)                                                    │
│    └─ EpisodeHandler.handle_turn(action, state)                              │
│        ├─ VerifierStack.verify(task, response, correct_answer)              │
│        │   └─ is_correct: True                                               │
│        │                                                                     │
│        ├─ Generate hint for next turn:                                       │
│        │   └─ hint: "Break the problem into steps: first negate, then add"   │
│        │                                                                     │
│        ├─ RewardCalculator.compute(is_correct=True, turn=3)                  │
│        │   └─ R_final: 0.6 (reduced for later turn)                          │
│        │                                                                     │
│        └─ Return: AliceObservation                                            │
│            ├─ turn_number: 3                                                 │
│            ├─ hint: "Break the problem into steps..."                        │
│            ├─ reward: 0.6                                                    │
│            └─ done: False                                                    │
│                                                                              │
│  Agent receives observation with hint                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TURN 4: FINAL ATTEMPT & EPISODE END                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Agent submits AliceAction:                                                  │
│    ├─ response: "-8"                                                         │
│    ├─ mode: "hunt"                                                           │
│    └─ task_id: "uuid-1234"                                                   │
│                                                                              │
│  Environment.step(action)                                                    │
│    └─ EpisodeHandler.handle_turn(action, state)                              │
│        ├─ VerifierStack.verify(task, response, correct_answer)              │
│        │   └─ is_correct: True                                               │
│        │                                                                     │
│        ├─ RewardCalculator.compute(is_correct=True, turn=4)                  │
│        │   └─ R_final: 0.4 (final turn penalty)                              │
│        │                                                                     │
│        ├─ CurriculumManager.record_outcome("negation_arithmetic", True)      │
│        │   └─ Update accuracy: 5/5 = 100% → Promote to MEDIUM tier          │
│        │                                                                     │
│        ├─ FailureBank: No failures to record (all correct)                   │
│        │                                                                     │
│        ├─ GradioDashboard.log_episode()                                      │
│        │   └─ Log: episode_id, success=True, reward=0.4, difficulty=easy     │
│        │                                                                     │
│        └─ Return: AliceObservation                                            │
│            ├─ turn_number: 4                                                 │
│            ├─ reward: 0.4                                                    │
│            ├─ done: True ◄─── EPISODE ENDS                                   │
│            └─ feedback: "Episode complete. Promoted to MEDIUM tier!"         │
│                                                                              │
│  Training data collected:                                                    │
│    {                                                                         │
│      "prompt": "NOT(5 + 3) = ?",                                             │
│      "chosen": "-8",  (highest reward response)                              │
│      "rejected": ""   (lowest reward response, if any)                       │
│    }                                                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
END EPISODE
```


---

## VERIFICATION STACK (3-TIER) FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VERIFIER STACK DECISION TREE                             │
└─────────────────────────────────────────────────────────────────────────────┘

Input: (task, agent_response, correct_answer)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: EXACT MATCH                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Compare: agent_response == correct_answer                                   │
│                                                                              │
│  Example:                                                                    │
│    agent_response: "-8"                                                      │
│    correct_answer: "-8"                                                      │
│    Match: YES ✓                                                              │
│                                                                              │
│  Result:                                                                     │
│    ├─ is_correct: True                                                       │
│    ├─ confidence: 1.0                                                        │
│    ├─ feedback: "Correct!"                                                   │
│    └─ Return immediately (no need for Tier 2/3)                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ├─ If match: RETURN (is_correct=True, confidence=1.0)
  │
  └─ If no match: Continue to Tier 2
      │
      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: SEMANTIC VERIFICATION (ORACLE)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Use LLM to evaluate semantic correctness                                    │
│                                                                              │
│  Oracle.score_task(task, agent_response, correct_answer)                    │
│    │                                                                         │
│    ├─ Construct prompt:                                                      │
│    │   "Task: NOT(5 + 3) = ?"                                                │
│    │   "Agent answer: -8"                                                    │
│    │   "Correct answer: -8"                                                  │
│    │   "Is the agent's answer correct? Answer YES or NO."                    │
│    │                                                                         │
│    ├─ Call HuggingFace API                                                   │
│    │   POST https://api-inference.huggingface.co/v1/chat/completions        │
│    │   Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0                             │
│    │   Temperature: 0.1 (deterministic)                                      │
│    │                                                                         │
│    ├─ Parse response:                                                        │
│    │   Response: "YES, the answer is correct."                               │
│    │   Extract: "YES" → is_correct = True                                    │
│    │                                                                         │
│    └─ Compute pass_rate over multiple attempts:                              │
│        ├─ Attempt 1: YES (1/1)                                               │
│        ├─ Attempt 2: YES (2/2)                                               │
│        ├─ Attempt 3: YES (3/3)                                               │
│        └─ pass_rate = 3/3 = 1.0 → confidence = 1.0                           │
│                                                                              │
│  Result:                                                                     │
│    ├─ is_correct: True                                                       │
│    ├─ confidence: 0.7-1.0 (based on pass_rate)                               │
│    ├─ feedback: "Oracle verified: Correct"                                   │
│    └─ Return (no need for Tier 3 if confidence > 0.7)                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ├─ If confidence > 0.7: RETURN (is_correct, confidence)
  │
  └─ If confidence ≤ 0.7: Continue to Tier 3
      │
      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: CONSISTENCY CHECK                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Check if agent gives consistent answers across multiple attempts            │
│                                                                              │
│  Consistency window (last 5 attempts):                                       │
│    ├─ Attempt 1: "-8"                                                        │
│    ├─ Attempt 2: "-8"                                                        │
│    ├─ Attempt 3: "-8"                                                        │
│    ├─ Attempt 4: "-8"                                                        │
│    └─ Attempt 5: "-8"                                                        │
│                                                                              │
│  Entropy calculation:                                                        │
│    ├─ Unique responses: 1 ("-8")                                             │
│    ├─ Frequency: 5/5 = 1.0                                                   │
│    ├─ Entropy: -1.0 * log(1.0) = 0.0 (perfect consistency)                   │
│    └─ Consistency score: 1.0 - entropy = 1.0                                 │
│                                                                              │
│  Result:                                                                     │
│    ├─ is_correct: True (if consistency > 0.8)                                │
│    ├─ confidence: 0.8-1.0 (based on consistency)                             │
│    ├─ feedback: "Consistent response pattern detected"                       │
│    └─ FINAL DECISION                                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
FINAL VERIFICATION RESULT:
  ├─ is_correct: True/False
  ├─ confidence: 0.0-1.0
  ├─ feedback: "Correct!" / "Incorrect" / "Uncertain"
  └─ Return to EpisodeHandler
```


---

## REWARD CALCULATION FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REWARD DECOMPOSITION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

Input: (is_correct, turn_number, attempt_history)
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 1: R_TASK (Task Correctness)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Reward for solving the task correctly                                       │
│                                                                              │
│  Calculation:                                                                │
│    if is_correct:                                                            │
│      if turn_number == 1:                                                    │
│        R_task = 1.0  (solved on first attempt)                               │
│      elif turn_number == 2:                                                  │
│        R_task = 0.8  (solved after feedback)                                 │
│      elif turn_number == 3:                                                  │
│        R_task = 0.5  (solved with hint)                                      │
│      else:  # turn_number >= 4                                               │
│        R_task = 0.2  (solved on final attempt)                               │
│    else:                                                                     │
│      R_task = 0.0  (not solved)                                              │
│                                                                              │
│  Example:                                                                    │
│    is_correct: True, turn_number: 1                                          │
│    R_task = 1.0                                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 2: R_EFFICIENCY (Turn Efficiency)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Reward for solving in fewer turns                                           │
│                                                                              │
│  Calculation:                                                                │
│    R_efficiency = 1.0 - (turn_number / max_turns)                            │
│    where max_turns = 4                                                       │
│                                                                              │
│  Examples:                                                                   │
│    Turn 1: R_efficiency = 1.0 - (1/4) = 0.75                                 │
│    Turn 2: R_efficiency = 1.0 - (2/4) = 0.50                                 │
│    Turn 3: R_efficiency = 1.0 - (3/4) = 0.25                                 │
│    Turn 4: R_efficiency = 1.0 - (4/4) = 0.00                                 │
│                                                                              │
│  Intuition: Faster solutions are better                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 3: R_CONSISTENCY (Response Stability)                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Reward for consistent, stable responses                                     │
│                                                                              │
│  Calculation:                                                                │
│    1. Collect last 5 responses in window                                     │
│    2. Count unique responses                                                 │
│    3. Calculate entropy:                                                     │
│       entropy = -Σ(p_i * log(p_i))                                           │
│       where p_i = frequency of response i                                    │
│    4. R_consistency = 1.0 - entropy                                          │
│                                                                              │
│  Examples:                                                                   │
│    Window: ["-8", "-8", "-8", "-8", "-8"]                                    │
│      Unique: 1, p_1 = 1.0                                                    │
│      Entropy: -1.0 * log(1.0) = 0.0                                          │
│      R_consistency = 1.0 - 0.0 = 1.0 (perfect consistency)                   │
│                                                                              │
│    Window: ["-8", "-7", "-8", "-9", "-8"]                                    │
│      Unique: 3, p_1=0.6, p_2=0.2, p_3=0.2                                    │
│      Entropy: -(0.6*log(0.6) + 0.2*log(0.2) + 0.2*log(0.2)) ≈ 0.95          │
│      R_consistency = 1.0 - 0.95 = 0.05 (low consistency)                     │
│                                                                              │
│  Intuition: Consistent responses indicate confidence                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ FINAL REWARD CALCULATION                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Weighted sum of components:                                                 │
│                                                                              │
│  R_final = w1 * R_task + w2 * R_efficiency + w3 * R_consistency              │
│                                                                              │
│  Default weights:                                                            │
│    w1 = 0.6  (60% - task correctness is most important)                      │
│    w2 = 0.2  (20% - efficiency matters)                                      │
│    w3 = 0.2  (20% - consistency matters)                                     │
│                                                                              │
│  Example calculation:                                                        │
│    Scenario: Correct on Turn 1, perfect consistency                          │
│    R_task = 1.0                                                              │
│    R_efficiency = 0.75                                                       │
│    R_consistency = 1.0                                                       │
│                                                                              │
│    R_final = 0.6*1.0 + 0.2*0.75 + 0.2*1.0                                    │
│            = 0.6 + 0.15 + 0.2                                                │
│            = 0.95                                                            │
│                                                                              │
│  Another example:                                                            │
│    Scenario: Correct on Turn 4, low consistency                              │
│    R_task = 0.2                                                              │
│    R_efficiency = 0.0                                                        │
│    R_consistency = 0.1                                                       │
│                                                                              │
│    R_final = 0.6*0.2 + 0.2*0.0 + 0.2*0.1                                     │
│            = 0.12 + 0.0 + 0.02                                               │
│            = 0.14                                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
Output: R_final (0.0 - 1.0)
  └─ Used for training signal in DPO trainer
```


---

## CURRICULUM PROGRESSION FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CURRICULUM MANAGER STATE MACHINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

Skill Domain: negation_arithmetic

┌──────────────────────────────────────────────────────────────────────────────┐
│ EASY TIER                                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Task Complexity: Simple negation + single operation                         │
│  Example: "NOT(5 + 3) = ?"                                                   │
│                                                                              │
│  Accuracy Tracking:                                                          │
│    ├─ Episode 1: Correct ✓ (1/1 = 100%)                                      │
│    ├─ Episode 2: Correct ✓ (2/2 = 100%)                                      │
│    ├─ Episode 3: Correct ✓ (3/3 = 100%)                                      │
│    ├─ Episode 4: Correct ✓ (4/4 = 100%)                                      │
│    ├─ Episode 5: Correct ✓ (5/5 = 100%)                                      │
│    └─ Accuracy: 100% > 60% threshold ✓                                       │
│                                                                              │
│  Promotion Decision:                                                         │
│    └─ PROMOTE TO MEDIUM TIER                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  │ (Accuracy > 75%)
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ MEDIUM TIER                                                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Task Complexity: Multiple operations + negation                             │
│  Example: "NOT((5 + 3) * 2) = ?"                                             │
│                                                                              │
│  Accuracy Tracking:                                                          │
│    ├─ Episode 6: Correct ✓ (1/1 = 100%)                                      │
│    ├─ Episode 7: Incorrect ✗ (1/2 = 50%)                                     │
│    ├─ Episode 8: Correct ✓ (2/3 = 67%)                                       │
│    ├─ Episode 9: Correct ✓ (3/4 = 75%)                                       │
│    ├─ Episode 10: Correct ✓ (4/5 = 80%)                                      │
│    └─ Accuracy: 80% > 75% threshold ✓                                        │
│                                                                              │
│  Promotion Decision:                                                         │
│    └─ PROMOTE TO HARD TIER                                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  │ (Accuracy > 85%)
  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ HARD TIER                                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Task Complexity: Complex nested operations                                  │
│  Example: "NOT(NOT(5 + 3) - 2) = ?"                                          │
│                                                                              │
│  Accuracy Tracking:                                                          │
│    ├─ Episode 11: Incorrect ✗ (0/1 = 0%)                                     │
│    ├─ Episode 12: Incorrect ✗ (0/2 = 0%)                                     │
│    ├─ Episode 13: Correct ✓ (1/3 = 33%)                                      │
│    ├─ Episode 14: Correct ✓ (2/4 = 50%)                                      │
│    ├─ Episode 15: Incorrect ✗ (2/5 = 40%)                                    │
│    └─ Accuracy: 40% < 50% threshold ✗                                        │
│                                                                              │
│  Demotion Decision:                                                          │
│    └─ DEMOTE TO MEDIUM TIER                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  │ (Accuracy < 50%)
  ▼
BACK TO MEDIUM TIER (Curriculum loop continues)

┌──────────────────────────────────────────────────────────────────────────────┐
│ CURRICULUM STATE FILE (curriculum_state.json)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  {                                                                           │
│    "negation_arithmetic": {                                                  │
│      "current_tier": "medium",                                               │
│      "accuracy": 0.75,                                                       │
│      "correct_count": 3,                                                     │
│      "total_count": 4,                                                       │
│      "last_updated": "2026-04-25T10:30:00Z"                                  │
│    }                                                                         │
│  }                                                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```


---

## FAILURE BANK & REPAIR MODE FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FAILURE BANK LIFECYCLE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

HUNT MODE (New Tasks)
  │
  ├─ Episode 1: Task "NOT(5 + 3) = ?"
  │   ├─ Agent response: "-7" (INCORRECT)
  │   ├─ Correct answer: "-8"
  │   └─ Action: Record failure
  │
  ├─ Episode 2: Task "NOT((5 + 3) * 2) = ?"
  │   ├─ Agent response: "16" (INCORRECT)
  │   ├─ Correct answer: "-16"
  │   └─ Action: Record failure
  │
  └─ Episode 3: Task "NOT(NOT(5 + 3) - 2) = ?"
      ├─ Agent response: "10" (INCORRECT)
      ├─ Correct answer: "-10"
      └─ Action: Record failure

┌──────────────────────────────────────────────────────────────────────────────┐
│ FAILURE BANK (failure_bank.jsonl)                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Line 1:                                                                      │
│ {                                                                            │
│   "task": "NOT(5 + 3) = ?",                                                  │
│   "agent_response": "-7",                                                    │
│   "correct_answer": "-8",                                                    │
│   "timestamp": "2026-04-25T10:15:00Z",                                       │
│   "skill_domain": "negation_arithmetic",                                     │
│   "difficulty_tier": "easy"                                                  │
│ }                                                                            │
│                                                                              │
│ Line 2:                                                                      │
│ {                                                                            │
│   "task": "NOT((5 + 3) * 2) = ?",                                            │
│   "agent_response": "16",                                                    │
│   "correct_answer": "-16",                                                   │
│   "timestamp": "2026-04-25T10:20:00Z",                                       │
│   "skill_domain": "negation_arithmetic",                                     │
│   "difficulty_tier": "medium"                                                │
│ }                                                                            │
│                                                                              │
│ Line 3:                                                                      │
│ {                                                                            │
│   "task": "NOT(NOT(5 + 3) - 2) = ?",                                         │
│   "agent_response": "10",                                                    │
│   "correct_answer": "-10",                                                   │
│   "timestamp": "2026-04-25T10:25:00Z",                                       │
│   "skill_domain": "negation_arithmetic",                                     │
│   "difficulty_tier": "hard"                                                  │
│ }                                                                            │
│                                                                              │
│ Total failures: 3                                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
REPAIR MODE (Learning from Failures)
  │
  ├─ Episode 4: REPAIR MODE
  │   ├─ TaskGenerator.generate_task(mode="repair")
  │   │   └─ Select from failure bank: Line 1
  │   │
  │   ├─ Task prompt (repair wrapper):
  │   │   "Fix this broken attempt:"
  │   │   "Original task: NOT(5 + 3) = ?"
  │   │   "Broken attempt: -7"
  │   │   "Correct answer: -8"
  │   │   "What was the error? Provide the correct answer."
  │   │
  │   ├─ Agent response: "-8" (CORRECT)
  │   ├─ Verification: PASS ✓
  │   └─ Reward: 0.8 (repair mode bonus)
  │
  ├─ Episode 5: REPAIR MODE
  │   ├─ TaskGenerator.generate_task(mode="repair")
  │   │   └─ Select from failure bank: Line 2
  │   │
  │   ├─ Task prompt (repair wrapper):
  │   │   "Fix this broken attempt:"
  │   │   "Original task: NOT((5 + 3) * 2) = ?"
  │   │   "Broken attempt: 16"
  │   │   "Correct answer: -16"
  │   │   "What was the error? Provide the correct answer."
  │   │
  │   ├─ Agent response: "-16" (CORRECT)
  │   ├─ Verification: PASS ✓
  │   └─ Reward: 0.8 (repair mode bonus)
  │
  └─ Episode 6: REPAIR MODE
      ├─ TaskGenerator.generate_task(mode="repair")
      │   └─ Select from failure bank: Line 3
      │
      ├─ Task prompt (repair wrapper):
      │   "Fix this broken attempt:"
      │   "Original task: NOT(NOT(5 + 3) - 2) = ?"
      │   "Broken attempt: 10"
      │   "Correct answer: -10"
      │   "What was the error? Provide the correct answer."
      │
      ├─ Agent response: "-10" (CORRECT)
      ├─ Verification: PASS ✓
      └─ Reward: 0.8 (repair mode bonus)

┌──────────────────────────────────────────────────────────────────────────────┐
│ REPAIR MODE BENEFITS                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Targeted Learning:                                                       │
│     - Agent learns from its own mistakes                                     │
│     - Focuses on failure patterns                                            │
│                                                                              │
│  2. Curriculum Reinforcement:                                                │
│     - Revisits failed tasks at same difficulty                               │
│     - Builds confidence before advancing                                     │
│                                                                              │
│  3. Data Efficiency:                                                         │
│     - Reuses failure data for training                                       │
│     - Reduces need for new task generation                                   │
│                                                                              │
│  4. Co-evolutionary Loop:                                                    │
│     - Agent failures → Repair tasks → Agent improvement                      │
│     - Continuous feedback loop                                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```


---

## TRAINING DATA GENERATION & DPO FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING DATA PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

EPISODE COLLECTION PHASE
  │
  ├─ Episode 1: Hunt Mode
  │   ├─ Turn 1: response="-8", reward=0.95 (BEST)
  │   ├─ Turn 2: response="-8", reward=0.85
  │   ├─ Turn 3: response="-8", reward=0.60
  │   └─ Turn 4: response="-8", reward=0.40 (WORST)
  │
  ├─ Episode 2: Hunt Mode
  │   ├─ Turn 1: response="-16", reward=0.90 (BEST)
  │   ├─ Turn 2: response="-16", reward=0.80
  │   ├─ Turn 3: response="-16", reward=0.50
  │   └─ Turn 4: response="-16", reward=0.30 (WORST)
  │
  └─ Episode 3: Repair Mode
      ├─ Turn 1: response="-10", reward=0.92 (BEST)
      ├─ Turn 2: response="-10", reward=0.82
      ├─ Turn 3: response="-10", reward=0.52
      └─ Turn 4: response="-10", reward=0.32 (WORST)

┌──────────────────────────────────────────────────────────────────────────────┐
│ TRAINING PAIR EXTRACTION                                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ For each episode:                                                            │
│   1. Identify highest reward response (chosen)                               │
│   2. Identify lowest reward response (rejected)                              │
│   3. Create training pair                                                    │
│                                                                              │
│ Episode 1 → Training Pair 1:                                                 │
│   {                                                                          │
│     "prompt": "NOT(5 + 3) = ?",                                              │
│     "chosen": "-8",        (reward=0.95)                                     │
│     "rejected": "-8"       (reward=0.40)                                     │
│   }                                                                          │
│                                                                              │
│ Episode 2 → Training Pair 2:                                                 │
│   {                                                                          │
│     "prompt": "NOT((5 + 3) * 2) = ?",                                        │
│     "chosen": "-16",       (reward=0.90)                                     │
│     "rejected": "-16"      (reward=0.30)                                     │
│   }                                                                          │
│                                                                              │
│ Episode 3 → Training Pair 3:                                                 │
│   {                                                                          │
│     "prompt": "NOT(NOT(5 + 3) - 2) = ?",                                     │
│     "chosen": "-10",       (reward=0.92)                                     │
│     "rejected": "-10"      (reward=0.32)                                     │
│   }                                                                          │
│                                                                              │
│ Note: In this example, chosen==rejected (same answer, different rewards)     │
│       In real scenarios, agent may give different answers per turn           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
DATASET ACCUMULATION
  │
  ├─ Collect 20 episodes (default for HF Spaces)
  ├─ Generate 20 training pairs
  └─ Format as HuggingFace Dataset

┌──────────────────────────────────────────────────────────────────────────────┐
│ DATASET FORMAT (for DPO Trainer)                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ [                                                                            │
│   {                                                                          │
│     "prompt": "NOT(5 + 3) = ?",                                              │
│     "chosen": "-8",                                                          │
│     "rejected": "-8"                                                         │
│   },                                                                         │
│   {                                                                          │
│     "prompt": "NOT((5 + 3) * 2) = ?",                                        │
│     "chosen": "-16",                                                         │
│     "rejected": "-16"                                                        │
│   },                                                                         │
│   ...                                                                        │
│   {                                                                          │
│     "prompt": "NOT(NOT(5 + 3) - 2) = ?",                                     │
│     "chosen": "-10",                                                         │
│     "rejected": "-10"                                                        │
│   }                                                                          │
│ ]                                                                            │
│                                                                              │
│ Total: 20 training pairs                                                     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
DPO TRAINING PHASE
  │
  ├─ Initialize TRL DPOTrainer
  │   ├─ Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  │   ├─ LoRA: rank=8, alpha=8
  │   ├─ Learning rate: 5e-6
  │   ├─ Batch size: 1
  │   ├─ Gradient accumulation: 2
  │   └─ Epochs: 1
  │
  ├─ For each training pair:
  │   ├─ Tokenize prompt
  │   ├─ Tokenize chosen response
  │   ├─ Tokenize rejected response
  │   ├─ Compute DPO loss:
  │   │   loss = -log(sigmoid(β * (log_prob_chosen - log_prob_rejected)))
  │   │   where β = 0.5 (temperature parameter)
  │   ├─ Backpropagate
  │   └─ Update LoRA weights
  │
  └─ Save trained model to ./output/alice-trained

┌──────────────────────────────────────────────────────────────────────────────┐
│ DPO LOSS EXPLANATION                                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ DPO (Direct Preference Optimization) Loss:                                   │
│                                                                              │
│ L_DPO = -log(sigmoid(β * (log_prob_chosen - log_prob_rejected)))             │
│                                                                              │
│ Components:                                                                  │
│   - log_prob_chosen: Log probability of chosen response                      │
│   - log_prob_rejected: Log probability of rejected response                  │
│   - β: Temperature (controls preference strength)                            │
│   - sigmoid: Converts difference to probability                              │
│                                                                              │
│ Intuition:                                                                   │
│   - Maximize: log_prob_chosen (make chosen more likely)                      │
│   - Minimize: log_prob_rejected (make rejected less likely)                  │
│   - Difference: Larger gap = stronger preference signal                      │
│                                                                              │
│ Example:                                                                     │
│   Chosen response: "-8" (log_prob = -0.5)                                    │
│   Rejected response: "-8" (log_prob = -0.8)                                  │
│   Difference: -0.5 - (-0.8) = 0.3                                            │
│   Loss: -log(sigmoid(0.5 * 0.3)) = -log(sigmoid(0.15)) ≈ 0.65              │
│                                                                              │
│ Training effect:                                                             │
│   - Model learns to prefer high-reward responses                             │
│   - Model learns to avoid low-reward responses                               │
│   - Improves task-solving ability over time                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
MODEL CHECKPOINT SAVING
  │
  ├─ Save every 50 steps
  ├─ Save final model to ./output/alice-trained
  └─ Model ready for inference/evaluation

┌──────────────────────────────────────────────────────────────────────────────┐
│ TRAINING METRICS (Logged during training)                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ [TRAIN] Step 1/20:                                                           │
│   loss: 0.65                                                                 │
│   learning_rate: 5e-6                                                        │
│                                                                              │
│ [TRAIN] Step 5/20:                                                           │
│   loss: 0.58                                                                 │
│   learning_rate: 5e-6                                                        │
│                                                                              │
│ [TRAIN] Step 10/20:                                                          │
│   loss: 0.52                                                                 │
│   learning_rate: 5e-6                                                        │
│                                                                              │
│ [TRAIN] Step 15/20:                                                          │
│   loss: 0.48                                                                 │
│   learning_rate: 5e-6                                                        │
│                                                                              │
│ [TRAIN] Step 20/20:                                                          │
│   loss: 0.45                                                                 │
│   learning_rate: 5e-6                                                        │
│                                                                              │
│ Training completed successfully!                                             │
│ Model saved to: ./output/alice-trained                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```


---

## COMPLETE ARROW REFERENCE & INTERACTION MATRIX

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALL 16 SYSTEM ARROWS SUMMARY                             │
└─────────────────────────────────────────────────────────────────────────────┘

ARROW 1: AGENT → ENVIRONMENT (Action Submission)
  ├─ Type: AliceAction
  ├─ Data: {response, mode, task_id}
  ├─ Frequency: Once per turn (up to 5 times per episode)
  ├─ Trigger: Agent generates response
  └─ Example: {response: "-8", mode: "hunt", task_id: "uuid-1234"}

ARROW 2: ENVIRONMENT → AGENT (Observation)
  ├─ Type: AliceObservation
  ├─ Data: {task, skill_domain, difficulty_tier, turn_number, hint, reward, done, feedback, task_id}
  ├─ Frequency: Once per turn
  ├─ Trigger: After environment processes action
  └─ Example: {task: "NOT(5+3)=?", turn_number: 1, reward: 0.85, done: False}

ARROW 3: ENVIRONMENT → EPISODE_HANDLER (Turn Processing)
  ├─ Type: AliceAction
  ├─ Method: handle_turn(action, state)
  ├─ Frequency: Once per turn
  ├─ Purpose: Delegate turn-specific logic
  └─ Returns: Updated AliceState

ARROW 4: EPISODE_HANDLER → TASK_GENERATOR (Task Request)
  ├─ Type: Task request with mode (hunt/repair)
  ├─ Frequency: Once per episode
  ├─ Purpose: Get next task
  └─ Returns: Task(prompt, correct_answer)

ARROW 5: TASK_GENERATOR → FAILURE_BANK (Query)
  ├─ Type: Query for repair mode tasks
  ├─ Frequency: When mode="repair"
  ├─ Purpose: Retrieve failed tasks
  └─ Returns: FailureRecord

ARROW 6: TASK_GENERATOR → CURRICULUM_MANAGER (Difficulty Query)
  ├─ Type: Query for skill domain
  ├─ Frequency: When mode="hunt"
  ├─ Purpose: Get current difficulty tier
  └─ Returns: Tier (easy/medium/hard)

ARROW 7: CURRICULUM_MANAGER → CURRICULUM_STATE (Persistence)
  ├─ Type: Read/Write JSON
  ├─ File: curriculum_state.json
  ├─ Frequency: On load and after each episode
  └─ Purpose: Persist accuracy/tier tracking

ARROW 8: EPISODE_HANDLER → VERIFIER_STACK (Verification Request)
  ├─ Type: (task, agent_response, correct_answer)
  ├─ Frequency: Once per turn
  ├─ Purpose: Verify if response is correct
  └─ Returns: (is_correct, feedback, confidence)

ARROW 9: VERIFIER_STACK → ORACLE (Semantic Check)
  ├─ Type: (task, agent_response, correct_answer)
  ├─ Frequency: When Tier 1 fails
  ├─ Purpose: Use LLM to evaluate
  └─ Returns: Correctness score (0.0-1.0)

ARROW 10: ORACLE → HF_API (Model Query)
  ├─ Type: HTTP POST
  ├─ Endpoint: https://api-inference.huggingface.co/v1
  ├─ Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  ├─ Frequency: When Tier 2 needed
  └─ Returns: Model evaluation

ARROW 11: EPISODE_HANDLER → REWARD_CALCULATOR (Reward Computation)
  ├─ Type: (is_correct, turn_number, consistency_score)
  ├─ Frequency: Once per turn
  ├─ Purpose: Calculate training reward
  └─ Returns: R_final (float)

ARROW 12: EPISODE_HANDLER → FAILURE_BANK (Failure Recording)
  ├─ Type: FailureRecord
  ├─ File: failure_bank.jsonl
  ├─ Frequency: When agent fails verification
  ├─ Purpose: Store failures for repair mode
  └─ Format: Append-only JSONL

ARROW 13: EPISODE_HANDLER → CURRICULUM_MANAGER (Outcome Recording)
  ├─ Type: (skill_domain, correct: bool)
  ├─ Frequency: At episode end
  ├─ Purpose: Update accuracy tracking
  └─ Effect: May trigger tier promotion/demotion

ARROW 14: EPISODE_HANDLER → GRADIO_DASHBOARD (Logging)
  ├─ Type: Episode summary
  ├─ Data: {episode_id, success, reward, difficulty}
  ├─ Frequency: Once per episode
  └─ Purpose: Real-time monitoring

ARROW 15: TRAINING_SCRIPT → ENVIRONMENT (Episode Loop)
  ├─ Type: Multiple reset() + step() calls
  ├─ Frequency: Once per episode in training
  ├─ Purpose: Generate training data
  └─ Count: 20 episodes (default for HF Spaces)

ARROW 16: ENVIRONMENT → TRAINING_SCRIPT (Dataset)
  ├─ Type: (prompt, chosen_response, rejected_response)
  ├─ Frequency: After all episodes collected
  ├─ Purpose: Provide training pairs for DPO
  └─ Format: HuggingFace Dataset

┌──────────────────────────────────────────────────────────────────────────────┐
│ INTERACTION MATRIX                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ FROM                    TO                      ARROW   DATA TYPE            │
│ ─────────────────────────────────────────────────────────────────────────   │
│ Agent                   Environment             1       AliceAction          │
│ Environment             Agent                   2       AliceObservation     │
│ Environment             EpisodeHandler          3       AliceAction          │
│ EpisodeHandler          TaskGenerator           4       Query                │
│ TaskGenerator           FailureBank             5       Query                │
│ TaskGenerator           CurriculumManager       6       Query                │
│ CurriculumManager       CurriculumState         7       JSON                 │
│ EpisodeHandler          VerifierStack           8       Verification         │
│ VerifierStack           Oracle                  9       Evaluation           │
│ Oracle                  HF API                  10      HTTP POST            │
│ EpisodeHandler          RewardCalculator        11      Computation          │
│ EpisodeHandler          FailureBank             12      FailureRecord        │
│ EpisodeHandler          CurriculumManager       13      Outcome              │
│ EpisodeHandler          GradioDashboard         14      Summary              │
│ TrainingScript          Environment             15      Episodes             │
│ Environment             TrainingScript          16      Dataset              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ CRITICAL PATHS (Most Important Flows)                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ PATH 1: Agent Action Processing                                              │
│   Agent → (Arrow 1) → Environment → (Arrow 3) → EpisodeHandler              │
│   → (Arrow 8) → VerifierStack → (Arrow 9) → Oracle → (Arrow 10) → HF API    │
│   → (Arrow 11) → RewardCalculator → (Arrow 2) → Agent                       │
│                                                                              │
│ PATH 2: Curriculum Progression                                               │
│   EpisodeHandler → (Arrow 13) → CurriculumManager → (Arrow 7) → State       │
│   → (Arrow 6) → TaskGenerator → (Arrow 4) → EpisodeHandler                  │
│                                                                              │
│ PATH 3: Failure Learning                                                     │
│   EpisodeHandler → (Arrow 12) → FailureBank → (Arrow 5) → TaskGenerator     │
│   → (Arrow 4) → EpisodeHandler (repair mode)                                │
│                                                                              │
│ PATH 4: Training Data Generation                                             │
│   TrainingScript → (Arrow 15) → Environment → (Arrow 16) → TrainingScript   │
│   → DPO Trainer → Trained Model                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```


---

## QUICK REFERENCE: COMPONENT ROLES & RESPONSIBILITIES

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT RESPONSIBILITY MATRIX                          │
└─────────────────────────────────────────────────────────────────────────────┘

AGENT (LLM Model)
  ├─ Receives: AliceObservation (task, difficulty, feedback)
  ├─ Generates: AliceAction (response, mode, task_id)
  ├─ Trained by: DPO trainer using collected episodes
  └─ Goal: Maximize reward by solving tasks correctly

ENVIRONMENT (AliceEnvironment)
  ├─ Orchestrates: Full RL loop (reset, step)
  ├─ Maintains: AliceState (episode state)
  ├─ Delegates: Turn logic to EpisodeHandler
  ├─ Collects: Training data for DPO
  └─ Goal: Provide consistent RL interface

EPISODE HANDLER (EpisodeHandler)
  ├─ Manages: 5-turn episode flow
  ├─ Coordinates: All verification and reward logic
  ├─ Records: Failures and outcomes
  ├─ Logs: Episode data to dashboard
  └─ Goal: Execute turn-by-turn episode logic

TASK GENERATOR (TaskGenerator)
  ├─ Creates: Hunt mode tasks (new from curriculum)
  ├─ Creates: Repair mode tasks (from failure bank)
  ├─ Queries: Curriculum for difficulty
  ├─ Queries: Failure bank for failures
  └─ Goal: Provide appropriate tasks for agent

CURRICULUM MANAGER (CurriculumManager)
  ├─ Tracks: Accuracy per skill domain
  ├─ Manages: Tier progression (easy → medium → hard)
  ├─ Decides: Promotion/demotion based on accuracy
  ├─ Persists: State to curriculum_state.json
  └─ Goal: Adapt task difficulty to agent ability

FAILURE BANK (FailureBank)
  ├─ Stores: Failed attempts (task, response, answer)
  ├─ Persists: Data to failure_bank.jsonl
  ├─ Provides: Repair mode tasks
  ├─ Tracks: Failure patterns
  └─ Goal: Enable learning from mistakes

ORACLE (Oracle)
  ├─ Evaluates: Semantic correctness of responses
  ├─ Calls: HuggingFace inference API
  ├─ Computes: Pass rates over multiple attempts
  ├─ Provides: Confidence scores
  └─ Goal: Verify responses when exact match fails

VERIFIER STACK (VerifierStack)
  ├─ Tier 1: Exact string matching
  ├─ Tier 2: Semantic verification via Oracle
  ├─ Tier 3: Consistency checking
  ├─ Combines: Results into final decision
  └─ Goal: Robust correctness verification

REWARD CALCULATOR (RewardCalculator)
  ├─ Computes: R_task (correctness)
  ├─ Computes: R_efficiency (turn number)
  ├─ Computes: R_consistency (response stability)
  ├─ Combines: Weighted sum into R_final
  └─ Goal: Provide training signal for DPO

GRADIO DASHBOARD (GradioDashboard)
  ├─ Logs: Episode summaries
  ├─ Displays: Real-time statistics
  ├─ Tracks: Success rates by difficulty
  ├─ Monitors: Failure bank size
  └─ Goal: Real-time training visualization

TRAINING SCRIPT (train.py)
  ├─ Generates: 20 episodes via environment
  ├─ Collects: Training pairs (prompt, chosen, rejected)
  ├─ Initializes: DPO trainer with TRL
  ├─ Trains: Model using collected data
  └─ Goal: Optimize model via DPO

HUGGINGFACE API
  ├─ Provides: TinyLlama model inference
  ├─ Evaluates: Task correctness
  ├─ Returns: Model predictions
  └─ Goal: External semantic verification

┌──────────────────────────────────────────────────────────────────────────────┐
│ DATA FLOW SUMMARY                                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ INPUT SOURCES:                                                               │
│   ├─ Agent: Generates responses to tasks                                     │
│   ├─ Curriculum: Provides difficulty progression                            │
│   ├─ Failure Bank: Provides repair tasks                                     │
│   └─ HF API: Provides semantic verification                                  │
│                                                                              │
│ PROCESSING STAGES:                                                           │
│   ├─ Task Generation: Hunt or Repair mode                                    │
│   ├─ Verification: 3-tier verification stack                                 │
│   ├─ Reward Calculation: Multi-component reward                              │
│   ├─ Curriculum Update: Accuracy tracking & tier progression                 │
│   └─ Failure Recording: Store failures for repair mode                       │
│                                                                              │
│ OUTPUT DESTINATIONS:                                                         │
│   ├─ Agent: Observations with feedback                                       │
│   ├─ Dashboard: Episode logs and statistics                                  │
│   ├─ Curriculum State: Persisted tier/accuracy                               │
│   ├─ Failure Bank: Stored failures                                           │
│   └─ Training Script: Dataset for DPO                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ TIMING & FREQUENCY                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ PER TURN (5 times per episode):                                              │
│   ├─ Agent submits action (Arrow 1)                                          │
│   ├─ Environment processes (Arrow 3)                                         │
│   ├─ Verification occurs (Arrows 8-10)                                       │
│   ├─ Reward calculated (Arrow 11)                                            │
│   └─ Observation returned (Arrow 2)                                          │
│                                                                              │
│ PER EPISODE (20 times per training):                                         │
│   ├─ Task generated (Arrow 4)                                                │
│   ├─ Curriculum queried (Arrow 6)                                            │
│   ├─ Failures recorded (Arrow 12)                                            │
│   ├─ Outcomes recorded (Arrow 13)                                            │
│   ├─ Dashboard logged (Arrow 14)                                             │
│   └─ Training pair collected (Arrow 16)                                      │
│                                                                              │
│ PER TRAINING RUN (once):                                                     │
│   ├─ 20 episodes collected (Arrow 15)                                        │
│   ├─ Dataset formatted (Arrow 16)                                            │
│   ├─ DPO trainer initialized                                                 │
│   ├─ Model trained (1 epoch)                                                 │
│   └─ Model saved                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## SUMMARY

This comprehensive diagram explains the ALICE RL environment architecture with:

1. **16 System Arrows** - All data flows between components
2. **5-Turn Episode Flow** - Complete turn-by-turn sequence
3. **3-Tier Verification** - Exact match → Semantic → Consistency
4. **Reward Decomposition** - Task + Efficiency + Consistency
5. **Curriculum Progression** - Easy → Medium → Hard tiers
6. **Failure Bank & Repair** - Learning from mistakes
7. **Training Data Pipeline** - Episode collection → DPO training
8. **Component Responsibilities** - Each component's role
9. **Critical Paths** - Most important data flows
10. **Timing & Frequency** - When each interaction occurs

All components work together in a co-evolutionary loop where:
- Agent learns from tasks
- Tasks adapt to agent ability (curriculum)
- Failures become repair tasks
- Episodes generate training data
- DPO training improves the model
- Cycle repeats with better agent
