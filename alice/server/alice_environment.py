"""
ALICE
This module defines the AliceEnvironment class, a custom RL environment
for the ALICE project based on OpenEnv's interface.

Replaces the echo placeholder with the full ALICE co-evolutionary loop:
  - CurriculumManager: rolling accuracy → difficulty tier
  - Oracle: discrimination scoring via HF Inference API
  - TaskGenerator: Hunt/Repair mode task synthesis with CoT wrapping
  - EpisodeHandler: 5-turn FSM (Turn 0–4)
ogrammatic + Tier 2 LLM judge + Tier 3 regression
  - FailureBank: JSONL persistence + n-gram similarity
  - RewardCalculator: R_final formula with deco

L are absent, the server starts with
Tier 1 only, template-based tasks, and no Oracle discrimination scoring.
"""

import logging
import math
import os
from collections import deque
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from alice.models import AliceAction, AliceObservation, AliceState
from alice.server import gradio_dashboard
from alice.server.curriculum_manager import CurriculumManager
andler
from alice.server.failure_bank import FailureBank
from alice.server.reward import RewardCalculator
from alice.server.task_generator import TaskGenerator
from alice.server.verifier import VerifierStack

logger = logging.getLogger(__name__)

# --- Environment variable configuration ---
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
FAILUREBANK_PATH = os.environ.get("FAILURE_BANK_PATH", "failure_bank.jsonl")
.environ.get("CURRICULUM_STATE_PATH", "curriculum_state.json")

_degraded = not (HF_TOKEN and API_BASE_URL)
if _degraded:
    logger.warning(
        "ALICE running in degraded mode (no HKEN/API_BASE_URL) — "
racle discrimination scoring."
    )

# --- Module-level singletons (shared across all sessions) ---
_failure_bank = FailureBank(FAILURE_BANK_PATH)
_curriculum = CurriculumManager(CURRICULUM_STATE_PATH)
_reward_calc = RewardCalculator()

if not _degraded:
    from alice.server.oracle import Oracle
    _oracle = Oracle(API_BASE_URL, MODEL_NAME, HF_TOKEN)
else:
    _oracle = None  # type: ignore[assignment]

_task_gen = TaskGenerator(_failure_bank, _oracle)
_verifier = VerifierStack(_oracle)
_episode_handler = EpisodeHandler(
    _task_gen, _verifier, _reward_calc, _curriculum, _failure_bank
)

# Rolling window for entropy monitoring (last 50 responses)
_response_window: deque = deque(maxlen=50)


def _compute_entropy(window: deque) -> float:
    """Shannon entropy over response distribution in the window."""
    if not window:
        return float("inf")
    counts: dict[str, int] = {}
    for r in window:
        counts[r] = counts.get(r, 0) + 1
    total = len(window)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
 return entropy


ment):
    """
    ALICE co-evolutionary RL environment.

    Each instance handles one concurrent WebSocket session with its own
    AliceState. All instances share the module-level singletons (FailureBank,
    CurriculumManager, etc.) so the co-evolutionary loop accumulates state
    across all sessions.

    Example:
        >>> env = AliceEnvironment()
        >>> obs = env.reset()
        >>> print(obs.task)  # CoT-wrapped negation arithmetic task
        >>>
        >>> obs Answer: 7", mode="hunt", task_id=obs.task_id))
        >>> print(obs.feedback)  # "Correct! Well done."
    """

window.append(action.response)
            entropy = _compute_entropy(_response_window)
            if entropy < 0.5:
                logger.warning(
                    f"Response entropy={entropy:.3f} bits — possible mode collapse detected"
                )

        return obs

    @property
    def state(self) -> AliceState:
        return self._state

            gradio_dashboard.log_episode(
                trajectory=[],  # trajectory detail can be added later
                reward_components=_reward_calc.decompose(
                    r_programmatic=0.0,
                    r_judge=0.0,
                    r_regression=0.0,
                    turn_number=4,
                    similarity=0.0,
                    repeat_count=0,
                ),
                discrimination_score=None,
            )
            # Entropy monitoring
            _response_# Log to dashboardulty_tier=_curriculum.get_tier("negation_arithmetic"),
            mode="hunt",
        )
        obs = _episode_handler.start_episode("negation_arithmetic")
        # Sync task_id into state
        self._state.task_id = obs.task_id
        return obs

    def step(self, action: AliceAction) -> AliceObservation:
        """Execute one turn. Returns the resulting observation."""
        self._state.step_count += 1
        obs = _episode_handler.handle_turn(action, self._state)

        if obs.done:
            ,
            skill_domain="negation_arithmetic",
            difficCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = AliceState(
            episode_id=str(uuid4()),
            step_count=0,
            skill_domain="negation_arithmetic",
            difficulty_tier=_curriculum.get_tier("negation_arithmetic"),
            mode="hunt",
        )

    def reset(self) -> AliceObservation:
        """Start a fresh episode. Returns Turn 0 observation."""
        self._state = AliceState(
            episode_id=str(uuid4()),
            step_count=0    SUPPORTS_CON