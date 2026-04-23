"""
ALICE Environment — co-evolutionary RL environment for capability self-improvement.
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
from alice.server.episode_handler import EpisodeHandler
from alice.server.failure_bank import FailureBank
from alice.server.reward import RewardCalculator
from alice.server.task_generator import TaskGenerator
from alice.server.verifier import VerifierStack

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
FAILURE_BANK_PATH = os.environ.get("FAILURE_BANK_PATH", "failure_bank.jsonl")
CURRICULUM_STATE_PATH = os.environ.get("CURRICULUM_STATE_PATH", "curriculum_state.json")

_degraded = not (HF_TOKEN and API_BASE_URL)
if _degraded:
    logger.warning(
        "ALICE running in degraded mode (no HF_TOKEN/API_BASE_URL) — "
        "Tier 1 only, template tasks, no Oracle discrimination scoring."
    )

_failure_bank = FailureBank(FAILURE_BANK_PATH)
_curriculum = CurriculumManager(CURRICULUM_STATE_PATH)
_reward_calc = RewardCalculator()

if not _degraded:
    from alice.server.oracle import Oracle
    _oracle = Oracle(API_BASE_URL, MODEL_NAME, HF_TOKEN)
else:
    _oracle = None

_task_gen = TaskGenerator(_failure_bank, _oracle)
_verifier = VerifierStack(_oracle)
_episode_handler = EpisodeHandler(
    _task_gen, _verifier, _reward_calc, _curriculum, _failure_bank
)

_response_window: deque = deque(maxlen=50)


def _compute_entropy(window: deque) -> float:
    if not window:
        return float("inf")
    counts: dict = {}
    for r in window:
        counts[r] = counts.get(r, 0) + 1
    total = len(window)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


class AliceEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = AliceState(
            episode_id=str(uuid4()),
            step_count=0,
            skill_domain="negation_arithmetic",
            difficulty_tier=_curriculum.get_tier("negation_arithmetic"),
            mode="hunt",
        )

    def reset(self) -> AliceObservation:
        self._state = AliceState(
            episode_id=str(uuid4()),
            step_count=0,
            skill_domain="negation_arithmetic",
            difficulty_tier=_curriculum.get_tier("negation_arithmetic"),
            mode="hunt",
        )
        obs = _episode_handler.start_episode("negation_arithmetic")
        self._state.task_id = obs.task_id
        return obs

    def step(self, action: AliceAction) -> AliceObservation:
        self._state.step_count += 1
        obs = _episode_handler.handle_turn(action, self._state)
        if obs.done:
            gradio_dashboard.log_episode(
                trajectory=[],
                reward_components=_reward_calc.decompose(0.0, 0.0, 0.0, 4, 0.0, 0),
                discrimination_score=None,
            )
            _response_window.append(action.response)
            entropy = _compute_entropy(_response_window)
            if entropy < 0.5:
                logger.warning(
                    f"Response entropy={entropy:.3f} bits — possible mode collapse detected"
                )
        return obs

    @property
    def state(self) -> AliceState:
        return self._state
