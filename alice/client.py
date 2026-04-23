# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Alice Environment Client — ALICE co-evolutionary RL environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AliceAction, AliceObservation


class AliceEnv(
    EnvClient[AliceAction, AliceObservation, State]
):
    """
    Client for the ALICE Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with AliceEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     print(obs.task)  # CoT-wrapped negation arithmetic task
        ...
        ...     action = AliceAction(response="Answer: 7", mode="hunt", task_id=obs.task_id)
        ...     result = client.step(action)
        ...     print(result.observation.feedback)

    Example with Docker:
        >>> client = AliceEnv.from_docker_image("alice-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     result = client.step(AliceAction(response="Answer: 7", mode="hunt", task_id=obs.task_id))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AliceAction) -> Dict:
        return {
            "response": action.response,
            "mode": action.mode,
            "task_id": action.task_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AliceObservation]:
        obs_data = payload.get("observation", {})
        observation = AliceObservation(
            task=obs_data.get("task", ""),
            skill_domain=obs_data.get("skill_domain", "negation_arithmetic"),
            difficulty_tier=obs_data.get("difficulty_tier", "easy"),
            turn_number=obs_data.get("turn_number", 0),
            hint=obs_data.get("hint"),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            feedback=obs_data.get("feedback", ""),
            task_id=obs_data.get("task_id", ""),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
