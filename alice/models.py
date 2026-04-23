# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ALICE RL Environment.

ALICE (Adversarial Loop for Inter-model Co-evolutionary Environment) is a
reinforcement learning training environment that targets the negation + arithmetic
composition failure mode in LLMs via a co-evolutionary task curriculum.
"""

from typing import Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class AliceAction(Action):
    """Action submitted by the agent for the current task."""

    response: str = Field(..., description="The agent's answer text for the current task")
    mode: str = Field(default="hunt", description="Episode mode: 'hunt' or 'repair'")
    task_id: str = Field(..., description="UUID of the current task, echoed from observation")


class AliceObservation(Observation):
    """Observation returned to the agent after each step."""

    task: str = Field(default="", description="The task prompt presented to the agent")
    skill_domain: str = Field(default="negation_arithmetic", description="Skill domain of the task")
    difficulty_tier: str = Field(default="easy", description="Difficulty tier: easy | medium | hard")
    turn_number: int = Field(default=0, description="Current turn within the episode (0-4)")
    hint: Optional[str] = Field(default=None, description="Structural hint provided on Turn 3")
    reward: float = Field(default=0.0, description="R_final for this step")
    done: bool = Field(default=False, description="True when episode is complete (Turn 4)")
    feedback: str = Field(default="", description="Verification feedback from VerifierStack")
    task_id: str = Field(default="", description="UUID of the current task")


class AliceState(State):
    """Internal server-side state for a running ALICE episode."""

    episode_id: str = Field(default="", description="UUID for the current episode")
    step_count: int = Field(default=0, description="Number of steps taken in this episode")
    current_task: Optional[str] = Field(default=None, description="Current task prompt text")
    turn_number: int = Field(default=0, description="Current turn (0-4)")
    attempt_history: list[str] = Field(default_factory=list, description="Agent responses this episode")
    skill_domain: str = Field(default="negation_arithmetic", description="Skill domain")
    difficulty_tier: str = Field(default="easy", description="Difficulty tier")
    mode: str = Field(default="hunt", description="hunt | repair")
    failure_bank_size: int = Field(default=0, description="Current number of stored failures")
    task_id: str = Field(default="", description="UUID of the current task")
    correct_answer: str = Field(default="", description="Ground truth answer — internal, never sent to agent")
    hint: Optional[str] = Field(default=None, description="Hint text for Turn 3")
    done: bool = Field(default=False, description="True when episode is complete")
