"""
Episode Handler — manages the 5-turn CoT-scaffolded episode FSM.

Turn 0: reset() — task presentation
Turn 1: first attempt — Tier 1 feedback only
Turn 2: reflection prompt — CoT re-examination
Turn 3: hint + retry — Tier 1+2 feedback with structural hint
Turn 4: terminal scoring — full VerificationResult + R_final
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alice.server.curriculum_manager import CurriculumManager
    from alice.server.failure_bank import FailureBank
    from alice.server.oracle import Oracle
    from alice.server.reward import RewardCalculator
    from alice.server.task_generator import Task, TaskGenerator
    from alice.server.verifier import VerifierStack

from alice.models import AliceAction, AliceObservation, AliceState
from alice.server.failure_bank import FailureRecord

logger = logging.getLogger(__name__)


class EpisodeHandler:
    MAX_TURNS: int = 4

    def __init__(
        self,
        task_generator: "TaskGenerator",
        verifier: "VerifierStack",
        reward: "RewardCalculator",
        curriculum: "CurriculumManager",
        failure_bank: "FailureBank",
    ) -> None:
        self._task_gen = task_generator
        self._verifier = verifier
        self._reward = reward
        self._curriculum = curriculum
        self._failure_bank = failure_bank
        self._current_task: "Task | None" = None
        self._terminal_obs: AliceObservation | None = None

    def start_episode(
        self, skill_domain: str = "negation_arithmetic"
    ) -> AliceObservation:
        """Turn 0: generate task, return CoT-wrapped prompt."""
        tier = self._curriculum.get_tier(skill_domain)
        task = self._task_gen.generate(skill_domain, tier)
        self._current_task = task
        self._terminal_obs = None
        return AliceObservation(
            task=task.task_text,
            skill_domain=skill_domain,
            difficulty_tier=tier,
            turn_number=0,
            done=False,
            reward=0.0,
            feedback="Start: think step by step, then write Answer: <number>",
            task_id=task.task_id,
        )

    def handle_turn(
        self, action: AliceAction, state: AliceState
    ) -> AliceObservation:
        """Dispatch to the appropriate turn handler."""
        if state.done:
            return self._terminal_obs  # type: ignore[return-value]
        next_turn = state.turn_number + 1
        if next_turn == 1:
            return self._turn_1(action, state)
        elif next_turn == 2:
            return self._turn_2(action, state)
        elif next_turn == 3:
            return self._turn_3(action, state)
        else:
            return self._turn_4(action, state)

    def _turn_1(self, action: AliceAction, state: AliceState) -> AliceObservation:
        state.attempt_history.append(action.response)
        state.turn_number = 1
        r_prog = self._verifier._tier1_programmatic(self._current_task, action.response)
        if r_prog >= 1.0:
            feedback = "Correct! Well done."
        else:
            feedback = "Incorrect. Re-examine your reasoning chain — where did the negation distract you?"
        return AliceObservation(
            task=self._current_task.task_text,
            skill_domain=state.skill_domain,
            difficulty_tier=state.difficulty_tier,
            turn_number=1,
            done=False,
            reward=0.0,
            feedback=feedback,
            task_id=self._current_task.task_id,
        )

    def _turn_2(self, action: AliceAction, state: AliceState) -> AliceObservation:
        state.attempt_history.append(action.response)
        state.turn_number = 2
        reflection_prompt = (
            "Your previous reasoning led to an incorrect answer. "
            "Re-examine each step of your reasoning chain. "
            "Where did the negation 'NOT X' affect your calculation? "
            "Show your corrected reasoning chain, then write: Answer: <number>"
        )
        return AliceObservation(
            task=reflection_prompt,
            skill_domain=state.skill_domain,
            difficulty_tier=state.difficulty_tier,
            turn_number=2,
            done=False,
            reward=0.0,
            feedback="Reflection turn — show corrected reasoning",
            task_id=self._current_task.task_id,
        )

    def _turn_3(self, action: AliceAction, state: AliceState) -> AliceObservation:
        state.attempt_history.append(action.response)
        state.turn_number = 3
        r_prog = self._verifier._tier1_programmatic(self._current_task, action.response)
        r_judge = self._verifier._tier2_llm_judge(self._current_task, action.response)
        feedback = f"T1={r_prog:.2f} T2={r_judge:.2f}"
        return AliceObservation(
            task=self._current_task.task_text,
            skill_domain=state.skill_domain,
            difficulty_tier=state.difficulty_tier,
            turn_number=3,
            hint=self._current_task.hint,
            done=False,
            reward=r_prog * 0.5,
            feedback=feedback,
            task_id=self._current_task.task_id,
        )

    def _turn_4(self, action: AliceAction, state: AliceState) -> AliceObservation:
        state.attempt_history.append(action.response)
        state.turn_number = 4

        # Count repeated responses (excluding the one just appended)
        repeat_count = state.attempt_history[:-1].count(action.response)

        # Full verification
        result = self._verifier.verify(
            self._current_task, action.response, 4, state.attempt_history
        )

        # Novelty similarity
        similarity = self._failure_bank.similarity_score(
            self._current_task.task_text
        )

        # Compute final reward
        r_final = self._reward.compute(
            result.r_programmatic,
            result.r_judge,
            result.r_regression,
            4,
            similarity,
            repeat_count,
        )

        # Update curriculum
        self._curriculum.record_outcome(
            state.skill_domain, result.r_programmatic >= 0.5
        )

        # Store failure if incorrect
        if result.r_programmatic < 0.5:
            self._failure_bank.append(
                FailureRecord(
                    task_id=self._current_task.task_id,
                    task_text=self._current_task.task_text,
                    skill_domain=state.skill_domain,
                    difficulty_tier=state.difficulty_tier,
                    agent_response=action.response,
                    correct_answer=self._current_task.correct_answer,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )

        state.failure_bank_size = self._failure_bank.size()
        state.done = True

        terminal_obs = AliceObservation(
            task=self._current_task.task_text,
            skill_domain=state.skill_domain,
            difficulty_tier=state.difficulty_tier,
            turn_number=4,
            done=True,
            reward=r_final,
            feedback=result.feedback,
            task_id=self._current_task.task_id,
        )
        self._terminal_obs = terminal_obs
        return terminal_obs
