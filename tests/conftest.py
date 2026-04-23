"""Shared fixtures and mocks for ALICE test suite."""

import pytest
import tempfile
from uuid import uuid4
from datetime import datetime

from alice.models import AliceAction, AliceObservation, AliceState
from alice.server.failure_bank import FailureBank, FailureRecord
from alice.server.task_generator import Task, TaskGenerator
from alice.server.episode_handler import EpisodeHandler
from alice.server.curriculum_manager import CurriculumManager
from alice.server.reward import RewardCalculator
from alice.server.verifier import VerifierStack


class MockOracle:
    """Mock Oracle for testing."""
    _client = None
    model_name = "mock"
    
    def score_task(self, task_text: str, correct_answer: str) -> float:
        return 0.5
    
    def is_in_discrimination_zone(self, score: float) -> bool:
        return True


class MockFailureBank:
    """Mock FailureBank for testing."""
    
    def size(self) -> int:
        return 0
    
    def similarity_score(self, task_text: str) -> float:
        return 0.0
    
    def get_recent(self, n: int = 20) -> list:
        return []
    
    def append(self, record: FailureRecord) -> None:
        pass


@pytest.fixture
def make_failure_record():
    """Factory fixture for creating FailureRecord instances."""
    def _make(task_text: str = "test task") -> FailureRecord:
        return FailureRecord(
            task_id=str(uuid4()),
            task_text=task_text,
            skill_domain="negation_arithmetic",
            difficulty_tier="easy",
            agent_response="7",
            correct_answer="7",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    return _make


@pytest.fixture
def make_task():
    """Factory fixture for creating Task instances."""
    def _make(task_text: str = "test", correct_answer: str = "7") -> Task:
        return Task(
            task_id=str(uuid4()),
            task_text=task_text,
            correct_answer=correct_answer,
            skill_domain="negation_arithmetic",
            difficulty_tier="easy",
            mode="hunt",
            hint="test hint"
        )
    return _make


@pytest.fixture
def make_episode_handler():
    """Factory fixture for creating EpisodeHandler instances."""
    def _make() -> EpisodeHandler:
        with tempfile.TemporaryDirectory() as tmp_dir:
            curriculum = CurriculumManager(f"{tmp_dir}/curriculum_state.json")
            reward_calc = RewardCalculator()
            mock_oracle = MockOracle()
            verifier = VerifierStack(mock_oracle)
            mock_failure_bank = MockFailureBank()
            task_gen = TaskGenerator(mock_failure_bank, mock_oracle)
            
            return EpisodeHandler(
                task_gen,
                verifier,
                reward_calc,
                curriculum,
                mock_failure_bank
            )
    return _make


@pytest.fixture
def make_initial_state():
    """Factory fixture for creating AliceState instances."""
    def _make() -> AliceState:
        return AliceState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task=None,
            turn_number=0,
            attempt_history=[],
            skill_domain="negation_arithmetic",
            difficulty_tier="easy",
            mode="hunt",
            failure_bank_size=0,
            task_id="test-id",
            correct_answer="",
            hint=None,
            done=False
        )
    return _make
