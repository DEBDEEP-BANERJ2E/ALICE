"""Unit tests for ALICE components."""

import pytest
import tempfile
from uuid import uuid4

from alice.models import AliceAction, AliceObservation, AliceState
from alice.server.failure_bank import FailureBank, FailureRecord
from alice.server.task_generator import cot_prompt_wrapper, TaskGenerator
from alice.server.verifier import _extract_answer, VerifierStack
from alice.server.curriculum_manager import CurriculumManager
from alice.server.reward import RewardCalculator
from tests.conftest import MockOracle, MockFailureBank


class TestCoTPromptWrapper:
    """Tests for CoT prompt wrapper."""
    
    def test_cot_prompt_wrapper_contains_scaffold(self):
        """Assert 'Answer: <your_number>' is in the wrapped prompt."""
        result = cot_prompt_wrapper("test")
        assert "Answer: <your_number>" in result
        assert "test" in result


class TestExtractAnswer:
    """Tests for answer extraction."""
    
    def test_extract_answer_prefers_answer_marker(self):
        """Assert Answer: marker is preferred over other numbers."""
        response = "Step 1: 3+4=7\nAnswer: 7"
        assert _extract_answer(response) == "7"
    
    def test_extract_answer_fallback_to_last_number(self):
        """Assert fallback to last number when no Answer: marker."""
        response = "the result is 42"
        assert _extract_answer(response) == "42"
    
    def test_extract_answer_returns_none_on_empty(self):
        """Assert None is returned when no numbers found."""
        response = "no numbers here"
        assert _extract_answer(response) is None


class TestCurriculumBoundaries:
    """Tests for curriculum tier boundaries."""
    
    def test_curriculum_boundary_hard(self):
        """Assert tier is 'hard' when accuracy > 0.75."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm = CurriculumManager(f"{tmp_dir}/curriculum_state.json")
            # Record 16 True + 4 False = 0.8 accuracy
            for _ in range(16):
                cm.record_outcome("negation_arithmetic", True)
            for _ in range(4):
                cm.record_outcome("negation_arithmetic", False)
            assert cm.get_tier("negation_arithmetic") == "hard"
    
    def test_curriculum_boundary_easy(self):
        """Assert tier is 'easy' when accuracy < 0.40."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm = CurriculumManager(f"{tmp_dir}/curriculum_state.json")
            # Record 4 True + 16 False = 0.2 accuracy
            for _ in range(4):
                cm.record_outcome("negation_arithmetic", True)
            for _ in range(16):
                cm.record_outcome("negation_arithmetic", False)
            assert cm.get_tier("negation_arithmetic") == "easy"
    
    def test_curriculum_boundary_medium(self):
        """Assert tier is 'medium' when 0.40 <= accuracy <= 0.75."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm = CurriculumManager(f"{tmp_dir}/curriculum_state.json")
            # Record 10 True + 10 False = 0.5 accuracy
            for _ in range(10):
                cm.record_outcome("negation_arithmetic", True)
            for _ in range(10):
                cm.record_outcome("negation_arithmetic", False)
            assert cm.get_tier("negation_arithmetic") == "medium"


class TestEpisodeTurnSequence:
    """Tests for episode turn sequence."""
    
    def test_episode_turn_sequence(self, make_episode_handler, make_initial_state):
        """Assert episode progresses through turns 0-4 with correct done flags."""
        handler = make_episode_handler()
        state = make_initial_state()
        
        # Turn 0: start_episode
        obs = handler.start_episode("negation_arithmetic")
        assert obs.turn_number == 0
        assert obs.done is False
        
        # Turn 1-3: handle_turn with done=False
        for turn in range(1, 4):
            action = AliceAction(response="7", mode="hunt", task_id=obs.task_id)
            state.turn_number = turn - 1
            obs = handler.handle_turn(action, state)
            assert obs.turn_number == turn
            assert obs.done is False
        
        # Turn 4: handle_turn with done=True
        action = AliceAction(response="7", mode="hunt", task_id=obs.task_id)
        state.turn_number = 3
        obs = handler.handle_turn(action, state)
        assert obs.turn_number == 4
        assert obs.done is True


class TestTier1Verification:
    """Tests for Tier 1 programmatic verification."""
    
    def test_tier1_correct_answer(self, make_task):
        """Assert Tier 1 returns 1.0 for correct answer."""
        mock_oracle = MockOracle()
        verifier = VerifierStack(mock_oracle)
        task = make_task(correct_answer="7")
        result = verifier._tier1_programmatic(task, "Answer: 7")
        assert result == 1.0
    
    def test_tier1_wrong_answer(self, make_task):
        """Assert Tier 1 returns 0.0 for wrong answer."""
        mock_oracle = MockOracle()
        verifier = VerifierStack(mock_oracle)
        task = make_task(correct_answer="7")
        result = verifier._tier1_programmatic(task, "Answer: 5")
        assert result == 0.0


class TestRewardFormula:
    """Tests for reward computation."""
    
    def test_reward_formula_known_values(self):
        """Assert reward formula produces expected values."""
        calc = RewardCalculator()
        # compute(1.0, 1.0, 1.0, 0, 0.0, 0)
        # raw = 1.0 * 1.0 * (1.0 - 0.1*0) + 0.3*1.0 - 0.2*0.0 - 0.15*0
        # raw = 1.0 + 0.3 = 1.3
        # clamped = min(1.0, 1.3) = 1.0
        result = calc.compute(1.0, 1.0, 1.0, 0, 0.0, 0)
        assert result == 1.0


class TestFailureBankPersistence:
    """Tests for failure bank persistence."""
    
    def test_failure_bank_append_and_reload(self, make_failure_record):
        """Assert failure bank persists records across instances."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/failure_bank.jsonl"
            
            # Create bank and append 3 records
            bank1 = FailureBank(path)
            for _ in range(3):
                bank1.append(make_failure_record())
            assert bank1.size() == 3
            
            # Create new bank from same path
            bank2 = FailureBank(path)
            assert bank2.size() == 3


class TestScoreResponse:
    """Tests for response scoring."""
    
    def test_score_response_cot_aware(self):
        """Assert score_response correctly extracts CoT Answer marker."""
        from inference import score_response
        response = "Step 1: 3+4=7\nAnswer: 7"
        result = score_response(response, "7")
        assert result == 1.0
