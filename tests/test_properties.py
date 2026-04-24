"""Property-based tests for ALICE using Hypothesis."""

import pytest
import tempfile
from datetime import datetime
from uuid import uuid4

from hypothesis import given, settings, strategies as st

from alice.models import AliceAction, AliceObservation, AliceState
from alice.server.failure_bank import FailureBank, FailureRecord
from alice.server.task_generator import cot_prompt_wrapper, TaskGenerator
from alice.server.verifier import _extract_answer, VerifierStack
from alice.server.curriculum_manager import CurriculumManager
from alice.server.reward import RewardCalculator
from tests.conftest import MockOracle, MockFailureBank


# Composite strategy for generating valid FailureRecord instances
@st.composite
def failure_record_strategy(draw):
    """Generate valid FailureRecord instances."""
    return FailureRecord(
        task_id=draw(st.uuids()).hex,
        task_text=draw(st.text(min_size=1, max_size=200)),
        skill_domain="negation_arithmetic",
        difficulty_tier=draw(st.sampled_from(["easy", "medium", "hard"])),
        agent_response=draw(st.text(min_size=1, max_size=100)),
        correct_answer=draw(st.integers(-999, 9999).map(str)),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


class TestRewardBounds:
    """Property 1: Reward is always in [-1.0, 1.0]."""
    
    @given(
        r_prog=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        r_judge=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        r_reg=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        turn=st.integers(0, 4),
        sim=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
        repeat=st.integers(0, 10)
    )
    @settings(max_examples=100)
    def test_reward_bounds(self, r_prog, r_judge, r_reg, turn, sim, repeat):
        """Assert reward is always in [-1.0, 1.0]."""
        # Feature: alice-rl-environment, Property 1: Reward is always in [-1.0, 1.0]
        calc = RewardCalculator()
        result = calc.compute(r_prog, r_judge, r_reg, turn, sim, repeat)
        assert -1.0 <= result <= 1.0


class TestEpisodeTermination:
    """Property 2: Episode always terminates by turn 4."""
    
    @given(responses=st.lists(st.text(min_size=1), min_size=4, max_size=4))
    @settings(max_examples=100)
    def test_episode_terminates_by_turn_4(self, responses, make_episode_handler, make_initial_state):
        """Assert episode terminates by turn 4."""
        # Feature: alice-rl-environment, Property 2: Episode always terminates by turn 4
        handler = make_episode_handler()
        state = make_initial_state()
        
        obs = handler.start_episode("negation_arithmetic")
        for i, response in enumerate(responses):
            action = AliceAction(response=response, mode="hunt", task_id=obs.task_id)
            state.turn_number = i
            obs = handler.handle_turn(action, state)
        
        assert obs.done is True
        assert obs.turn_number == 4


class TestFailureBankSimilarity:
    """Property 3: Failure bank similarity score is always in [0.0, 1.0]."""
    
    @given(
        stored=st.lists(st.text(min_size=1), min_size=0, max_size=20),
        query=st.text(min_size=1)
    )
    @settings(max_examples=100)
    def test_failure_bank_similarity_bounds(self, stored, query):
        """Assert similarity score is always in [0.0, 1.0]."""
        # Feature: alice-rl-environment, Property 3: Failure bank similarity score is always in [0.0, 1.0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            bank = FailureBank(f"{tmp_dir}/failure_bank.jsonl")
            for text in stored:
                record = FailureRecord(
                    task_id=str(uuid4()),
                    task_text=text,
                    skill_domain="negation_arithmetic",
                    difficulty_tier="easy",
                    agent_response="test",
                    correct_answer="7",
                    timestamp=datetime.utcnow().isoformat() + "Z"
                )
                bank.append(record)
            
            score = bank.similarity_score(query)
            assert 0.0 <= score <= 1.0


class TestCurriculumTierValidity:
    """Property 4: Curriculum tier is always one of {easy, medium, hard}."""
    
    @given(outcomes=st.lists(st.booleans(), min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_curriculum_tier_validity(self, outcomes):
        """Assert curriculum tier is always valid."""
        # Feature: alice-rl-environment, Property 4: Curriculum tier is always one of {easy, medium, hard}
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm = CurriculumManager(f"{tmp_dir}/curriculum_state.json")
            for outcome in outcomes:
                cm.record_outcome("negation_arithmetic", outcome)
            
            tier = cm.get_tier("negation_arithmetic")
            assert tier in {"easy", "medium", "hard"}


class TestInferenceScoreBounds:
    """Property 5: Inference script score is always in [0.0, 1.0]."""
    
    @given(responses=st.lists(st.text(), min_size=3, max_size=3))
    @settings(max_examples=100)
    def test_inference_score_bounds(self, responses):
        """Assert inference score is always in [0.0, 1.0]."""
        # Feature: alice-rl-environment, Property 5: Inference script score is always in [0.0, 1.0]
        from inference import score_response
        
        correct_answers = ["7", "42", "122"]
        scores = [score_response(r, c) for r, c in zip(responses, correct_answers)]
        mean_score = sum(scores) / len(scores)
        assert 0.0 <= mean_score <= 1.0


class TestDataModelRoundTrip:
    """Property 6: Data model JSON round-trip."""
    
    @given(
        response=st.text(min_size=1),
        mode=st.sampled_from(["hunt", "repair"]),
        task_id=st.uuids().map(str)
    )
    @settings(max_examples=100)
    def test_data_model_json_round_trip(self, response, mode, task_id):
        """Assert data model JSON round-trip works."""
        # Feature: alice-rl-environment, Property 6: Data model JSON round-trip
        action = AliceAction(response=response, mode=mode, task_id=task_id)
        json_str = action.model_dump_json()
        restored = AliceAction.model_validate_json(json_str)
        assert restored == action


class TestFailureBankRoundTrip:
    """Property 7: Failure bank append round-trip."""
    
    @given(records=st.lists(failure_record_strategy(), min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_failure_bank_round_trip(self, records):
        """Assert failure bank append round-trip works."""
        # Feature: alice-rl-environment, Property 7: Failure bank append round-trip
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f"{tmp_dir}/failure_bank.jsonl"
            
            # Append all records
            bank1 = FailureBank(path)
            for record in records:
                bank1.append(record)
            
            # Create new bank from same path
            bank2 = FailureBank(path)
            assert bank2.size() == len(records)


class TestVerifierNoExceptions:
    """Property 8: Verifier stack never propagates exceptions."""
    
    @given(
        response=st.text(),
        task_text=st.text(),
        correct_answer=st.text()
    )
    @settings(max_examples=100)
    def test_verifier_no_exceptions(self, response, task_text, correct_answer):
        """Assert verifier never propagates exceptions."""
        # Feature: alice-rl-environment, Property 8: Verifier stack never propagates exceptions
        from alice.server.task_generator import Task
        
        mock_oracle = MockOracle()
        verifier = VerifierStack(mock_oracle)
        task = Task(
            task_id=str(uuid4()),
            task_text=task_text,
            correct_answer=correct_answer,
            skill_domain="negation_arithmetic",
            difficulty_tier="easy",
            mode="hunt",
            hint="test"
        )
        
        # Should not raise
        result = verifier.verify(task, response, 1, [])
        
        # All tier scores should be in [0.0, 1.0]
        assert 0.0 <= result.r_programmatic <= 1.0
        assert 0.0 <= result.r_judge <= 1.0
        assert 0.0 <= result.r_regression <= 1.0
        assert -1.0 <= result.r_final <= 1.0


class TestTaskIDUniqueness:
    """Property 9: Task IDs are unique."""
    
    @given(n=st.integers(min_value=2, max_value=20))
    @settings(max_examples=100)
    def test_task_ids_unique(self, n):
        """Assert task IDs are unique."""
        # Feature: alice-rl-environment, Property 9: Task IDs are unique
        mock_failure_bank = MockFailureBank()
        mock_oracle = MockOracle()
        task_gen = TaskGenerator(mock_failure_bank, mock_oracle)
        
        tasks = [task_gen.generate("negation_arithmetic", "easy") for _ in range(n)]
        task_ids = [task.task_id for task in tasks]
        
        assert len(set(task_ids)) == n


class TestCurriculumWindowBounded:
    """Property 10: Curriculum window is bounded at 20."""
    
    @given(outcomes=st.lists(st.booleans(), min_size=21, max_size=100))
    @settings(max_examples=100)
    def test_curriculum_window_bounded(self, outcomes):
        """Assert curriculum window is bounded at 20."""
        # Feature: alice-rl-environment, Property 10: Curriculum window is bounded at 20
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm = CurriculumManager(f"{tmp_dir}/curriculum_state.json")
            for outcome in outcomes:
                cm.record_outcome("negation_arithmetic", outcome)
            
            # Compute expected tier from last 20 outcomes
            last_20 = outcomes[-20:]
            acc = sum(last_20) / len(last_20)
            
            if acc > 0.75:
                expected_tier = "hard"
            elif acc < 0.40:
                expected_tier = "easy"
            else:
                expected_tier = "medium"
            
            assert cm.get_tier("negation_arithmetic") == expected_tier


class TestCoTAnswerExtraction:
    """Property 11: CoT Answer marker extraction is consistent."""
    
    @given(number=st.integers(-999, 9999))
    @settings(max_examples=100)
    def test_cot_answer_extraction_consistent(self, number):
        """Assert CoT Answer marker extraction is consistent."""
        # Feature: alice-rl-environment, Property 11: CoT Answer marker extraction is consistent
        response = f"Step 1: blah\nAnswer: {number}"
        extracted = _extract_answer(response)
        assert extracted == str(number)


class TestCoTWrapperIdempotence:
    """Property 12: CoT wrapper always contains original task and scaffold."""
    
    @given(task_text=st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_cot_wrapper_idempotence(self, task_text):
        """Assert CoT wrapper contains original task and scaffold."""
        # Feature: alice-rl-environment, Property 12: CoT wrapper always contains original task and scaffold
        wrapped = cot_prompt_wrapper(task_text)
        assert task_text in wrapped
        assert "Answer: <your_number>" in wrapped
