"""
Gradio Dashboard — module-level state store for live training visualisation.

AliceEnvironment calls log_episode() after each Turn 4.
The OpenEnv Gradio UI reads get_dashboard_data() to update live views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alice.server.curriculum_manager import CurriculumManager
    from alice.server.failure_bank import FailureBank

_episode_log: list[dict] = []
_discrimination_series: list[float] = []
_MAX_ENTRIES = 100


def log_episode(
    trajectory: list[dict],
    reward_components: dict,
    discrimination_score: float | None = None,
) -> None:
    """Called by AliceEnvironment after each completed episode (Turn 4)."""
    _episode_log.append({"trajectory": trajectory, "reward_components": reward_components})
    if len(_episode_log) > _MAX_ENTRIES:
        _episode_log.pop(0)
    if discrimination_score is not None:
        _discrimination_series.append(discrimination_score)
        if len(_discrimination_series) > _MAX_ENTRIES:
            _discrimination_series.pop(0)


def get_dashboard_data(
    curriculum_manager: "CurriculumManager",
    failure_bank: "FailureBank",
) -> dict:
    """Return current dashboard state for all live views."""
    domains = ["negation_arithmetic", "pure_arithmetic"]
    return {
        "curriculum": {
            domain: {
                "tier": curriculum_manager.get_tier(domain),
                "accuracy": curriculum_manager.get_accuracy(domain),
            }
            for domain in domains
        },
        "discrimination_series": list(_discrimination_series),
        "failure_bank": [
            {
                "task_text": r.task_text,
                "agent_response": r.agent_response,
                "correct_answer": r.correct_answer,
                "difficulty_tier": r.difficulty_tier,
            }
            for r in failure_bank.get_recent(20)
        ],
        "last_episode_trajectory": (
            _episode_log[-1]["trajectory"] if _episode_log else []
        ),
        "reward_components": [
            e["reward_components"] for e in _episode_log[-20:]
        ],
    }
