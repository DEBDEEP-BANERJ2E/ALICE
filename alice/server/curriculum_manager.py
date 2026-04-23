"""
Curriculum Manager — tracks per-domain rolling accuracy and assigns difficulty tiers.

Maintains a rolling window of the last 20 episode outcomes per skill domain
and maps accuracy to one of three difficulty tiers: easy, medium, hard.
"""

import json
import logging
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class CurriculumManager:
    WINDOW_SIZE: int = 20
    HARD_THRESHOLD: float = 0.75
    EASY_THRESHOLD: float = 0.40

    def __init__(self, state_path: str = "curriculum_state.json") -> None:
        self._state_path = Path(state_path)
        self._windows: dict[str, deque] = {}
        self._load()

    def record_outcome(self, skill_domain: str, correct: bool) -> None:
        if skill_domain not in self._windows:
            self._windows[skill_domain] = deque(maxlen=self.WINDOW_SIZE)
        self._windows[skill_domain].append(correct)
        self._save()

    def get_tier(self, skill_domain: str) -> str:
        window = self._windows.get(skill_domain)
        if not window:
            return "easy"
        acc = sum(window) / len(window)
        if acc > self.HARD_THRESHOLD:
            return "hard"
        if acc < self.EASY_THRESHOLD:
            return "easy"
        return "medium"

    def get_accuracy(self, skill_domain: str) -> float:
        window = self._windows.get(skill_domain)
        if not window:
            return 0.0
        return sum(window) / len(window)

    def _save(self) -> None:
        try:
            data = {domain: list(window) for domain, window in self._windows.items()}
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError as e:
            logger.error(f"CurriculumManager: failed to save state: {e}")

    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for domain, outcomes in data.items():
                self._windows[domain] = deque(outcomes, maxlen=self.WINDOW_SIZE)
        except Exception:
            pass  # silently ignore missing or corrupt state file
