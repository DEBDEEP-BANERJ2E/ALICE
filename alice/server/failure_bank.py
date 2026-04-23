"""
Failure Bank — persistent store of discovered failure modes.

Stores FailureRecord entries as JSON Lines and provides n-gram cosine
similarity scoring for novelty detection.
"""

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    task_id: str
    task_text: str
    skill_domain: str
    difficulty_tier: str
    agent_response: str
    correct_answer: str
    timestamp: str  # ISO 8601


class FailureBank:
    def __init__(self, path: str = "failure_bank.jsonl") -> None:
        self._path = Path(path)
        self._records: list[FailureRecord] = []
        self._fingerprints: list[dict[str, int]] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    record = FailureRecord(**data)
                    self._records.append(record)
                    self._fingerprints.append(self._ngram_fingerprint(record.task_text))
        except Exception as e:
            logger.error(f"FailureBank: failed to load {self._path}: {e}")

    def append(self, record: FailureRecord) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except OSError as e:
            logger.error(f"FailureBank: failed to write to {self._path}: {e}")
        self._records.append(record)
        self._fingerprints.append(self._ngram_fingerprint(record.task_text))

    def _ngram_fingerprint(self, text: str, n: int = 3) -> dict[str, int]:
        text = text.lower()
        return dict(Counter(text[i:i+n] for i in range(len(text) - n + 1)))

    def _cosine_similarity(self, a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def similarity_score(self, task_text: str) -> float:
        if not self._records:
            return 0.0
        fp = self._ngram_fingerprint(task_text)
        return max(self._cosine_similarity(fp, r_fp) for r_fp in self._fingerprints)

    def size(self) -> int:
        return len(self._records)

    def get_recent(self, n: int = 20) -> list[FailureRecord]:
        return self._records[-n:]
