"""
Task Generator — produces CoT-wrapped arithmetic tasks with distractor framing.

Supports two modes:
  - hunt:   generate a fresh task from a template
  - repair: reconstruct a task from a known failure in the FailureBank
"""

import random
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alice.server.failure_bank import FailureBank
    from alice.server.oracle import Oracle


@dataclass
class Task:
    task_id: str
    task_text: str
    correct_answer: str
    skill_domain: str
    difficulty_tier: str
    mode: str
    hint: str


def cot_prompt_wrapper(task_text: str) -> str:
    return (
        f"{task_text}\n\n"
        "Think step by step. Show your full reasoning chain.\n"
        "Then on the final line write exactly: Answer: <your_number>"
    )


def repair_cot_wrapper(broken_attempt: str, task_text: str) -> str:
    return (
        f'The following solution is WRONG: "{broken_attempt}"\n'
        f"The task was: {task_text}\n\n"
        "Think step by step about where the reasoning went wrong.\n"
        "Show your corrected reasoning chain.\n"
        "Then on the final line write exactly: Answer: <your_number>"
    )


r = random.randint

EASY_TEMPLATES = [
    {
        "template": "If the result is NOT {wrong}, what is the actual result of {a} + {b}?",
        "generator": lambda: {"a": r(1, 9), "b": r(1, 9), "wrong": r(10, 20)},
        "answer": lambda p: p["a"] + p["b"],
    },
    {
        "template": "The answer is NOT {wrong}. What is {a} - {b}?",
        "generator": lambda: {"a": r(5, 15), "b": r(1, 5), "wrong": r(0, 4)},
        "answer": lambda p: p["a"] - p["b"],
    },
    {
        "template": "Ignore the false claim that the result is {wrong}. Compute {a} * {b}.",
        "generator": lambda: {"a": r(2, 9), "b": r(2, 9), "wrong": r(1, 10)},
        "answer": lambda p: p["a"] * p["b"],
    },
]

MEDIUM_TEMPLATES = [
    {
        "template": "If the result is NOT {wrong}, what is {a} + {b} * {c}?",
        "generator": lambda: {"a": r(10, 50), "b": r(2, 9), "c": r(2, 9), "wrong": r(1, 9)},
        "answer": lambda p: p["a"] + p["b"] * p["c"],
    },
    {
        "template": "The answer is NOT {wrong}. Compute ({a} + {b}) * {c}.",
        "generator": lambda: {"a": r(5, 20), "b": r(5, 20), "c": r(2, 9), "wrong": r(50, 100)},
        "answer": lambda p: (p["a"] + p["b"]) * p["c"],
    },
    {
        "template": "Disregard the claim that the result equals {wrong}. What is {a} - {b} + {c}?",
        "generator": lambda: {"a": r(20, 99), "b": r(1, 19), "c": r(1, 19), "wrong": r(0, 10)},
        "answer": lambda p: p["a"] - p["b"] + p["c"],
    },
]

HARD_TEMPLATES = [
    {
        "template": (
            "Someone claims the result is NOT {wrong1} and also NOT {wrong2}. "
            "What is the actual value of {a} + {b} * {c} - {d}?"
        ),
        "generator": lambda: {
            "a": r(100, 500),
            "b": r(2, 9),
            "c": r(2, 9),
            "d": r(1, 50),
            "wrong1": r(1, 50),
            "wrong2": r(51, 100),
        },
        "answer": lambda p: p["a"] + p["b"] * p["c"] - p["d"],
    },
    {
        "template": "The result is NOT {wrong}. Compute ({a} + {b}) * ({c} - {d}).",
        "generator": lambda: {
            "a": r(10, 99),
            "b": r(10, 99),
            "c": r(20, 99),
            "d": r(1, 19),
            "wrong": r(1, 100),
        },
        "answer": lambda p: (p["a"] + p["b"]) * (p["c"] - p["d"]),
    },
    {
        "template": (
            "Ignore the false hint that the answer is {wrong}. "
            "What is {a} * {b} + {c} * {d} - {e}?"
        ),
        "generator": lambda: {
            "a": r(10, 50),
            "b": r(2, 9),
            "c": r(10, 50),
            "d": r(2, 9),
            "e": r(1, 99),
            "wrong": r(1, 50),
        },
        "answer": lambda p: p["a"] * p["b"] + p["c"] * p["d"] - p["e"],
    },
]


class TaskGenerator:
    HUNT_PROBABILITY = 0.7

    def __init__(self, failure_bank: "FailureBank", oracle: "Oracle") -> None:
        self.failure_bank = failure_bank
        self.oracle = oracle

    def _select_mode(self) -> str:
        return "hunt" if random.random() < self.HUNT_PROBABILITY else "repair"

    def _get_templates(self, difficulty_tier: str) -> list:
        if difficulty_tier == "easy":
            return EASY_TEMPLATES
        elif difficulty_tier == "medium":
            return MEDIUM_TEMPLATES
        else:
            return HARD_TEMPLATES

    def _build_hint(self, correct_answer: str, task_text: str) -> str:
        return (
            f"Hint: The negation 'NOT X' is a distractor. "
            f"Ignore it and compute the arithmetic directly. "
            f"The answer is {correct_answer}."
        )

    def _hunt_mode(self, skill_domain: str, difficulty_tier: str) -> Task:
        templates = self._get_templates(difficulty_tier)
        template_entry = random.choice(templates)
        params = template_entry["generator"]()
        correct_answer = str(template_entry["answer"](params))
        task_text = template_entry["template"].format(**params)
        wrapped_text = cot_prompt_wrapper(task_text)
        task_id = str(uuid.uuid4())
        hint = self._build_hint(correct_answer, task_text)
        return Task(
            task_id=task_id,
            task_text=wrapped_text,
            correct_answer=correct_answer,
            skill_domain=skill_domain,
            difficulty_tier=difficulty_tier,
            mode="hunt",
            hint=hint,
        )

    def _repair_mode(self, skill_domain: str, difficulty_tier: str) -> Task:
        if self.failure_bank.size() == 0:
            return self._hunt_mode(skill_domain, difficulty_tier)
        recent = self.failure_bank.get_recent(20)
        record = random.choice(recent)
        wrapped_text = repair_cot_wrapper(record.agent_response, record.task_text)
        task_id = str(uuid.uuid4())
        hint = self._build_hint(record.correct_answer, record.task_text)
        return Task(
            task_id=task_id,
            task_text=wrapped_text,
            correct_answer=record.correct_answer,
            skill_domain=skill_domain,
            difficulty_tier=difficulty_tier,
            mode="repair",
            hint=hint,
        )

    def generate(self, skill_domain: str, difficulty_tier: str) -> Task:
        mode = self._select_mode()
        if mode == "hunt":
            return self._hunt_mode(skill_domain, difficulty_tier)
        return self._repair_mode(skill_domain, difficulty_tier)
