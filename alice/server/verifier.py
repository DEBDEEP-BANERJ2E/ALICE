"""
Verifier Stack — three-tier answer verification for CoT-aware RL tasks.

Tier 1 (Programmatic): exact numeric match via regex extraction
Tier 2 (LLM Judge):    LLM-scored correctness in [0.0, 1.0]
Tier 3 (Regression):   stability check against a fixed battery of 20 tasks
"""

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alice.server.oracle import Oracle

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    r_programmatic: float
    r_judge: float
    r_regression: float
    r_final: float
    feedback: str


def _extract_answer(response: str) -> str | None:
    """Extract the last numeric answer from a response string."""
    matches = re.findall(r"Answer:\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if matches:
        return matches[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", response)
    if nums:
        return nums[-1]
    return None


JUDGE_PROMPT_TEMPLATE = (
    "You are a strict math grader. Given the task and the student's response, "
    "output a single float in [0.0, 1.0] representing correctness.\n"
    "0.0 = completely wrong, 1.0 = completely correct, "
    "0.5 = partially correct reasoning but wrong final answer.\n"
    "Also reward clear step-by-step reasoning even if the final answer is slightly off.\n"
    "Output ONLY the float, nothing else.\n\n"
    "Task: {task_text}\n"
    "Correct answer: {correct_answer}\n"
    "Student response: {response}\n\n"
    "Score:"
)

REGRESSION_BATTERY = [
    ("If the result is NOT 15, what is the actual result of 3 + 4?", "7"),
    ("The answer is NOT 2. What is 8 - 3?", "5"),
    ("Ignore the false claim that the result is 5. Compute 4 * 6.", "24"),
    ("If the result is NOT 3, what is 10 + 2 * 5?", "20"),
    ("The answer is NOT 80. Compute (6 + 4) * 7.", "70"),
    ("Disregard the claim that the result equals 1. What is 15 - 3 + 8?", "20"),
    ("If the result is NOT 12, what is the actual result of 5 + 7?", "12"),
    ("The answer is NOT 0. What is 9 - 4?", "5"),
    ("Ignore the false claim that the result is 3. Compute 7 * 8.", "56"),
    ("If the result is NOT 5, what is 20 + 3 * 4?", "32"),
    ("The answer is NOT 100. Compute (8 + 12) * 3.", "60"),
    ("Disregard the claim that the result equals 0. What is 30 - 7 + 2?", "25"),
    ("If the result is NOT 20, what is the actual result of 6 + 9?", "15"),
    ("The answer is NOT 1. What is 12 - 5?", "7"),
    ("Ignore the false claim that the result is 10. Compute 3 * 9.", "27"),
    ("If the result is NOT 8, what is 15 + 4 * 6?", "39"),
    ("The answer is NOT 50. Compute (7 + 3) * 8.", "80"),
    ("Disregard the claim that the result equals 5. What is 25 - 8 + 3?", "20"),
    ("Someone claims the result is NOT 10 and also NOT 20. What is the actual value of 100 + 3 * 5 - 10?", "105"),
    ("The result is NOT 50. Compute (5 + 5) * (8 - 2).", "60"),
]


class VerifierStack:
    def __init__(self, oracle: "Oracle | None") -> None:
        self._oracle = oracle
        self._client = oracle._client if oracle is not None else None

    def _tier1_programmatic(self, task, response: str) -> float:
        try:
            extracted = _extract_answer(response)
            if extracted is None:
                return 0.0
            if abs(float(extracted) - float(task.correct_answer)) < 1e-6:
                return 1.0
            return 0.0
        except Exception as e:
            logger.warning(f"Tier1 programmatic check failed: {e}")
            return 0.0

    def _tier2_llm_judge(self, task, response: str) -> float:
        if self._client is None:
            return 0.0
        try:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                task_text=task.task_text,
                correct_answer=task.correct_answer,
                response=response,
            )
            completion = self._client.chat.completions.create(
                model=self._oracle.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            reply = completion.choices[0].message.content.strip()
            nums = re.findall(r"\d+\.\d+|\d+", reply)
            if not nums:
                return 0.0
            score = float(nums[0])
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Tier2 LLM judge failed: {e}")
            return 0.0

    def _tier3_regression(self, response: str) -> float:
        try:
            candidate = _extract_answer(response)
            if candidate is None:
                return 0.0
            pass_count = 0
            for _task_text, correct_answer in REGRESSION_BATTERY:
                try:
                    if abs(float(candidate) - float(correct_answer)) < 1e-6:
                        pass_count += 1
                except Exception:
                    pass
            return pass_count / 20
        except Exception as e:
            logger.warning(f"Tier3 regression check failed: {e}")
            return 0.0

    def verify(self, task, response: str, turn_number: int, attempt_history: list) -> VerificationResult:
        r_prog = self._tier1_programmatic(task, response)
        r_judge = self._tier2_llm_judge(task, response)
        r_reg = self._tier3_regression(response)
        extracted = _extract_answer(response)
        feedback = f"T1={r_prog:.2f} T2={r_judge:.2f} T3={r_reg:.2f} | extracted='{extracted}'"
        return VerificationResult(
            r_programmatic=r_prog,
            r_judge=r_judge,
            r_regression=r_reg,
            r_final=0.0,
            feedback=feedback,
        )
