"""
Oracle Interface — computes discrimination scores via the HF Inference API.

Discrimination_Score = reference_pass_rate - target_pass_rate

Tasks in the discrimination zone [0.2, 0.8] are harder for the target model
than for reference models — these are the most informative training tasks.
"""

import logging
import re

import openai

logger = logging.getLogger(__name__)

REFERENCE_MODEL_POOL = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
]


class Oracle:
    DISCRIMINATION_LOW: float = 0.2
    DISCRIMINATION_HIGH: float = 0.8

    def __init__(self, api_base_url: str, model_name: str, hf_token: str) -> None:
        self._client = openai.OpenAI(base_url=api_base_url, api_key=hf_token)
        self.model_name = model_name

    def _query_model(self, model_name: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def _compute_pass_rate(
        self,
        model_name: str,
        task_text: str,
        correct_answer: str,
        n: int = 5,
    ) -> float:
        correct = 0
        for _ in range(n):
            try:
                resp = self._query_model(model_name, task_text)
                nums = re.findall(r"-?\d+(?:\.\d+)?", resp)
                if nums and abs(float(nums[-1]) - float(correct_answer)) < 1e-6:
                    correct += 1
            except Exception:
                pass
        return correct / n

    def score_task(self, task_text: str, correct_answer: str) -> float:
        """Returns discrimination_score = reference_pass_rate - target_pass_rate."""
        try:
            ref_rates = [
                self._compute_pass_rate(m, task_text, correct_answer)
                for m in REFERENCE_MODEL_POOL
            ]
            reference_pass_rate = sum(ref_rates) / len(ref_rates)
            target_pass_rate = self._compute_pass_rate(
                self.model_name, task_text, correct_answer
            )
            return reference_pass_rate - target_pass_rate
        except Exception as e:
            logger.warning(f"Oracle.score_task failed: {e}")
            return 0.5  # neutral fallback

    def is_in_discrimination_zone(self, score: float) -> bool:
        return self.DISCRIMINATION_LOW <= score <= self.DISCRIMINATION_HIGH
