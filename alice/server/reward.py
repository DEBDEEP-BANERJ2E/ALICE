"""
Reward Calculator — computes R_final from the three-tier verifier scores.

Formula:
    R_final = R_programmatic * R_regression * (1 - attempt_decay * turn_number)
            + lambda_judge * R_judge
            - novelty_penalty * similarity
            - repetition_penalty * repeat_count

Clamped to [-1.0, 1.0].
"""


class RewardCalculator:
    ATTEMPT_DECAY: float = 0.1
    LAMBDA_JUDGE: float = 0.3
    NOVELTY_PENALTY: float = 0.2
    REPETITION_PENALTY: float = 0.15

    def compute(
        self,
        r_programmatic: float,
        r_judge: float,
        r_regression: float,
        turn_number: int,
        similarity: float,
        repeat_count: int,
    ) -> float:
        raw = (
            r_programmatic * r_regression * (1.0 - self.ATTEMPT_DECAY * turn_number)
            + self.LAMBDA_JUDGE * r_judge
            - self.NOVELTY_PENALTY * similarity
            - self.REPETITION_PENALTY * repeat_count
        )
        return max(-1.0, min(1.0, raw))

    def decompose(
        self,
        r_programmatic: float,
        r_judge: float,
        r_regression: float,
        turn_number: int,
        similarity: float,
        repeat_count: int,
    ) -> dict:
        """Return each reward component separately for training monitoring."""
        r_prog_x_reg = r_programmatic * r_regression * (1.0 - self.ATTEMPT_DECAY * turn_number)
        judge_contrib = self.LAMBDA_JUDGE * r_judge
        novelty_pen = self.NOVELTY_PENALTY * similarity
        repeat_pen = self.REPETITION_PENALTY * repeat_count
        r_final = self.compute(r_programmatic, r_judge, r_regression, turn_number, similarity, repeat_count)
        return {
            "r_prog_x_reg": round(r_prog_x_reg, 4),
            "judge_contrib": round(judge_contrib, 4),
            "novelty_penalty": round(novelty_pen, 4),
            "repeat_penalty": round(repeat_pen, 4),
            "r_final": round(r_final, 4),
        }
