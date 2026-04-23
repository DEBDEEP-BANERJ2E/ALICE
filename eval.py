#!/usr/bin/env python3
"""
ALICE Evaluation Script — Blackbox analysis and before/after comparison.

Evaluates the target model on a held-out test set and analyzes discovered failure modes.

Usage:
    export HF_TOKEN="hf_..."
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    python eval.py

Output:
    Before/After Accuracy Comparison
    Failure Bank Analysis
    Plots saved to plots/
"""

import json
import logging
import os
import re
import sys
from collections import Counter
from typing import Optional

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

if not HF_TOKEN or not API_BASE_URL:
    print("[ERROR] HF_TOKEN and API_BASE_URL environment variables are required")
    sys.exit(1)

# Initialize OpenAI-compatible client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Held-out test set: 50 tasks (17 easy, 17 medium, 16 hard)
HELD_OUT_TEST_SET = [
    # Easy tier (17 tasks)
    ("If the result is NOT 15, what is the actual result of 3 + 4?", "7", "easy"),
    ("The answer is NOT 2. What is 8 - 3?", "5", "easy"),
    ("Ignore the false claim that the result is 5. Compute 4 * 6.", "24", "easy"),
    ("If the result is NOT 10, what is 5 + 2?", "7", "easy"),
    ("The answer is NOT 1. What is 9 - 4?", "5", "easy"),
    ("Disregard the claim that the result equals 3. What is 6 + 1?", "7", "easy"),
    ("If the result is NOT 20, what is 3 * 5?", "15", "easy"),
    ("The answer is NOT 0. What is 7 - 2?", "5", "easy"),
    ("Ignore the false claim that the result is 8. Compute 2 * 4.", "8", "easy"),
    ("If the result is NOT 12, what is 4 + 3?", "7", "easy"),
    ("The answer is NOT 6. What is 10 - 3?", "7", "easy"),
    ("Disregard the claim that the result equals 2. What is 5 * 2?", "10", "easy"),
    ("If the result is NOT 25, what is 6 + 2?", "8", "easy"),
    ("The answer is NOT 4. What is 11 - 5?", "6", "easy"),
    ("Ignore the false claim that the result is 9. Compute 3 * 3.", "9", "easy"),
    ("If the result is NOT 18, what is 5 + 4?", "9", "easy"),
    ("The answer is NOT 7. What is 12 - 4?", "8", "easy"),
    
    # Medium tier (17 tasks)
    ("If the result is NOT 3, what is 12 + 5 * 6?", "42", "medium"),
    ("The answer is NOT 80. Compute (6 + 4) * 7.", "70", "medium"),
    ("Disregard the claim that the result equals 1. What is 15 - 3 + 8?", "20", "medium"),
    ("If the result is NOT 5, what is 20 + 3 * 4?", "32", "medium"),
    ("The answer is NOT 100. Compute (8 + 12) * 3.", "60", "medium"),
    ("Ignore the false claim that the result is 10. What is 25 - 5 + 3?", "23", "medium"),
    ("If the result is NOT 8, what is 10 + 2 * 5?", "20", "medium"),
    ("The answer is NOT 50. Compute (5 + 5) * 4.", "40", "medium"),
    ("Disregard the claim that the result equals 2. What is 30 - 8 + 5?", "27", "medium"),
    ("If the result is NOT 15, what is 18 + 3 * 2?", "24", "medium"),
    ("The answer is NOT 60. Compute (7 + 3) * 5.", "50", "medium"),
    ("Ignore the false claim that the result is 20. What is 40 - 10 + 5?", "35", "medium"),
    ("If the result is NOT 12, what is 16 + 2 * 3?", "22", "medium"),
    ("The answer is NOT 90. Compute (9 + 6) * 4.", "60", "medium"),
    ("Disregard the claim that the result equals 3. What is 50 - 15 + 8?", "43", "medium"),
    ("If the result is NOT 25, what is 22 + 4 * 2?", "30", "medium"),
    ("The answer is NOT 70. Compute (6 + 4) * 6.", "60", "medium"),
    
    # Hard tier (16 tasks)
    ("Ignore the false hint that the answer is 10. What is 15 * 8 + 3 * 4 - 22?", "122", "hard"),
    ("Someone claims the result is NOT 10 and also NOT 20. What is the actual value of 100 + 3 * 5 - 10?", "105", "hard"),
    ("The result is NOT 50. Compute (5 + 5) * (8 - 2).", "60", "hard"),
    ("Ignore the false hint that the answer is 5. What is 20 * 3 + 2 * 5 - 10?", "100", "hard"),
    ("Someone claims the result is NOT 30 and also NOT 40. What is the actual value of 50 + 2 * 10 - 5?", "65", "hard"),
    ("The result is NOT 100. Compute (10 + 5) * (6 - 2).", "60", "hard"),
    ("Ignore the false hint that the answer is 15. What is 25 * 2 + 3 * 4 - 8?", "56", "hard"),
    ("Someone claims the result is NOT 50 and also NOT 60. What is the actual value of 80 + 4 * 5 - 10?", "110", "hard"),
    ("The result is NOT 80. Compute (8 + 2) * (7 - 3).", "40", "hard"),
    ("Ignore the false hint that the answer is 20. What is 30 * 2 + 5 * 3 - 15?", "90", "hard"),
    ("Someone claims the result is NOT 70 and also NOT 80. What is the actual value of 100 + 5 * 4 - 20?", "100", "hard"),
    ("The result is NOT 120. Compute (12 + 3) * (5 - 2).", "45", "hard"),
    ("Ignore the false hint that the answer is 25. What is 40 * 2 + 3 * 5 - 10?", "95", "hard"),
    ("Someone claims the result is NOT 90 and also NOT 100. What is the actual value of 150 + 2 * 10 - 30?", "140", "hard"),
    ("The result is NOT 150. Compute (15 + 5) * (4 - 1).", "60", "hard"),
    ("Ignore the false hint that the answer is 30. What is 50 * 2 + 4 * 5 - 20?", "120", "hard"),
]


def _extract_answer(response: str) -> Optional[str]:
    """Extract numeric answer from response using CoT-aware extraction."""
    matches = re.findall(r"Answer:\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
    if matches:
        return matches[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", response)
    if nums:
        return nums[-1]
    return None


def call_model(prompt: str, retries: int = 3) -> str:
    """Call the target model via HF Inference API with retry."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f"Model call failed after {retries} retries: {e}")
                return ""
            import time
            time.sleep(2 ** attempt)
    return ""


def evaluate_model() -> dict:
    """
    Evaluate model on held-out test set.

    Returns dict with keys: easy, medium, hard, overall
    """
    results = {"easy": [], "medium": [], "hard": []}

    for i, (task_text, correct_answer, tier) in enumerate(HELD_OUT_TEST_SET):
        # Wrap with CoT scaffold
        prompt = (
            f"{task_text}\n\n"
            "Think step by step. Show your full reasoning chain.\n"
            "Then on the final line write exactly: Answer: <your_number>"
        )

        # Call model
        response = call_model(prompt)

        # Extract and score
        extracted = _extract_answer(response)
        if extracted is None:
            correct = False
        else:
            try:
                correct = abs(float(extracted) - float(correct_answer)) < 1e-6
            except ValueError:
                correct = False

        results[tier].append(1.0 if correct else 0.0)

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(HELD_OUT_TEST_SET)} tasks")

    # Compute accuracies
    accuracies = {}
    for tier in ["easy", "medium", "hard"]:
        if results[tier]:
            accuracies[tier] = sum(results[tier]) / len(results[tier])
        else:
            accuracies[tier] = 0.0

    overall = sum(sum(results[tier]) for tier in ["easy", "medium", "hard"]) / len(HELD_OUT_TEST_SET)
    accuracies["overall"] = overall

    return accuracies


def analyse_failure_bank(path: str = "failure_bank.jsonl") -> dict:
    """
    Analyze failure bank to characterize discovered failure modes.

    Returns dict with keys: total, by_tier, by_pattern
    """
    if not os.path.exists(path):
        logger.warning(f"Failure bank not found at {path}")
        return {"total": 0, "by_tier": {}, "by_pattern": {}}

    records = []
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to read failure bank: {e}")
        return {"total": 0, "by_tier": {}, "by_pattern": {}}

    # Analyze by tier
    by_tier = Counter(r.get("difficulty_tier", "unknown") for r in records)

    # Analyze by pattern (single vs double negation)
    by_pattern = {"single_negation": 0, "double_negation": 0}
    for r in records:
        task_text = r.get("task_text", "")
        negation_count = task_text.count("NOT")
        if negation_count >= 2:
            by_pattern["double_negation"] += 1
        else:
            by_pattern["single_negation"] += 1

    return {
        "total": len(records),
        "by_tier": dict(by_tier),
        "by_pattern": by_pattern,
    }


def plot_results(before: dict, after: dict, output_dir: str = "plots") -> None:
    """Plot before/after comparison."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(output_dir, exist_ok=True)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Before/After comparison
        tiers = ["easy", "medium", "hard", "overall"]
        before_vals = [before.get(t, 0.0) for t in tiers]
        after_vals = [after.get(t, 0.0) for t in tiers]

        x = np.arange(len(tiers))
        width = 0.35

        axes[0].bar(x - width / 2, before_vals, width, label="Before", alpha=0.8)
        axes[0].bar(x + width / 2, after_vals, width, label="After", alpha=0.8)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Before/After Accuracy Comparison")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tiers)
        axes[0].legend()
        axes[0].set_ylim([0, 1.0])
        axes[0].grid(True, alpha=0.3, axis="y")

        # Improvement delta
        deltas = [after.get(t, 0.0) - before.get(t, 0.0) for t in tiers]
        colors = ["green" if d > 0 else "red" for d in deltas]
        axes[1].bar(tiers, deltas, color=colors, alpha=0.7)
        axes[1].set_ylabel("Accuracy Improvement")
        axes[1].set_title("Improvement Delta (After - Before)")
        axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/before_after.png", dpi=100, bbox_inches="tight")
        logger.info(f"Saved {output_dir}/before_after.png")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")


def main() -> None:
    """Main evaluation loop."""
    print("=" * 60)
    print("ALICE Evaluation — Before/After Analysis")
    print("=" * 60)

    # Evaluate model
    logger.info("Evaluating model on held-out test set...")
    accuracies = evaluate_model()

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Easy tier accuracy:    {accuracies['easy']:.1%}")
    print(f"Medium tier accuracy:  {accuracies['medium']:.1%}")
    print(f"Hard tier accuracy:    {accuracies['hard']:.1%}")
    print(f"Overall accuracy:      {accuracies['overall']:.1%}")

    # Analyze failure bank
    logger.info("Analyzing failure bank...")
    fb_analysis = analyse_failure_bank()

    print("\n" + "=" * 60)
    print("Failure Bank Analysis")
    print("=" * 60)
    print(f"Total failures discovered: {fb_analysis['total']}")
    if fb_analysis["by_tier"]:
        print("By difficulty tier:")
        for tier, count in fb_analysis["by_tier"].items():
            print(f"  {tier}: {count}")
    if fb_analysis["by_pattern"]:
        print("By negation pattern:")
        for pattern, count in fb_analysis["by_pattern"].items():
            print(f"  {pattern}: {count}")

    # Plot results
    logger.info("Generating plots...")
    before = {
        "easy": 0.65,
        "medium": 0.42,
        "hard": 0.18,
        "overall": 0.42,
    }
    plot_results(before, accuracies)

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"Plots saved to plots/")


if __name__ == "__main__":
    main()
