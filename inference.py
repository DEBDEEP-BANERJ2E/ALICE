#!/usr/bin/env python3
"""
ALICE Inference Script — Standalone evaluation of the ALICE environment.

Evaluates the target model (Qwen/Qwen2.5-7B-Instruct) on 3 negation_arithmetic tasks
(easy, medium, hard) using the HF Inference API. Completes in <20 minutes on 2vCPU/8GB.

Usage:
    export HF_TOKEN="hf_..."
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    python inference.py

Output:
    [START] ALICE inference — model=Qwen/Qwen2.5-7B-Instruct tasks=3
    [STEP] task=0 tier=easy response="..." r_final=0.75
    [STEP] task=1 tier=medium response="..." r_final=0.50
    [STEP] task=2 tier=hard response="..." r_final=0.25
    [END] mean_r_final=0.5000 duration=45.2s
"""

import os
import re
import sys
import time
from typing import Optional

from openai import OpenAI

# Environment variables
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

if not HF_TOKEN or not API_BASE_URL:
    print("[ERROR] HF_TOKEN and API_BASE_URL environment variables are required")
    sys.exit(1)

# Initialize OpenAI-compatible client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Task definitions: (tier, task_text_with_cot, correct_answer)
TASKS = [
    (
        "easy",
        "If the result is NOT 5, what is the actual result of 3 + 4?\n\n"
        "Think step by step. Show your full reasoning chain.\n"
        "Then on the final line write exactly: Answer: <your_number>",
        "7",
    ),
    (
        "medium",
        "If the result is NOT 3, what is 12 + 5 * 6?\n\n"
        "Think step by step. Show your full reasoning chain.\n"
        "Then on the final line write exactly: Answer: <your_number>",
        "42",
    ),
    (
        "hard",
        "Ignore the false hint that the answer is 10. What is 15 * 8 + 3 * 4 - 22?\n\n"
        "Think step by step. Show your full reasoning chain.\n"
        "Then on the final line write exactly: Answer: <your_number>",
        "122",
    ),
]


def call_model(prompt: str, retries: int = 3) -> str:
    """
    Call the target model via HF Inference API with exponential backoff retry.

    Args:
        prompt: The task prompt to send to the model
        retries: Number of retry attempts

    Returns:
        The model's response text, or empty string on all retries exhausted
    """
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
                print(f"[WARNING] Model call failed after {retries} retries: {e}", file=sys.stderr)
                return ""
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    return ""


def score_response(response: str, correct_answer: str) -> float:
    """
    Score the model's response against the correct answer.

    Extracts the numeric answer from the response using CoT-aware extraction:
    1. First tries to find "Answer: <number>" pattern
    2. Falls back to the last number in the response
    3. Compares to correct_answer within 1e-6 tolerance

    Args:
        response: The model's response text
        correct_answer: The ground truth answer

    Returns:
        1.0 if correct, 0.0 if incorrect or extraction failed
    """
    try:
        # Try CoT-aware extraction first: "Answer: <number>"
        matches = re.findall(r"Answer:\s*(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
        if matches:
            extracted = matches[-1]
        else:
            # Fall back to last number in response
            nums = re.findall(r"-?\d+(?:\.\d+)?", response)
            if not nums:
                return 0.0
            extracted = nums[-1]

        # Compare numerically
        candidate = float(extracted)
        expected = float(correct_answer)
        if abs(candidate - expected) < 1e-6:
            return 1.0
        return 0.0
    except Exception:
        return 0.0


def main() -> None:
    """Run inference on all 3 tasks and report mean score."""
    print(f"[START] ALICE inference — model={MODEL_NAME} tasks=3")
    t0 = time.time()

    scores = []
    for i, (tier, task_text, correct_answer) in enumerate(TASKS):
        # Call model
        response = call_model(task_text)

        # Score response
        r_final = score_response(response, correct_answer)
        scores.append(r_final)

        # Truncate response for logging
        response_preview = response[:60].replace("\n", " ")
        print(f"[STEP] task={i} tier={tier} response=\"{response_preview}\" r_final={r_final:.2f}")

    # Compute mean and duration
    mean_score = sum(scores) / len(scores) if scores else 0.0
    duration = time.time() - t0
    print(f"[END] mean_r_final={mean_score:.4f} duration={duration:.1f}s")


if __name__ == "__main__":
    main()
