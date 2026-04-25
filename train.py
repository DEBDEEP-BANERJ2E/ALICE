#!/usr/bin/env python3
"""
ALICE Training Script — GRPO fine-tuning for negation-arithmetic repair.

Connects Qwen2.5-7B-Instruct to the ALICE environment via TRL's GRPOTrainer
(+ optional Unsloth acceleration) to fix negation-arithmetic failure modes.

Usage (local, requires GPU + running ALICE server):
    export HF_TOKEN="hf_..."
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
    # In another terminal: cd alice && uvicorn server.app:app --port 8000
    python train.py --server-url http://localhost:8000 --num-episodes 300

For Colab (recommended), open train.ipynb — it handles server start + plotting.
"""

import argparse
import logging
import os
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


# ── Reward function ────────────────────────────────────────────────────────────

def make_alice_reward_fn(server_url: str):
    """
    Return a reward function for TRL's GRPOTrainer.

    Signature: fn(prompts, completions, **kwargs) -> list[float]
    Each call resets the environment, steps with the completion, and returns R_final.
    """
    def reward_fn(prompts: list, completions: list, **kwargs) -> list:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            try:
                reset = requests.post(
                    f"{server_url}/reset", json={}, timeout=10
                ).json()
                task_id = reset.get("observation", {}).get("task_id", "")
                step = requests.post(
                    f"{server_url}/step",
                    json={"response": completion, "mode": "hunt", "task_id": task_id},
                    timeout=15,
                ).json()
                rewards.append(float(step.get("reward", 0.0)))
            except Exception as exc:
                logger.warning("Reward error: %s", exc)
                rewards.append(0.0)
        return rewards

    return reward_fn


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_dataset(server_url: str, n: int):
    """Pull n task prompts from ALICE /reset and return a HF Dataset."""
    from datasets import Dataset

    rows = []
    for i in range(n):
        try:
            resp = requests.post(f"{server_url}/reset", json={}, timeout=10).json()
            obs  = resp.get("observation", {})
            rows.append({"prompt": obs.get("task", ""), "task_id": obs.get("task_id", "")})
        except Exception as exc:
            logger.warning("Dataset build failed at %d: %s", i, exc)
        if (i + 1) % 50 == 0:
            logger.info("Built %d/%d prompts", i + 1, n)

    logger.info("Dataset ready: %d examples", len(rows))
    return Dataset.from_list(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_plots(log_history: list, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt

        os.makedirs(f"{output_dir}/plots", exist_ok=True)

        steps   = [e["step"] for e in log_history if "loss" in e]
        losses  = [e["loss"] for e in log_history if "loss" in e]
        rewards = [e.get("reward", 0.0) for e in log_history if "loss" in e]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(steps, losses, color="#C44E52", linewidth=1.5)
        axes[0].set_xlabel("Training Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("GRPO Training Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(steps, rewards, color="#4C72B0", linewidth=1.5, label="Mean R_final")
        axes[1].axhline(0, color="grey", linewidth=0.7, linestyle="--")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Mean Reward (R_final)")
        axes[1].set_title("ALICE GRPO Reward Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = f"{output_dir}/plots/reward_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", out)
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ALICE GRPO Training")
    parser.add_argument("--server-url",     default="http://localhost:8000")
    parser.add_argument("--model-name",     default=MODEL_NAME)
    parser.add_argument("--num-episodes",   type=int, default=300)
    parser.add_argument("--output-dir",     default="./output/alice-grpo")
    parser.add_argument("--lora-rank",      type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--no-unsloth",     action="store_true",
                        help="Fall back to plain transformers + PEFT")
    args = parser.parse_args()

    logger.info("ALICE GRPO training — model=%s episodes=%d", args.model_name, args.num_episodes)

    # ── Wait for ALICE server ─────────────────────────────────────
    for attempt in range(30):
        try:
            if requests.get(f"{args.server_url}/health", timeout=3).status_code == 200:
                logger.info("ALICE server ready at %s", args.server_url)
                break
        except Exception:
            pass
        if attempt == 29:
            logger.error("ALICE server not reachable. Start it first and retry.")
            sys.exit(1)
        time.sleep(1)

    # ── Load model ────────────────────────────────────────────────
    import torch

    if not args.no_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                load_in_4bit=True,
                token=HF_TOKEN or None,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=args.lora_rank,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            logger.info("Loaded with Unsloth 4-bit QLoRA")
        except ImportError:
            logger.warning("Unsloth not found, using transformers + PEFT")
            args.no_unsloth = True

    if args.no_unsloth:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN or None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN or None,
        )
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        logger.info("Loaded with transformers + PEFT")

    # ── Build dataset ─────────────────────────────────────────────
    dataset = build_dataset(args.server_url, args.num_episodes)
    if len(dataset) == 0:
        logger.error("Empty dataset — aborting.")
        sys.exit(1)

    # ── GRPO config ───────────────────────────────────────────────
    from trl import GRPOConfig, GRPOTrainer

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=args.max_seq_length // 2,
        max_completion_length=args.max_seq_length // 2,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        num_train_epochs=1,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        temperature=0.9,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=[make_alice_reward_fn(args.server_url)],
    )

    # ── Train ─────────────────────────────────────────────────────
    logger.info("Starting GRPO training (%d examples)...", len(dataset))
    result = trainer.train()
    logger.info("Training complete: %s", result)

    # ── Save ─────────────────────────────────────────────────────
    adapter_path = f"{args.output_dir}/lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved to %s", adapter_path)

    save_plots(trainer.state.log_history, args.output_dir)

    print(f"\n[END] Training complete. Adapter: {adapter_path}")


if __name__ == "__main__":
    main()
