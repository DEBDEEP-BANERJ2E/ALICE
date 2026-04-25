#!/usr/bin/env python3
"""
ALICE Training Script — GRPO training with synthetic dataset (no ngrok needed).
"""

import argparse
import logging
import os
import sys
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def generate_synthetic_dataset(num_samples: int = 20) -> list[dict]:
    """Generate synthetic training data for DPO (chosen/rejected pairs)."""
    templates = [
        "If the result is NOT {wrong}, what is the actual result of {a} + {b}?",
        "The answer is NOT {wrong}. What is {a} - {b}?",
        "Ignore the false claim that the result is {wrong}. Compute {a} * {b}.",
    ]
    
    dataset = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        a, b, wrong = (i % 10) + 1, (i % 5) + 1, (i % 8) + 1
        
        prompt = template.format(a=a, b=b, wrong=wrong)
        
        # Chosen: correct reasoning
        chosen = f"{prompt}\n\nThink step by step. Show your full reasoning chain.\nThen on the final line write exactly: Answer: {a + b}"
        
        # Rejected: incorrect reasoning
        rejected = f"{prompt}\n\nAnswer: {wrong}"
        
        dataset.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    logger.info(f"[TRAIN] Generated {len(dataset)} synthetic DPO pairs")
    return dataset



def main():
    parser = argparse.ArgumentParser(description="ALICE GRPO Training Script")
    parser.add_argument("--model-name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="./output/alice-trained")
    parser.add_argument("--lora-rank", type=int, default=8)

    args = parser.parse_args()

    logger.info(f"[TRAIN] Starting ALICE training")
    logger.info(f"[TRAIN] Model: {args.model_name}")
    logger.info(f"[TRAIN] Episodes: {args.num_episodes}")

    # Generate synthetic dataset (no server needed)
    logger.info(f"[TRAIN] Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(args.num_episodes)

    if not dataset:
        logger.error("[TRAIN] Failed to generate dataset. Exiting.")
        sys.exit(1)

    # Load model
    logger.info("[TRAIN] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        token=HF_TOKEN or None,
        torch_dtype="auto",
    )

    # Apply LoRA
    logger.info("[TRAIN] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Setup DPO training
    logger.info("[TRAIN] Setting up DPO training...")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        logging_steps=1,
        save_steps=50,
        fp16=False,
        bf16=False,
        report_to="none",
        max_length=256,
        max_prompt_length=128,
    )

    train_dataset = Dataset.from_list(dataset)

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    logger.info("[TRAIN] Starting DPO training loop...")
    trainer.train()

    logger.info("[TRAIN] Saving trained model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("[TRAIN] Training completed successfully")
    print(f"[END] Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
