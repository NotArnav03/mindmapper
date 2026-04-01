"""QLoRA fine-tuning script for MindMapper.

Fine-tunes Mistral-7B-Instruct-v0.2 on therapy transcript SOAP note data
using 4-bit QLoRA with PEFT, optimized for T4 GPU.
"""

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from training.dataset import get_dataset


def load_config() -> dict:
    """Load training configuration from config.yaml.

    Returns:
        Dict of training hyperparameters.
    """
    config_path = "training/config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train() -> None:
    """Run QLoRA fine-tuning pipeline.

    Loads the base model in 4-bit, applies LoRA adapters, trains on
    therapy SOAP note data, and saves the adapter weights.
    """
    config = load_config()

    print("Loading dataset...")
    train_dataset, val_dataset = get_dataset()
    print(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = config["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        fp16=True,
        learning_rate=config["learning_rate"],
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        eval_strategy="epoch",
        report_to="none",
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=config["max_seq_length"],
    )

    result = trainer.train()

    # Print training summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"  Total steps: {result.global_step}")
    print(f"  Training loss: {result.training_loss:.4f}")
    print("=" * 50)

    # Print loss curve
    print("\nTraining Loss Curve:")
    log_history = trainer.state.log_history
    losses = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]
    if losses:
        max_loss = max(l for _, l in losses)
        for step, loss in losses:
            bar_len = int(40 * loss / max_loss) if max_loss > 0 else 0
            print(f"  Step {step:4d} | {'█' * bar_len} {loss:.4f}")

    print(f"\nSaving adapter to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    train()
