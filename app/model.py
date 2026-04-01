"""Model loading and inference for MindMapper.

Loads Mistral-7B-Instruct-v0.2 in 4-bit quantization using bitsandbytes,
with optional LoRA adapter support for fine-tuned weights.
"""

import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_ADAPTER_PATH = "./lora_adapter"


@lru_cache(maxsize=1)
def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Mistral-7B in 4-bit quantization with optional LoRA adapter.

    Uses double quantization with NF4 for optimal T4 GPU memory usage (~5GB).
    Automatically loads a LoRA adapter from ./lora_adapter if the directory exists.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter if available
    if os.path.isdir(LORA_ADAPTER_PATH):
        from peft import PeftModel

        print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

    model.eval()
    return model, tokenizer


def generate(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """Generate text from a prompt using the loaded model.

    Args:
        prompt: The input prompt string.
        model: The loaded language model.
        tokenizer: The corresponding tokenizer.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        The generated text string, with the original prompt stripped.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip the prompt from the output
    if generated.startswith(prompt):
        generated = generated[len(prompt):]

    return generated.strip()
