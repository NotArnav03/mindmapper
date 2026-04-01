"""Dataset loading and formatting for therapy SOAP note fine-tuning.

Loads therapy transcripts from HuggingFace and formats them as
instruction-tuning prompt/completion pairs for Mistral.
"""

from datasets import Dataset, load_dataset


PROMPT_TEMPLATE = (
    "[INST] You are a clinical documentation assistant. "
    "Summarize this therapy session as a SOAP note in JSON format.\n\n"
    "Transcript:\n{transcript} [/INST]"
)

SYNTHETIC_EXAMPLES = [
    {
        "transcript": (
            "T: How have you been this week?\n"
            "P: Honestly, not great. I've been having trouble sleeping again.\n"
            "T: What's been keeping you up?\n"
            "P: Work stress mostly. My boss keeps piling on deadlines.\n"
            "T: How does that make you feel?\n"
            "P: Overwhelmed. Like I can never catch up."
        ),
        "soap": (
            '{"subjective": ["Patient reports poor sleep due to work stress", '
            '"Feels overwhelmed by increasing deadlines"], '
            '"objective": ["Appears fatigued", "Flat affect during discussion of work"], '
            '"assessment": ["Work-related anxiety impacting sleep hygiene", '
            '"Possible burnout trajectory"], '
            '"plan": ["Introduce sleep hygiene checklist", '
            '"Discuss boundary-setting strategies next session"]}'
        ),
    },
    {
        "transcript": (
            "T: Last time we talked about your relationship with your mother.\n"
            "P: Yeah, I've been thinking about that a lot.\n"
            "T: What came up for you?\n"
            "P: I realized I've been trying to earn her approval my whole life.\n"
            "T: That's a significant insight. How does that awareness feel?\n"
            "P: Sad, but also kind of freeing."
        ),
        "soap": (
            '{"subjective": ["Patient reflects on relationship with mother", '
            '"Reports realization about lifelong approval-seeking pattern", '
            '"Describes mixed emotions: sadness and liberation"], '
            '"objective": ["Engaged and reflective", "Tearful but composed"], '
            '"assessment": ["Progressing in insight development around attachment patterns", '
            '"Emotional processing is healthy and age-appropriate"], '
            '"plan": ["Continue exploration of family dynamics", '
            '"Introduce journaling exercise on attachment needs"]}'
        ),
    },
]


def _generate_synthetic_dataset(n: int = 50) -> list[dict]:
    """Generate synthetic training examples by varying the base examples.

    Args:
        n: Number of synthetic examples to generate.

    Returns:
        List of dicts with 'text' key containing formatted prompt/completion pairs.
    """
    examples = []
    for i in range(n):
        base = SYNTHETIC_EXAMPLES[i % len(SYNTHETIC_EXAMPLES)]
        text = PROMPT_TEMPLATE.format(transcript=base["transcript"]) + "\n" + base["soap"]
        examples.append({"text": text})
    return examples


def get_dataset() -> tuple[Dataset, Dataset]:
    """Load and format therapy transcript dataset for fine-tuning.

    Attempts to load from HuggingFace first, falls back to synthetic data.
    Formats examples as instruction-tuning prompt/completion pairs.

    Returns:
        Tuple of (train_dataset, val_dataset) with 90/10 split.
    """
    try:
        raw_dataset = load_dataset("helpmefind/therapy-transcripts", split="train")
        formatted = []
        for example in raw_dataset:
            transcript = example.get("text", example.get("transcript", ""))
            if not transcript:
                continue
            text = (
                PROMPT_TEMPLATE.format(transcript=transcript)
                + "\n"
                + '{"subjective": [], "objective": [], "assessment": [], "plan": []}'
            )
            formatted.append({"text": text})

        if len(formatted) < 10:
            raise ValueError("Too few examples, falling back to synthetic data")

        dataset = Dataset.from_list(formatted)
        print(f"Loaded {len(dataset)} examples from HuggingFace")

    except Exception as e:
        print(f"HuggingFace dataset unavailable ({e}), using synthetic data")
        synthetic = _generate_synthetic_dataset(50)
        dataset = Dataset.from_list(synthetic)
        print(f"Generated {len(dataset)} synthetic examples")

    split = dataset.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]
