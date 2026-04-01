"""SOAP note generation and parsing for therapy transcripts.

Prompts Mistral to produce structured clinical documentation in SOAP format
(Subjective, Objective, Assessment, Plan) from therapy session transcripts.
"""

import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

from app.model import generate


SOAP_PROMPT_TEMPLATE = """\
[INST] You are a clinical documentation assistant. Analyze the following therapy \
session transcript and produce a SOAP note in valid JSON format.

Return ONLY a JSON object with these exact keys:
- "subjective": list of bullet points (patient's reported feelings, concerns, experiences)
- "objective": list of bullet points (therapist's observations of behavior, affect, engagement)
- "assessment": list of bullet points (clinical impressions, themes, progress)
- "plan": list of bullet points (recommended next steps, homework, follow-up)

Transcript:
{transcript}
[/INST]"""


def generate_soap(
    transcript: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, list[str]]:
    """Generate a SOAP note from a therapy transcript.

    Args:
        transcript: The full therapy session transcript text.
        model: The loaded language model.
        tokenizer: The corresponding tokenizer.

    Returns:
        Dict with keys 'subjective', 'objective', 'assessment', 'plan',
        each mapping to a list of bullet-point strings.
    """
    if not transcript or not transcript.strip():
        return {
            "subjective": ["No transcript provided."],
            "objective": ["No transcript provided."],
            "assessment": ["No transcript provided."],
            "plan": ["No transcript provided."],
        }

    prompt = SOAP_PROMPT_TEMPLATE.format(transcript=transcript.strip())
    response = generate(prompt, model, tokenizer, max_new_tokens=512, temperature=0.3)

    return _parse_soap_json(response)


def _parse_soap_json(response: str) -> dict[str, list[str]]:
    """Safely parse SOAP JSON from model output, with fallback handling.

    Args:
        response: Raw model output string that should contain JSON.

    Returns:
        Parsed SOAP dict with all four keys guaranteed.
    """
    # Try to extract JSON from the response
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            soap = {}
            for key in ("subjective", "objective", "assessment", "plan"):
                value = parsed.get(key, [])
                if isinstance(value, list):
                    soap[key] = [str(item) for item in value]
                elif isinstance(value, str):
                    soap[key] = [value]
                else:
                    soap[key] = [str(value)]
            return soap
        except json.JSONDecodeError:
            pass

    # Fallback: return raw response as subjective with empty other fields
    return {
        "subjective": [response if response else "Failed to parse model output."],
        "objective": ["[Parse error — raw output returned in Subjective]"],
        "assessment": ["[Parse error — raw output returned in Subjective]"],
        "plan": ["[Parse error — re-run or manually review]"],
    }


def format_soap_markdown(soap_dict: dict[str, list[str]]) -> str:
    """Format a SOAP dict into readable markdown.

    Args:
        soap_dict: Dict with keys 'subjective', 'objective', 'assessment', 'plan'.

    Returns:
        Formatted markdown string.
    """
    sections = [
        ("Subjective", "subjective"),
        ("Objective", "objective"),
        ("Assessment", "assessment"),
        ("Plan", "plan"),
    ]

    lines = ["# SOAP Note\n"]
    for title, key in sections:
        lines.append(f"## {title}")
        for item in soap_dict.get(key, []):
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines)
