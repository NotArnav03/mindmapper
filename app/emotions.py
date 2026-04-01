"""Emotion extraction and timeline visualization for therapy transcripts.

Splits transcripts into chunks, prompts Mistral for emotion analysis on each,
and renders an interactive Plotly timeline chart.
"""

import json
import re

import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.model import generate


EMOTION_PROMPT_TEMPLATE = """\
[INST] You are a clinical emotion analyst. Read this excerpt from a therapy session \
and identify the dominant emotion.

Return ONLY a JSON object with these exact keys:
- "emotion": the primary emotion (e.g., anxiety, sadness, anger, joy, calm, frustration, grief, fear, hope)
- "intensity": a float from 0.0 to 1.0 indicating emotional intensity
- "quote": a short direct quote from the text that best captures this emotion

Excerpt:
{chunk}
[/INST]"""

# Color mapping for emotion categories
EMOTION_COLORS: dict[str, str] = {
    "joy": "teal",
    "calm": "teal",
    "hope": "teal",
    "contentment": "teal",
    "relief": "teal",
    "sadness": "royalblue",
    "grief": "royalblue",
    "loneliness": "royalblue",
    "depression": "royalblue",
    "anger": "coral",
    "frustration": "coral",
    "irritation": "coral",
    "resentment": "coral",
    "anxiety": "goldenrod",
    "fear": "goldenrod",
    "worry": "goldenrod",
    "panic": "goldenrod",
    "nervousness": "goldenrod",
}


def extract_emotions(
    transcript: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    num_chunks: int = 5,
) -> list[dict]:
    """Extract emotions from evenly-split transcript chunks.

    Args:
        transcript: Full therapy session transcript text.
        model: The loaded language model.
        tokenizer: The corresponding tokenizer.
        num_chunks: Number of segments to split the transcript into.

    Returns:
        List of dicts, each with keys 'emotion', 'intensity', 'quote', 'chunk_index'.
    """
    if not transcript or not transcript.strip():
        return []

    chunks = _split_into_chunks(transcript.strip(), num_chunks)
    emotions = []

    for i, chunk in enumerate(chunks):
        prompt = EMOTION_PROMPT_TEMPLATE.format(chunk=chunk)
        response = generate(prompt, model, tokenizer, max_new_tokens=200, temperature=0.3)
        parsed = _parse_emotion_json(response, chunk)
        parsed["chunk_index"] = i
        emotions.append(parsed)

    return emotions


def _split_into_chunks(text: str, num_chunks: int) -> list[str]:
    """Split text into approximately equal chunks.

    Args:
        text: The text to split.
        num_chunks: Desired number of chunks.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if not words:
        return [""]

    chunk_size = max(1, len(words) // num_chunks)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    # Merge any small trailing chunk into the last full chunk
    while len(chunks) > num_chunks and len(chunks) > 1:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


def _parse_emotion_json(response: str, fallback_chunk: str) -> dict:
    """Safely parse emotion JSON from model output.

    Args:
        response: Raw model output string.
        fallback_chunk: Original text chunk for fallback quote.

    Returns:
        Dict with keys 'emotion', 'intensity', 'quote'.
    """
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            return {
                "emotion": str(parsed.get("emotion", "unknown")).lower(),
                "intensity": min(1.0, max(0.0, float(parsed.get("intensity", 0.5)))),
                "quote": str(parsed.get("quote", fallback_chunk[:80])),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {
        "emotion": "unknown",
        "intensity": 0.5,
        "quote": fallback_chunk[:80] + "...",
    }


def plot_emotion_timeline(emotion_data: list[dict]) -> go.Figure:
    """Create an interactive Plotly line chart of emotions across the session.

    Args:
        emotion_data: List of emotion dicts from extract_emotions().

    Returns:
        Plotly Figure with color-coded emotion timeline.
    """
    if not emotion_data:
        fig = go.Figure()
        fig.add_annotation(text="No emotion data to display", showarrow=False)
        fig.update_layout(title="Emotion Timeline")
        return fig

    n = len(emotion_data)
    x_labels = _generate_x_labels(n)

    x_vals = list(range(n))
    y_vals = [e["intensity"] for e in emotion_data]
    colors = [EMOTION_COLORS.get(e["emotion"], "gray") for e in emotion_data]
    hover_texts = [
        f"<b>{e['emotion'].title()}</b> ({e['intensity']:.2f})<br><i>\"{e['quote']}\"</i>"
        for e in emotion_data
    ]

    fig = go.Figure()

    # Line trace
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color="slategray", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Individual points colored by emotion
    for i, e in enumerate(emotion_data):
        color = EMOTION_COLORS.get(e["emotion"], "gray")
        fig.add_trace(
            go.Scatter(
                x=[x_vals[i]],
                y=[y_vals[i]],
                mode="markers",
                marker=dict(size=14, color=color, line=dict(width=2, color="white")),
                hovertext=hover_texts[i],
                hoverinfo="text",
                name=e["emotion"].title(),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(text="Emotion Timeline", font=dict(size=18, color="#2c3e50")),
        xaxis=dict(
            tickvals=x_vals,
            ticktext=x_labels,
            title="Session Progress",
            showgrid=False,
        ),
        yaxis=dict(
            title="Emotional Intensity",
            range=[0, 1.05],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        plot_bgcolor="rgba(248,250,252,1)",
        paper_bgcolor="rgba(248,250,252,1)",
        height=400,
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    return fig


def _generate_x_labels(n: int) -> list[str]:
    """Generate readable x-axis labels for emotion timeline.

    Args:
        n: Number of data points.

    Returns:
        List of label strings from 'Session start' to 'Session end'.
    """
    if n <= 1:
        return ["Session"]
    if n == 2:
        return ["Session start", "Session end"]

    labels = ["Session start"]
    for i in range(1, n - 1):
        labels.append(f"Part {i + 1}")
    labels.append("Session end")
    return labels
