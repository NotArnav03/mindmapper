"""Gradio web interface for MindMapper.

Provides a clean, professional UI for uploading therapy transcripts
and viewing SOAP notes with emotion timeline visualizations.
"""

import os

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.model import load_model
from app.soap import generate_soap, format_soap_markdown
from app.emotions import extract_emotions, plot_emotion_timeline


SAMPLE_TRANSCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "sample_transcript.txt"
)

# Calming blue/teal theme
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.gradio-container { max-width: 960px !important; }
.prose h1 { color: #2c3e50 !important; }
.prose h2 { color: #34495e !important; }
"""

# Module-level model cache — can be set externally to reuse an already-loaded model
_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None


def set_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    """Inject a pre-loaded model into the UI to avoid reloading.

    Call this before build_app() when running inline in Colab to share
    the already-loaded model instead of loading it again in a subprocess.

    Args:
        model: Pre-loaded language model.
        tokenizer: Corresponding tokenizer.
    """
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


def load_sample() -> str:
    """Load the sample transcript from disk.

    Returns:
        The sample transcript text, or an error message if the file is missing.
    """
    if os.path.exists(SAMPLE_TRANSCRIPT_PATH):
        with open(SAMPLE_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "Sample transcript not found. Please paste a transcript manually."


def read_uploaded_file(file) -> str:
    """Read text content from an uploaded file.

    Args:
        file: Gradio file object.

    Returns:
        The file's text content.
    """
    if file is None:
        return ""
    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()


def process_transcript(
    transcript: str, progress: gr.Progress = gr.Progress()
) -> tuple[str, object, dict]:
    """Run SOAP generation and emotion extraction on a transcript.

    Args:
        transcript: The therapy session transcript text.
        progress: Gradio progress tracker for UI feedback.

    Returns:
        Tuple of (SOAP markdown, Plotly figure, raw SOAP dict).
    """
    if not transcript or not transcript.strip():
        return (
            "**Please paste or upload a therapy transcript.**",
            None,
            {"error": "No transcript provided"},
        )

    progress(0.1, desc="Loading model...")
    model = _model
    tokenizer = _tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    progress(0.3, desc="Generating SOAP note...")
    soap_dict = generate_soap(transcript, model, tokenizer)
    soap_md = format_soap_markdown(soap_dict)

    progress(0.6, desc="Analyzing emotions...")
    emotions = extract_emotions(transcript, model, tokenizer)

    progress(0.9, desc="Creating timeline...")
    fig = plot_emotion_timeline(emotions)

    return soap_md, fig, soap_dict


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns:
        Configured Gradio Blocks instance.
    """
    with gr.Blocks(title="MindMapper") as app:
        gr.Markdown(
            "# MindMapper\n"
            "*AI-powered therapy session summarizer — "
            "paste a transcript to generate a SOAP note and emotion timeline.*"
        )

        with gr.Row():
            with gr.Column(scale=2):
                transcript_input = gr.Textbox(
                    label="Paste therapy transcript",
                    placeholder="T: How are you feeling today?\nP: I've been struggling with...",
                    lines=15,
                    max_lines=30,
                )
                file_upload = gr.File(
                    label="Or upload a .txt file",
                    file_types=[".txt"],
                    type="filepath",
                )
            with gr.Column(scale=1):
                sample_btn = gr.Button("Load Sample Transcript", variant="secondary")
                submit_btn = gr.Button("Analyze Session", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            soap_output = gr.Markdown(label="SOAP Note")

        with gr.Row():
            emotion_plot = gr.Plot(label="Emotion Timeline")

        with gr.Accordion("Raw SOAP Data (JSON)", open=False):
            json_output = gr.JSON(label="SOAP JSON")

        # Event handlers
        sample_btn.click(fn=load_sample, outputs=transcript_input)

        file_upload.change(
            fn=read_uploaded_file, inputs=file_upload, outputs=transcript_input
        )

        submit_btn.click(
            fn=process_transcript,
            inputs=transcript_input,
            outputs=[soap_output, emotion_plot, json_output],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True, server_name="0.0.0.0", theme=THEME, css=CSS)
