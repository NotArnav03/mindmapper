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


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Variables ─────────────────────────────────────────────── */
:root {
    --bg:           #f0f3f7;
    --surface:      #ffffff;
    --surface-2:    #f7f9fc;
    --primary:      #1c3557;
    --primary-mid:  #24466e;
    --accent:       #3d7d7e;
    --accent-soft:  #eaf3f3;
    --success:      #4a8b71;
    --text-1:       #111827;
    --text-2:       #4b5563;
    --text-3:       #9ca3af;
    --border:       #dde3ed;
    --border-focus: #3d7d7e;
    --shadow-xs:    0 1px 2px rgba(0,0,0,0.05);
    --shadow-sm:    0 2px 8px rgba(0,0,0,0.07);
    --shadow-md:    0 6px 20px rgba(0,0,0,0.09);
    --r:            10px;
}

/* ── Page ───────────────────────────────────────────────────── */
body { background: var(--bg) !important; }

.gradio-container {
    background: var(--bg) !important;
    max-width: 1080px !important;
    padding: 0 1rem 3rem !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

footer { display: none !important; }

/* ── Header ─────────────────────────────────────────────────── */
.mm-header {
    background: linear-gradient(135deg, #1c3557 0%, #24466e 60%, #2a5580 100%);
    border-radius: 0 0 20px 20px;
    padding: 2.2rem 2.8rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.mm-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.mm-header::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 300px; height: 180px;
    background: rgba(61,125,126,0.15);
    border-radius: 50%;
}
.mm-logo {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 2rem;
    font-weight: 400;
    color: #ffffff;
    letter-spacing: -0.01em;
    margin: 0 0 0.3rem;
    position: relative; z-index: 1;
}
.mm-logo span {
    color: #7ecece;
}
.mm-tagline {
    font-size: 0.875rem;
    font-weight: 400;
    color: rgba(255,255,255,0.6);
    letter-spacing: 0.03em;
    margin: 0;
    position: relative; z-index: 1;
}
.mm-pills {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.2rem;
    position: relative; z-index: 1;
}
.mm-pill {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Section labels ─────────────────────────────────────────── */
.mm-section-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-3) !important;
    margin-bottom: 0.5rem !important;
}

/* ── Input panel ────────────────────────────────────────────── */
.input-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem;
    box-shadow: var(--shadow-xs);
}

.input-panel textarea {
    background: var(--surface-2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    color: var(--text-1) !important;
    resize: vertical !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.input-panel textarea:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px rgba(61,125,126,0.12) !important;
    outline: none !important;
}

/* ── Controls panel ─────────────────────────────────────────── */
.controls-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.4rem;
    box-shadow: var(--shadow-xs);
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

/* ── Buttons ────────────────────────────────────────────────── */
.btn-analyze button {
    width: 100%;
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.72rem 1.2rem !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 2px 8px rgba(28,53,87,0.25) !important;
}
.btn-analyze button:hover {
    background: var(--primary-mid) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(28,53,87,0.32) !important;
}
.btn-analyze button:active {
    transform: translateY(0) !important;
}

.btn-sample button {
    width: 100%;
    background: transparent !important;
    color: var(--text-2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.2rem !important;
    transition: all 0.2s !important;
}
.btn-sample button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── File upload ────────────────────────────────────────────── */
.file-upload .wrap {
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface-2) !important;
    padding: 0.9rem !important;
    transition: border-color 0.2s !important;
}
.file-upload .wrap:hover {
    border-color: var(--accent) !important;
    background: var(--accent-soft) !important;
}

/* ── Divider ────────────────────────────────────────────────── */
.mm-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Output cards ───────────────────────────────────────────── */
.output-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.6rem 1.8rem;
    box-shadow: var(--shadow-xs);
}

/* SOAP markdown */
.soap-output .prose {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-1) !important;
    line-height: 1.75 !important;
    font-size: 0.9rem !important;
}
.soap-output .prose h1 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 1.35rem !important;
    font-weight: 400 !important;
    color: var(--primary) !important;
    border-bottom: 2px solid var(--accent-soft) !important;
    padding-bottom: 0.6rem !important;
    margin-bottom: 1.2rem !important;
}
.soap-output .prose h2 {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--accent) !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.5rem !important;
}
.soap-output .prose ul {
    padding-left: 1.1rem !important;
}
.soap-output .prose li {
    color: var(--text-2) !important;
    margin-bottom: 0.4rem !important;
    font-size: 0.875rem !important;
}
.soap-output .prose li::marker {
    color: var(--accent) !important;
}

/* Plot card */
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    box-shadow: var(--shadow-xs);
    overflow: hidden;
}

/* JSON accordion */
.json-accordion {
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-xs) !important;
}
.json-accordion > .label-wrap {
    padding: 0.8rem 1.2rem !important;
}
.json-accordion > .label-wrap span {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-3) !important;
}

/* Gradio labels */
label > span {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: var(--text-3) !important;
}

/* Progress bar */
.progress-bar-wrap .progress-bar {
    background: var(--accent) !important;
}

/* Status text under progress */
.eta-bar { color: var(--text-3) !important; font-size: 0.78rem !important; }
"""

HEADER_HTML = """
<div class="mm-header">
    <p class="mm-logo">Mind<span>Mapper</span></p>
    <p class="mm-tagline">Clinical session intelligence — structured documentation, effortlessly.</p>
    <div class="mm-pills">
        <span class="mm-pill">SOAP Notes</span>
        <span class="mm-pill">Emotion Timeline</span>
        <span class="mm-pill">Mistral-7B · 4-bit</span>
        <span class="mm-pill">Fully Private</span>
    </div>
</div>
"""


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
            "_Please paste or upload a therapy transcript to begin._",
            None,
            {"error": "No transcript provided"},
        )

    progress(0.1, desc="Initialising model...")
    model = _model
    tokenizer = _tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    progress(0.3, desc="Generating SOAP note...")
    soap_dict = generate_soap(transcript, model, tokenizer)
    soap_md = format_soap_markdown(soap_dict)

    progress(0.65, desc="Analysing emotional arc...")
    emotions = extract_emotions(transcript, model, tokenizer)

    progress(0.9, desc="Building timeline...")
    fig = plot_emotion_timeline(emotions)

    progress(1.0, desc="Done.")
    return soap_md, fig, soap_dict


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns:
        Configured Gradio Blocks instance.
    """
    with gr.Blocks(title="MindMapper") as app:

        # ── Header ──────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Input row ───────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=3, elem_classes="input-panel"):
                gr.HTML('<p class="mm-section-label">Session Transcript</p>')
                transcript_input = gr.Textbox(
                    show_label=False,
                    placeholder=(
                        "Paste the therapy session transcript here…\n\n"
                        "T: How have you been since our last session?\n"
                        "P: It's been a difficult week. I've been feeling…"
                    ),
                    lines=14,
                    max_lines=30,
                )

            with gr.Column(scale=1, elem_classes="controls-panel"):
                gr.HTML('<p class="mm-section-label">Actions</p>')
                submit_btn = gr.Button(
                    "Analyse Session",
                    variant="primary",
                    elem_classes="btn-analyze",
                )
                sample_btn = gr.Button(
                    "Load Sample",
                    variant="secondary",
                    elem_classes="btn-sample",
                )
                gr.HTML('<hr class="mm-divider" style="margin: 0.25rem 0;">')
                gr.HTML('<p class="mm-section-label">Upload</p>')
                file_upload = gr.File(
                    show_label=False,
                    file_types=[".txt"],
                    type="filepath",
                    elem_classes="file-upload",
                )

        gr.HTML('<hr class="mm-divider">')

        # ── Outputs ─────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1, elem_classes="output-card soap-output"):
                gr.HTML('<p class="mm-section-label">SOAP Note</p>')
                soap_output = gr.Markdown(
                    value="_Your structured clinical note will appear here._",
                    show_label=False,
                )

        with gr.Row():
            with gr.Column(elem_classes="chart-card"):
                emotion_plot = gr.Plot(show_label=False)

        with gr.Accordion(
            "Raw JSON Output", open=False, elem_classes="json-accordion"
        ):
            json_output = gr.JSON(show_label=False)

        # ── Event handlers ───────────────────────────────────────
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
