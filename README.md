# MindMapper

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Mistral-7B](https://img.shields.io/badge/model-Mistral--7B-orange.svg)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
[![Gradio](https://img.shields.io/badge/UI-Gradio-green.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

**AI-powered therapy session summarizer** — upload a therapy transcript and get a structured SOAP note plus an interactive emotion timeline chart.

## The Problem

Therapists spend significant time writing clinical notes after sessions. MindMapper automates this by generating structured SOAP (Subjective, Objective, Assessment, Plan) notes and visualizing emotional patterns across a session — all running locally with no data sent to external APIs.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Gradio UI (ui.py)                  │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ Transcript   │  │ SOAP Note  │  │  Emotion     │ │
│  │ Input        │  │ (Markdown) │  │  Timeline    │ │
│  │ (.txt/paste) │  │            │  │  (Plotly)    │ │
│  └──────┬───────┘  └─────▲──────┘  └──────▲───────┘ │
└─────────┼────────────────┼─────────────────┼─────────┘
          │                │                 │
          ▼                │                 │
┌─────────────────┐  ┌─────┴───────┐  ┌─────┴────────┐
│  model.py       │  │  soap.py    │  │ emotions.py  │
│  Mistral-7B     │◄─┤  SOAP       │  │ Emotion      │
│  4-bit QLoRA    │◄─┤  Generation │  │ Extraction   │
│  + LoRA adapter │  │  + Parsing  │  │ + Plotly     │
└─────────────────┘  └─────────────┘  └──────────────┘
          ▲
          │ (optional fine-tune)
┌─────────┴─────────────────────────┐
│  training/                         │
│  train.py + dataset.py + config   │
│  QLoRA fine-tuning pipeline       │
└───────────────────────────────────┘
```

## Setup

### Local (requires NVIDIA GPU with 8GB+ VRAM)

```bash
git clone https://github.com/YOUR_USERNAME/mindmapper.git
cd mindmapper
pip install -r requirements.txt
python -m app.ui
```

### Google Colab (recommended — free T4 GPU)

See [notebooks/COLAB_GUIDE.md](notebooks/COLAB_GUIDE.md) for step-by-step instructions.

## Usage

### Run the Gradio App

```bash
python -m app.ui
```

This launches a web interface where you can:
1. Paste a therapy transcript or upload a `.txt` file
2. Click "Analyze Session" to generate a SOAP note and emotion timeline
3. View the structured SOAP note in markdown format
4. Explore the interactive emotion timeline chart
5. Access raw SOAP data in JSON format

### Fine-tune on Therapy Data

```bash
python training/train.py
```

This runs QLoRA fine-tuning with the config in `training/config.yaml`. The adapter is saved to `./lora_adapter` and automatically loaded on next model start.

## T4 GPU Memory Usage

| Operation | VRAM Usage | Time |
|-----------|-----------|------|
| Model load (4-bit) | ~5 GB | 3-4 min |
| SOAP inference | ~7 GB peak | 15-30 sec |
| Emotion extraction (5 chunks) | ~7 GB peak | 60-90 sec |
| Fine-tuning (batch=2, grad_accum=4) | ~12 GB peak | ~45 min |

## Example Output

### SOAP Note

```markdown
## Subjective
- Patient reports intense work stress with multiple overlapping deadlines
- Describes crying episode in parking lot after working until midnight
- Reports impostor syndrome feelings that intensify with sleep deprivation
- Acknowledges difficulty relaxing and guilt around non-productive activities

## Objective
- Patient appears fatigued but engaged in session
- Shows emotional awareness when connecting sleep patterns to negative self-talk
- Demonstrates willingness to try behavioral interventions

## Assessment
- Work-related anxiety with significant impact on sleep quality
- Impostor syndrome exacerbated by chronic sleep deprivation
- Self-worth heavily tied to productivity — pattern recognized by patient
- Notable progress in self-awareness compared to 6 months ago

## Plan
- Replace phone-checking with 4-7-8 breathing on 2 nights this week
- Observe morning self-talk patterns following breathing exercises
- Follow-up next Thursday to review sleep intervention outcomes
- Consider structured relaxation techniques in future sessions
```

## Tech Stack

- **Model:** [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) with 4-bit quantization
- **Fine-tuning:** QLoRA via [PEFT](https://github.com/huggingface/peft) + [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Training:** [TRL SFTTrainer](https://github.com/huggingface/trl)
- **UI:** [Gradio](https://gradio.app/)
- **Visualization:** [Plotly](https://plotly.com/python/)
- **Data:** [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
