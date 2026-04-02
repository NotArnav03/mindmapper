# MindMapper — Colab Execution Guide

## Before You Start
- Runtime: GPU → T4
- RAM: ~12GB used, 15GB available
- Estimated total run time: ~8 mins (first load) / ~2 mins (subsequent)

---

## Step 1 — Install dependencies
```python
!pip install torch transformers peft bitsandbytes trl datasets gradio plotly accelerate scipy sentencepiece
```

---

## Step 2 — Check GPU
```python
import torch
print(torch.cuda.get_device_name(0))
print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```
Expected output: Tesla T4, ~15.0 GB

---

## Step 3 — Clone the repo
```python
!git clone https://github.com/YOUR_USERNAME/mindmapper.git
%cd mindmapper
```

---

## Step 4 — Load the model
```python
from app.model import load_model
model, tokenizer = load_model()
```
Note: First run downloads ~4GB. Takes 3-4 mins on T4.

---

## Step 5 — Run inference on sample transcript
```python
from app.soap import generate_soap, format_soap_markdown
transcript = open("data/sample_transcript.txt").read()
soap = generate_soap(transcript, model, tokenizer)
print(format_soap_markdown(soap))
```

---

## Step 6 — Generate emotion timeline
```python
from app.emotions import extract_emotions, plot_emotion_timeline
emotions = extract_emotions(transcript, model, tokenizer)
fig = plot_emotion_timeline(emotions)
fig.show()
```

---

## Step 7 — Launch Gradio UI
```python
import importlib
import app.ui
importlib.reload(app.ui)

from app.ui import set_model, build_app
set_model(model, tokenizer)  # reuse already-loaded model — avoids VRAM crash
app = build_app()
app.launch(share=True)
```
Click the public URL that appears. Share it for demos.

> **Why not `!python -m app.ui`?** That spawns a new subprocess which tries to reload the
> model into already-occupied VRAM and crashes. Running inline shares the model already in memory.

---

## Step 8 (Optional) — Fine-tune on therapy data
```python
!python training/train.py
```
Estimated time: ~45 mins on T4 for 3 epochs.
Adapter saved to: ./lora_adapter

---

## Step 9 (Optional) — Reload model with fine-tuned adapter
```python
# Restart runtime first, then:
from app.model import load_model
model, tokenizer = load_model()  # auto-detects ./lora_adapter
```

---

## Troubleshooting
| Error | Fix |
|-------|-----|
| CUDA out of memory | Runtime → Factory reset, re-run from Step 1 |
| bitsandbytes not found | Re-run Step 1, restart runtime |
| JSON parse error in SOAP | Known edge case — re-run Step 5, model is non-deterministic |
| Gradio URL not appearing | Add share=True in ui.py launch() call |
| `cannot import name 'set_model'` | Run `!git pull` then use `importlib.reload(app.ui)` before importing |
