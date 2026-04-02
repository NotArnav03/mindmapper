# MindMapper — Progress Log

## Project Overview
- **Goal:** AI therapy session summarizer using QLoRA fine-tuned Mistral-7B
- **Stack:** Mistral-7B-Instruct, QLoRA, PEFT, Gradio, Plotly
- **Target:** T4 GPU (Google Colab free tier)
- **Status:** 🟡 In Progress

---

## Changelog

### [INIT] Project scaffolded
- Date: 2026-04-01
- Created full project structure
- All core files written: model.py, soap.py, emotions.py, ui.py
- Training pipeline ready: train.py, dataset.py, config.yaml
- Colab guide created
- Sample transcript added

---

## File Index
| File | Purpose | Status |
|------|---------|--------|
| app/model.py | 4-bit model loading + inference | ✅ Written |
| app/soap.py | SOAP note generation + parsing | ✅ Written |
| app/emotions.py | Emotion extraction + Plotly chart | ✅ Written |
| app/ui.py | Gradio interface | ✅ Written |
| training/train.py | QLoRA fine-tune script | ✅ Written |
| training/dataset.py | Dataset loader + formatter | ✅ Written |
| training/config.yaml | Training hyperparams | ✅ Written |
| notebooks/COLAB_GUIDE.md | Step-by-step Colab execution guide | ✅ Written |
| data/sample_transcript.txt | Sample therapy transcript | ✅ Written |

---

## Known Issues / TODOs
- [ ] Test on actual T4 GPU in Colab
- [ ] Validate JSON parsing edge cases in soap.py
- [ ] Find or confirm best HuggingFace therapy dataset
- [ ] Fine-tune and evaluate SOAP quality

---

## Session Notes
> Use this section to log what you did each time you work on this project.
> Format: ### [DATE] What you did

### [2026-04-02] Update Colab guide to inline UI launch method
- Replaced `!python -m app.ui` in Step 7 with inline `set_model()` + `build_app()` pattern
- Added troubleshooting entry for `cannot import name 'set_model'` error

### [2026-04-02] Fix VRAM crash when launching UI in Colab
- Added `set_model()` to `ui.py` so a pre-loaded model can be injected instead of reloaded
- Moved `theme`/`css` from `gr.Blocks()` constructor to `launch()` to fix Gradio 6.0 deprecation warnings
- Root cause: `!python -m app.ui` spawns a new process that tries to reload the model, exhausting T4 VRAM already used by the notebook kernel

### [2026-04-01] Initial project scaffold
- Created all 13 project files from spec
- Implemented 4-bit quantized Mistral-7B loading with LoRA adapter support
- Built SOAP note generation with JSON parsing and fallback handling
- Built emotion extraction pipeline with 5-chunk analysis and Plotly visualization
- Created Gradio UI with calming teal/blue theme, file upload, and sample loader
- Set up QLoRA training pipeline with SFTTrainer and config.yaml
- Wrote 400-word sample therapy transcript
- Created Colab execution guide and README with architecture diagram

---

## How to Update This File
Whenever you make a significant change, add an entry under Changelog:
### [YYYY-MM-DD] Short title
- What changed
- Why it changed
- Any issues encountered
Update the File Index status and Known Issues accordingly.

LOGGING RULE FOR CLAUDE CODE:
Whenever Claude Code makes any significant change to this project — adding a file, fixing a bug, changing architecture, completing a feature, running a test — it MUST append an entry to logs/PROGRESS.md under the Changelog section. Significant changes include:
- Any new file created
- Any function added or refactored
- Any bug fixed
- Any dependency added
- Any architectural decision made
- Any test run (pass or fail)
- Any model trained or evaluated
