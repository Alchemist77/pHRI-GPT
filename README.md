# pHRI-GPT

pHRI-GPT is a four-stage pipeline for **post-trial human experience description** in **physical human–robot interaction (pHRI)**.

Given synchronized multimodal pHRI observations (e.g., interaction time-series and RGB-D image streams), the pipeline learns an interaction-aware latent representation, aligns it to an instruction-tuned LLM, generates a trial-level natural-language experience description, and evaluates semantic consistency against the trial self-report using an external LLM judge. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

This repository contains the main cross-validation scripts for the full pipeline:

- `stage0_train_cv.py` — representation learning + encoder-only classification (subject-wise CV; saves fold indices and encoder checkpoints) :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
- `stage1_train_cv.py` — LoRA training with **frozen Stage-0 encoders** and a latent-to-prefix projector for LLM conditioning :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
- `stage2_generate_cv.py` — description generation using the trained projector + LoRA adapter (fold-wise generation) :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
- `stage3_evaluate_cv.py` — external LLM-based judging (Negative/Positive) and fold-wise aggregation of evaluation results :contentReference[oaicite:8]{index=8}

> **Status:** This work has been submitted to **IEEE Robotics and Automation Letters (RA-L)** (under review).

---

## Pipeline Overview (4 stages)

### Stage 0 — Multimodal representation learning / encoder-only evaluation
Trains time-series, image, or fusion encoders (`ts`, `img`, `fusion`) with subject-wise cross-validation and saves per-fold checkpoints and metrics. Output includes encoder weights, classifier head, fold splits, and validation metrics. :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}

### Stage 1 — Latent-to-language alignment (LoRA + projector)
Loads **frozen Stage-0 encoder(s)**, extracts latent vectors, and trains a projector + LoRA adapter so the LLM can describe the human experience from latent-conditioned prefix embeddings. The prompt template is fixed in code. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}

### Stage 2 — Description generation
Generates one natural English sentence per trial using the Stage-1 adapter/projector and fold-specific Stage-0 encoders. The script injects the latent-conditioned prefix into the prompt and decodes only the generated continuation. :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

### Stage 3 — LLM-judge evaluation
Loads Stage-2 generated descriptions and evaluates them with an external judge LLM (default path in script points to GPT-OSS-20B), producing fold-wise predictions and summary metrics. :contentReference[oaicite:15]{index=15}

---

## Requirements (high level)

- Python + PyTorch
- Hugging Face `transformers`
- `peft` (LoRA)
- `unsloth` (used in Stage 1 script) :contentReference[oaicite:16]{index=16}
- Dataset loaders used by the scripts:
  - `interaction_timeseries_loader.py`
  - `interaction_image_loader.py` :contentReference[oaicite:17]{index=17}

> The scripts expect a dataset root and a CSV label file (default examples use `../dataset_v5` and `generated_phri_captions_rebalanced.csv`). 

---

## How to Run (4 scripts)

Below is a typical **fusion** pipeline example (`Transformer + ResNet18`, 5-fold CV, `K=16` frames).

### 1) Stage 0 — Train encoder(s) + classifier (CV)
```bash
python stage0_train_cv.py \
  --root ../dataset_v5 \
  --label_csv ../dataset_v5/generated_phri_captions_rebalanced.csv \
  --task fusion \
  --ts_backbone transformer \
  --img_backbone resnet18 \
  --k_frames 16 \
  --kfold 5 \
  --seed 42 \
  --latent 256 \
  --out_root ./stage0_out
