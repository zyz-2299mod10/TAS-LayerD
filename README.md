[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YIWZWoQd)
# 2025 DLCV Final Project: Challenge 2 - Layer Decomposition

**Collaboration with PicCollage**

**Slides:** https://docs.google.com/presentation/d/1wbZBPT7-C7j0wt5RqpNxZMWoqZKYw1fdizRz9hojfeQ/edit?usp=sharing

This repository contains the official evaluation framework and dataset utilities for Challenge 2 of the 2025 Deep Learning for Computer Vision (DLCV) final project.

## 📌 Task Overview

The goal of this challenge is **Layer Decomposition**. Given a single flattened input image, your model must decompose it into a sequence of layers (RGBA) that, when composited from bottom to top, reconstruct the original image.

Unlike standard layer decomposition tasks, this challenge involves:
* **Complex Samples:** High variability in element patterns and number of layers.
* **Overlapping Elements:** Significant occlusion between layers that requires inpainting/generation.
* **Ordering:** The output order matters (background → foreground).

#### Test Set Assumptions
1. **Opacity:** All layers are fully opaque (opacity = 1.0).  
2. **Background:** A background layer is always present.

---

## 📂 Dataset

The dataset contains raster graphic designs with metadata and pre-rendered elements.

**Download:** https://huggingface.co/datasets/WalkerHsu/DLCV2025_final_project_piccollage

### Data Structure

Each sample includes:

#### 1. Canvas-Level Information
- `canvas_width`, `canvas_height`
- `image`: fully-rendered RGB image (input)
- Metadata (format, tags)

#### 2. Element-Level Information
- Sequence of elements (image, SVG, text, background)
- Pre-rendered PNG with alpha
- Geometry: `left`, `top`, `width`, `height`, `rotation_angle`
- Ground Truth generation uses `src/utils/crello_utils.py`

---

## 🚀 Usage

#### Install (uv recommended)
Make sure you have installed `uv`
```bash
pip install uv
```

install the environment
```bash
uv sync
```

activate environment
```bash
source ./.venv/bin/activate
```
### Quick inference
**Download checkpoints**
```bash
bash download_ckpt.sh
```
This will download the checkpoints to `hi_sam/pretrained_checkpoint/efficient_hi_sam_s.pth`, `hi_sam/pretrained_checkpoint/sam_vit_h_4b8939.pth` and `src/layerd/weight/matting_model.pth`

**Inference**
```bash
python ./infer.py \
    --input <input folder> \
    --output-dir <output dir> \
    --matting-weight-path ./src/layerd/weight/matting_model.pth \
    --extract-text \
    --device cuda
```

### Training
#### data preprocessing
```bash
python ./data_preprocess.py --output-dir <output folder> --split train --save-layers --inpainting
```
This may cost 20~25 hours

#### Train
```bash

```

---
## 🚫 Rules & Restrictions

Violations result in **0 score**.

- No recovering ground truth layer metadata  
- No external data  
- No paid/closed-source API models (GPT‑4o, Claude, Midjourney, etc.)  
- Must be fully reproducible  

---

# Submission Rules
### Deadline
114/12/22 (Mon.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntu-dlcv-2025-fall-ta@googlegroups.com](mailto:ntu-dlcv-2025-fall-ta@googlegroups.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
