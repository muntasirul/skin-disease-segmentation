# Leakage-Aware, Source-Grouped Benchmarking of Multi-Class Clinical Skin-Disease Instance Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLO11%20%7C%20YOLO12%20%7C%20YOLO26-blue)](https://github.com/ultralytics/ultralytics)
[![SAM 2](https://img.shields.io/badge/Meta-SAM%202.1-green)](https://github.com/facebookresearch/sam2)

Official repository for **"Leakage-Aware, Source-Grouped Benchmarking of Multi-Class Clinical Skin-Disease Instance Segmentation Across Three Datasets: Comparative Evaluation of Modern Detectors and SAM2-Assisted Mask Refinement"**.

Authors: **N. M. Muntasirul Haque**, **A F H Dhrubo**, and **Mohsin Sajjad**

---

## 📌 Overview

This repository provides the complete dataset, benchmark evaluation code, and reproducible training notebooks accompanying our research paper. The work addresses critical methodological gaps in clinical dermatology computer vision:
1. **Source-level & Patient-level Data Leakage**: Standard random splits inflate test metrics due to co-occurrence of multiple lesions from the same patient across train/val/test splits. We benchmark patient-grouped vs. random splits.
2. **Comprehensive Instance Segmentation Benchmark**: Systematic head-to-head evaluation across YOLO family models (YOLO11, YOLO12, YOLO26, YOLOv9), two-stage / query-based architectures (Cascade Mask R-CNN, Mask2Former, CondInst, PointRend, RTMDet-Ins), and semantic baselines (U-Net).
3. **SAM 2 Foundation Model Refinement**: Cascaded detector-to-SAM 2 prompting pipeline to refine coarse bounding box / polygon predictions into precise boundary segmentations.

---

## 📂 Repository Structure

```text
skin-disease-segmentation/
├── dataset/
│   └── Image_Annotation_rebuilt/       # 4-class clinical instance segmentation dataset
│       ├── data.yaml                   # YOLO dataset configuration
│       ├── train/
│       │   ├── images/                 # 6,057 training images
│       │   └── labels/                 # YOLO polygon segmentations
│       ├── valid/
│       │   ├── images/                 # 1,085 validation images
│       │   └── labels/
│       └── test/
│           ├── images/                 # 1,030 test images
│           └── labels/
├── trained_notebooks/
│   ├── image_annotation_rebuild/       # Rebuilt dataset benchmark sessions
│   │   ├── session-18-yolo11l-seg.ipynb
│   │   ├── session-19-yolo11x-seg-5-fold-cv.ipynb
│   │   ├── session-20-rtmdet-ins-l.ipynb
│   │   ├── session-21-mask2former.ipynb
│   │   ├── session-22-yolov9e-seg.ipynb
│   │   ├── session-23-cascade-mask-r-cnn.ipynb
│   │   ├── session-25-sam2 fold 1:5 epoch 20.ipynb
│   │   ├── session-25-sam2 fold 5:5 6 epoch.ipynb
│   │   ├── session-26-real-pointrend-r101.ipynb
│   │   ├── session-27-mask-r-cnn.ipynb
│   │   ├── session-28-v3-condinst-r50-fpn.ipynb
│   │   ├── session-29-pretrained-resnet-34-u-net.ipynb
│   │   ├── session-34-yolo26n-seg.ipynb
│   │   ├── session-35-yolo12n-seg.ipynb
│   │   ├── session-36-detector-sam2-refinement.ipynb
│   │   ├── session-39-matched-leakage-experiment.ipynb
│   │   └── session-40-matched-leakage-experiment.ipynb
│   ├── skin_disease/                   # Roboflow Universe 19-class benchmark sessions
│   │   ├── skin-disease-session-18-yolo11l-seg.ipynb
│   │   └── skin-disease-session-19-yolo11x-seg.ipynb
│   └── skindisnet/                     # SkinDisNet Mendeley source sessions
│       ├── skindisnet-session-28-v3-condinst-r50-fpn.ipynb
│       └── skindisnet-session-41-yolo26n-seg-patient-grouped.ipynb
├── .gitignore
└── README.md
```

---

## 🔬 Dataset Details (`Image_Annotation_rebuilt`)

- **Total Images**: 8,172 clinical images
- **Classes (4 Target Conditions)**:
  - `0: Nail_psoriasis`
  - `1: SJS-TEN` (Stevens–Johnson syndrome / toxic epidermal necrolysis)
  - `2: acne`
  - `3: vitiligo`
- **Annotation Format**: Standard YOLO instance segmentation polygon coordinates normalised to $[0, 1]$.
- **Data Split Breakdown**:
  - **Train**: 6,057 images (74.1%)
  - **Validation**: 1,085 images (13.3%)
  - **Test**: 1,030 images (12.6%)

---

## 🚀 Quick Start

### 1. Requirements

```bash
git clone https://github.com/muntasirul/skin-disease-segmentation.git
cd skin-disease-segmentation
pip install ultralytics torch torchvision opencv-python matplotlib
```

### 2. Training with Ultralytics YOLO

To train a YOLO11 instance segmentation model on the dataset:

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11l-seg.pt")

# Train on dataset
results = model.train(
    data="dataset/Image_Annotation_rebuilt/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

Or via CLI:
```bash
yolo segment train data=dataset/Image_Annotation_rebuilt/data.yaml model=yolo11l-seg.pt epochs=100 imgsz=640
```

### 3. Running Benchmark Notebooks

All Jupyter notebooks are fully self-contained and pre-configured for Kaggle / Google Colab / local multi-GPU environments.
- Open any notebook inside `trained_notebooks/image_annotation_rebuild/` to reproduce specific model runs.
- `session-36-detector-sam2-refinement.ipynb` demonstrates the zero-shot & fine-tuned SAM 2 cascaded mask refinement pipeline.
- `session-39` & `session-40` reproduce the matched-leakage vs. leakage-free patient grouping comparative study.

---

## 📜 Citation

If you use this dataset, benchmark, or notebooks in your research, please cite our paper:

```bibtex
@article{haque2026leakage,
  title={Leakage-Aware, Source-Grouped Benchmarking of Multi-Class Clinical Skin-Disease Instance Segmentation Across Three Datasets: Comparative Evaluation of Modern Detectors and SAM2-Assisted Mask Refinement},
  author={Haque, N. M. Muntasirul and Dhrubo, A F H and Sajjad, Mohsin},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This repository is distributed under the MIT License. See [LICENSE](LICENSE) for details.
