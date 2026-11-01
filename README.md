# Leakage-Aware, Source-Grouped Benchmarking of Multi-Class Clinical Skin-Disease Instance Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLO11%20%7C%20YOLO12%20%7C%20YOLO26-blue)](https://github.com/ultralytics/ultralytics)
[![SAM 2](https://img.shields.io/badge/Meta-SAM%202.1-green)](https://github.com/facebookresearch/sam2)

Official repository for **"Leakage-Aware, Source-Grouped Benchmarking of Multi-Class Clinical Skin-Disease Instance Segmentation Across Three Datasets: Comparative Evaluation of Modern Detectors and SAM2-Assisted Mask Refinement"**.

Authors: **N. M. Muntasirul Haque**, **A F H Dhrubo**, and **Mohsin Sajjad**

---

## 📌 Overview

This repository provides the complete dataset collection, benchmark evaluation code, and reproducible training notebooks accompanying our research paper. The work addresses critical methodological gaps in clinical dermatology computer vision:
1. **Source-level & Patient-level Data Leakage**: Standard random splits inflate test metrics due to co-occurrence of multiple lesions/augmentation variants from the same patient across train/val/test splits. We benchmark patient-grouped vs. random splits.
2. **Comprehensive Instance Segmentation Benchmark**: Systematic head-to-head evaluation across YOLO family models (YOLO11, YOLO12, YOLO26, YOLOv9), two-stage / query-based architectures (Cascade Mask R-CNN, Mask2Former, CondInst, PointRend, RTMDet-Ins, SOLOv2, YOLACT), and semantic baselines (U-Net).
3. **SAM 2 Foundation Model Refinement**: Cascaded detector-to-SAM 2 prompting pipeline to refine coarse bounding box / polygon predictions into precise boundary segmentations.

---

## 📂 Repository Structure

```text
skin-disease-segmentation/
├── dataset/
│   ├── Image_Annotation_rebuilt/       # 4-class clinical instance segmentation dataset
│   │   ├── data.yaml                   # YOLO dataset configuration
│   │   ├── train/                      # 6,057 training images + labels
│   │   ├── valid/                      # 1,085 validation images + labels
│   │   └── test/                       # 1,030 test images + labels
│   ├── SkinDisNet_2_Annotation/        # 6-class SkinDisNet-2 instance segmentation dataset
│   │   ├── data.yaml                   # YOLO dataset configuration
│   │   ├── manifest.csv                # Source lineage & split tracking (6,000 entries)
│   │   ├── summary.json                # Dataset evaluation notes & provenance
│   │   ├── coco/                       # COCO format annotations (train/valid/test.json)
│   │   ├── train/                      # 4,000 images + polygon labels
│   │   ├── valid/                      # 1,000 images + polygon labels
│   │   └── test/                       # 1,000 images + polygon labels (locked external test)
│   └── skin_disease_yolo/              # 19-class Roboflow clinical skin disease dataset
│       ├── data.yaml                   # YOLO dataset configuration
│       ├── train/                      # 2,690 images + polygon labels
│       ├── valid/                      # 658 images + polygon labels
│       └── test/                       # 364 images + polygon labels
├── trained_notebooks/
│   ├── image_annotation_rebuild/       # Rebuilt dataset benchmark sessions (Sessions 18-40)
│   ├── skin_disease/                   # 19-class benchmark sessions (Sessions 18-35)
│   └── skindisnet/                     # SkinDisNet benchmark sessions (Sessions 19-41)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔬 Datasets

### 1. `Image_Annotation_rebuilt` (Primary Rebuilt Dataset)
- **Total Images**: 8,172 clinical images
- **Classes (4 Target Conditions)**:
  - `0: Nail_psoriasis`
  - `1: SJS-TEN` (Stevens–Johnson syndrome / toxic epidermal necrolysis)
  - `2: acne`
  - `3: vitiligo`
- **Splits**: Train (6,057 images, 74.1%), Validation (1,085 images, 13.3%), Test (1,030 images, 12.6%)
- **Format**: YOLO polygon segmentation coordinates normalized to $[0, 1]$.

### 2. `SkinDisNet_2_Annotation` (SkinDisNet Mendeley Collection)
- **Total Images**: 6,000 images
- **Classes (6 Target Conditions)**:
  - `0: Atopic Dermatitis`
  - `1: Contact Dermatitis`
  - `2: Eczema`
  - `3: Scabies`
  - `4: Seborrheic Dermatitis`
  - `5: Tinea Corporis`
- **Splits**: Train (4,000 images), Validation (1,000 images), Test (1,000 images, locked external test set)
- **Formats**: Both YOLO polygon labels and COCO format (`coco/train.json`, `coco/valid.json`, `coco/test.json`).
- **Lineage**: Includes `manifest.csv` mapping every image to its original source identifier for leak-free source-grouped cross-validation.

### 3. `skin_disease_yolo` (Roboflow Universe Multi-Disease Dataset)
- **Total Images**: 3,712 images
- **Classes (19 Conditions)**: `acne`, `Darier_s-Disease`, `Epidermolysis-Bullosa-Pruriginosa`, `Hailey-Hailey-Disease`, `Hemangiome`, `Impetigo`, `Leishmanios`, `Lupus-Erythematosus-Chronicus-Discoides`, `Melanoma`, `Molluscum-Contagiosum`, `Porokeratosis`, `Psoriasis`, `Tinea-Corporis`, `Tungiasis`, `basal-cell-carcinoma`, `eczema`, `lichen`, `nevus`, `normal skin`
- **Splits**: Train (2,690 images), Validation (658 images), Test (364 images).

---

## 🚀 Quick Start

### 1. Requirements

```bash
git clone https://github.com/muntasirul/skin-disease-segmentation.git
cd skin-disease-segmentation
pip install ultralytics torch torchvision opencv-python matplotlib
```

### 2. Training with Ultralytics YOLO

To train a YOLO11/YOLO12/YOLO26 instance segmentation model on any of the datasets:

```python
from ultralytics import YOLO

# Load model checkpoint
model = YOLO("yolo26n-seg.pt")

# Train on SkinDisNet_2_Annotation
results = model.train(
    data="dataset/SkinDisNet_2_Annotation/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

Or via CLI:

```bash
# Image_Annotation_rebuilt
yolo segment train data=dataset/Image_Annotation_rebuilt/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

# SkinDisNet_2_Annotation
yolo segment train data=dataset/SkinDisNet_2_Annotation/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640

# skin_disease_yolo
yolo segment train data=dataset/skin_disease_yolo/data.yaml model=yolo26n-seg.pt epochs=100 imgsz=640
```

### 3. Running Benchmark Notebooks

All Jupyter notebooks are fully self-contained and pre-configured for Kaggle / Google Colab / local multi-GPU environments:
- **`trained_notebooks/image_annotation_rebuild/`**: Sessions 18–40 on the rebuilt dataset.
- **`trained_notebooks/skin_disease/`**: Sessions 18–35 on the 19-class dataset.
- **`trained_notebooks/skindisnet/`**: Sessions 19–41 on the SkinDisNet dataset, including leak-free patient-grouped and source-grouped 5-fold cross-validation.

---

## 📜 Citation

If you use these datasets, benchmarks, or notebooks in your research, please cite our paper:

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

This repository is distributed under the MIT License. See [LICENSE](https://github.com/muntasirul/skin-disease-segmentation/blob/main/LICENSE) for details.
