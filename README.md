# Multimodal Seizure Detection Pipeline

An end-to-end seizure detection pipeline using EEG, ECG, EMG, and accelerometry. 1D convolutional encoders are pretrained via autoencoders, then fused through three architectures (baseline concat, soft-attention, deep attention) with grid search. Best model (Attention Fusion CNN) is interpreted via gradients, occlusion, and post-hoc fMRI.

---

## Table of contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Pipeline walkthrough](#pipeline-walkthrough)
- [Models](#models)
- [Grid search & training](#grid-search--training)
- [Explainability](#explainability)
- [Results](#results)
- [Limitations](#limitations)

---

## Overview

The notebook builds a clean, reproducible seizure classification pipeline from raw BIDS recordings to cross-modal interpretation. All stages live in a single notebook: data loading, windowing, normalization, encoder pretraining, model training with grid search, evaluation, and XAI.

---

## Datasets

| Dataset | Use | Subjects |
|---|---|---|
| `ds005873` | Wearable physiology (EEG, ECG, EMG, MOV) | sub-001, sub-002 |
| `ds007313` | Post-hoc fMRI interpretation only | sub-A006 |

> **Note:** The fMRI subject is from a different dataset than the wearable seizure subjects. Cross-modal comparison is illustrative, not subject-matched validation.

---

## Repository structure

```
.
├── multimodal_seizure_pipeline.ipynb   # Main notebook (all stages)
├── ds005873/                           # Wearable BIDS data
│   └── sub-001/, sub-002/
└── ds007313/                           # fMRI BIDS derivatives
    └── derivatives/denoising/...
```

---

## Setup

```bash
pip install mne nibabel nilearn torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

Tested on Python 3.12 · PyTorch 2.x · CUDA (falls back to CPU automatically).

---

## Pipeline walkthrough

### 1. Data loading & preprocessing

- BIDS annotation parsing with `mne` and custom JSON readers
- Modalities: EEG (2 ch), ECG (1 ch), EMG (1 ch), MOV accelerometer (3 ch) — all at 256 Hz
- Seizure-centered windows: ±120 s padding, 10 s window, 10 s stride
- Labels: positive only if the window overlaps a real seizure annotation; `impd` intervals excluded
- Stratified train / val / test split: 75 / 15 / 25 %
- Z-score normalization fitted on the training set only

### 2. Encoder pretraining

Each modality encoder is pretrained independently as a `SignalAutoencoder1D` (MSELoss) before being loaded into the fusion models.

| Family | Blocks | Channels | Kernels |
|---|---|---|---|
| Shallow | 2 | EEG 2→32→64, others 1(3)→16→32 | k7, k5 |
| Deep | 3 | EEG 2→32→64→96, others 1(3)→16→32→48 | k7, k5, k3 |

All blocks use BatchNorm + ReLU + MaxPool + Dropout(0.35). Pretraining: Adam lr=1e-3, WD=1e-5, 4 epochs, batch 64.

### 3. Models

#### Baseline Fusion CNN

Four shallow encoders → `torch.cat` (dim 256) → FC(128) + BN + ReLU + Dropout → FC(1).

#### Attention Fusion CNN ★ best

Four shallow encoders → soft attention (FC 64→32 Tanh → FC 1 → softmax) → weighted concat (256) → FC(128) + BN + ReLU + Dropout → FC(1).

#### Deep Attention Fusion CNN

Four deep encoders → deeper attention (FC 64→48 ReLU Drop → FC 1 → softmax) → weighted concat (256) → FC(192) + BN + ReLU + Drop → FC(96) + ReLU → FC(1).

All models output a single logit; loss is `BCEWithLogitsLoss` with class-weighted `pos_weight`.

---

## Grid search & training

```
dropout        : [0.25, 0.35]
lr             : [1e-3, 5e-4]
weight_decay   : [1e-4]
freeze_first_block : [False, True]  # pretrained models only
classifier_hidden  : [128, 192]     # 192 for Deep Attention only
```

- Baseline CNN / Attention CNN: 4 trials each
- Pretrained Attention CNN: 8 trials
- Pretrained Deep Attention CNN: 8 trials
- Early stopping: patience = 4, min_delta = 1e-3, criterion = val-F1
- Best checkpoint restored at end of training

---

## Explainability

### Wearable signal XAI

| Method | What it gives |
|---|---|
| Input gradients | Dominant EEG channel and most important 10 s time segment |
| Integrated gradients | 24-step Riemann attribution, IG-confirmed dominant channel |
| Attention weights | Softmax modality scores from the attention head |
| Modality occlusion | Zero-out each modality, measure Δp to find dominant modality |

### fMRI post-hoc XAI (ds007313, illustrative only)

- Mean activation map + peak coordinate in MNI mm
- Temporal variability map (std across 265 volumes)
- Top voxel ranking tables (mean + std)

---

## Results

| Model | Best val-F1 | Notes |
|---|---|---|
| Baseline CNN | — | Reference baseline |
| Attention Fusion CNN | **0.727** | Best overall · selected for XAI |
| Pretrained Attention CNN | — | Shallow pretrained bank |
| Pretrained Deep Attention CNN | — | Deep pretrained bank |

**Selected test window:** seizure probability = 0.988, event `sz_foc_ia_m_hyperkinetic`, localization `cen_par`, lateralization right.

**Dominant EEG channel (input gradients + IG):** `CROSStop SD` → cross-head temporo-parietal trajectory.

**Dominant modality (attention):** EMG. **Largest ablation drop:** EEG.

---


