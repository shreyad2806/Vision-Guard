# VisionGuard Training Pipeline

## Overview

This directory contains the ML training pipeline for the VisionGuard image quality
assessment system. The pipeline trains a classical ML regressor that predicts a
0–100 quality score from hand-crafted OpenCV features.

## Pipeline Architecture

```
Image → OpenCV Feature Extraction → Feature Vector → StandardScaler → ML Regressor → Quality Score
```

The **same** feature extractor is used during training and backend inference.

## Datasets

### KADID-10K

- 81 distorted images per reference, 80 reference images
- 5 distortion types × 5 levels × ~3.2 quality ratings
- DMOS scores: 1.0 (worst) to ~5.0 (best)
- Normalized to 0–100: `(dmos - 1) / 4 * 100`

### KonIQ-10K (optional, if available)

- 10,125 images with MOS quality ratings
- Expected at `ml/data/koniq-10k/`

## Usage

### 1. Inspect datasets

```bash
cd ml/training
python inspect_datasets.py
```

### 2. Train model (fast mode — ~3000 samples)

```bash
python train_model.py --mode fast --max-samples-fast 3000
```

### 3. Train model (full mode — all samples)

```bash
python train_model.py --mode full
```

### 4. Evaluate trained model

```bash
python evaluate_model.py
```

## Feature Extractors

The following features are extracted per image:

| Feature | Method | Description |
|---------|--------|-------------|
| sharpness | Laplacian variance | Edge detail quality |
| brightness | Mean grayscale | Overall exposure |
| contrast | Grayscale std | Dynamic range |
| noise_estimate | MAD of Laplacian | Noise level |
| entropy | Histogram entropy | Information content |
| saturation | Mean HSV saturation | Color intensity |

Additional features may be added during training but must also be available
during backend inference.

## Output Artifacts

Saved to `backend/models/`:

- `quality_model.joblib` — trained regressor
- `feature_scaler.joblib` — StandardScaler
- `model_metadata.json` — model info, feature names, metrics

## Model Integration

The trained model is automatically loaded by `backend/apps/ml/model_loader.py`
when the FastAPI server starts. No retraining happens at runtime.

## Fast vs Full Mode

| Mode | Samples | Time (approx) | Purpose |
|------|---------|---------------|---------|
| fast | ~3000 | 2-5 min | Development, iteration |
| full | ~8100 | 5-15 min | Production quality |
