# Edge Fluency Classifier

A lightweight, edge-ready pipeline for scoring pronunciation and fluency from short speech clips. It supports training compact MLP models, distilling them for efficiency, exporting to ONNX, and profiling on mobile devices.

## Quick Start

### 1. Setup
Requires Python 3.11.
```bash
# Using your own conda env
make setup
```

### 2. Data Preparation
Requires a Hugging Face account.
```bash
huggingface-cli login
make download
make features
```

### 3. Training & Evaluation
```bash
# Train Teacher & Student Models
make train_teacher
make train_student

# Evaluate all models (including SNR sweeps)
make evaluate_all
```

### 4. Edge Deployment
Quantize to INT8, export to ONNX, and profile.
```bash
make quantize
make export_onnx
make profile          # Host profiling
make android_profile  # Android profiling (requires ADB)
```

## Key Results

| Model | Accuracy | Macro F1 | Size (MB) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher (MLP)** | ~85% | ~0.82 | 3.8 | ~15 |
| **Student (Distilled)** | ~84% | ~0.81 | 0.9 | ~8 |
| **PLDA** | ~78% | ~0.75 | 0.1 | <1 |

*Latency measured on Pixel Tablet (CPU).*

## License
MIT License. Please respect the Speechocean 762 dataset license.
