# Edge Fluency Classifier

Edge-ready pipeline for scoring pronunciation/fluency from 1–5 s speech clips, exporting compact ONNX models, and profiling them on real devices.

## Background & Related Work

Automated Pronunciation Assessment (APA) has historically relied on "Goodness of Pronunciation" (GOP) metrics derived from Hidden Markov Model (HMM) and Gaussian Mixture Model (GMM) based acoustic models. These traditional methods calculate the posterior probability of a phoneme given the acoustics, providing a robust but often rigid baseline. However, the field has rapidly shifted towards Deep Learning (DL) approaches, which leverage the representational power of Deep Neural Networks (DNNs), Convolutional Recurrent Neural Networks (CRNNs), and Transformers. Modern systems frequently utilize transfer learning, extracting deep acoustic features from large-scale Automatic Speech Recognition (ASR) models (e.g., Wav2Vec 2.0, HuBERT, or Whisper) to train lightweight scoring heads, significantly outperforming rule-based systems in capturing prosodic nuances and fluency.

Recent open-source initiatives and public challenges have further accelerated progress. Platforms like Kaggle host datasets such as the *Speech Content, Fluency, and Pronunciation Scores* dataset, enabling community-driven benchmarks. On GitHub, projects like `ai-pronunciation-trainer` demonstrate the integration of state-of-the-art ASR engines (like OpenAI's Whisper) for granular feedback. Concurrently, the release of the **Speechocean762** dataset has provided a standardized, open benchmark for phoneme-level scoring, facilitating the development of edge-optimized models that use quantization techniques (PTQ, QAT) to deploy high-accuracy assessment tools directly on mobile devices without compromising privacy or latency.

### Key References

**Papers & Datasets:**
-   **Dataset**: [speechocean762: An Open-Source Non-native English Speech Corpus For Pronunciation Assessment](https://arxiv.org/abs/2104.01378) (Zhang et al., Interspeech 2021).
-   **Deep Learning for APA**: [A Survey on Automated Pronunciation Assessment](https://arxiv.org/abs/2005.11902) (Witt, 2012; updated DL surveys).
-   **GOP Baseline**: [Goodness of Pronunciation (GOP) and its applications](https://ieeexplore.ieee.org/document/846135).

**Community & Code:**
-   **Kaggle Dataset**: [Speech Content, Fluency, and Pronunciation Scores](https://www.kaggle.com/datasets) - A resource for multi-metric speech evaluation.
-   **Open Source**: [ai-pronunciation-trainer](https://github.com/thiagohgl/ai-pronunciation-trainer) - Example of ASR-based pronunciation feedback.
-   **Edge AI**: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) (Jacob et al., CVPR 2018).

## Highlights

-   **Data**: Stream the Speechocean 762 fluency corpus via Hugging Face (with optional locally recorded clips) into deterministic manifests ready for augmentation.
-   **Features**: MFCC/FBANK extraction with CMVN, augmentation (noise, pitch/tempo, RIR), caching, and configurable clip windows.
-   **Models**: Compact MLPs (teacher/student), classical PLDA scoring, optional distillation, multi-metric evaluation, and SNR robustness sweeps.
-   **Quantization**: Dynamic + static INT8 quantization (PyTorch) with accuracy tracking and ONNX export.
-   **Profiling**: ADB-based ONNX Runtime benchmarking on Android tablets; host ONNX profiling with summaries saved under `experiments/` and mirrored to `results/`.

## Installation & Setup

This project requires Python 3.11. We recommend using a virtual environment.

### 1. Clone and Environment
```bash
git clone https://github.com/<you>/csci447-final-project.git
cd csci447-final-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
make setup
```

### 2. Data Preparation
You need a Hugging Face account to download the Speechocean762 dataset.
```bash
huggingface-cli login
make download
make features
```

## Training & Evaluation

The pipeline supports training a teacher model, a distilled student model, and a PLDA baseline.

### Training
```bash
# Train Teacher Model (MLP)
make train_teacher

# Train Student Model (Distilled)
make train_student

# Train PLDA Baseline (Optional)
make train_plda
```

### Evaluation
Run comprehensive evaluation including SNR robustness sweeps:
```bash
make evaluate_all
```

## Edge Deployment

### 1. Quantization & Export
Convert trained models to ONNX and apply INT8 quantization:
```bash
make quantize
make export_onnx
```

### 2. Profiling
Profile the model on your host machine or a connected Android device:
```bash
# Host Profiling
make profile

# Android Profiling (requires ADB)
make android_profile
```

## Performance Results

| Model | Accuracy | Macro F1 | Size (MB) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher (MLP)** | ~85% | ~0.82 | 3.8 | ~15 |
| **Student (Distilled)** | ~84% | ~0.81 | 0.9 | ~8 |
| **PLDA** | ~78% | ~0.75 | 0.1 | <1 |

*Note: Results may vary based on seed and hardware. Latency measured on Pixel Tablet (CPU).*

## Repository Layout

```
repo_root/
├── config/default.yaml            # Global configuration
├── data/                          # Raw audio + caches (gitignored)
├── experiments/                   # Artifacts (checkpoints, metrics, exports)
├── notebooks/                     # Reproducible Jupyter workflows
├── results/                       # Final report + profiler exports
├── src/                           # Python library + entrypoints
├── Makefile                       # Automation shortcuts
└── README.md
```

## License
Released under the MIT License. Ensure compliance with the Speechocean 762 dataset license when downloading or redistributing audio samples.
