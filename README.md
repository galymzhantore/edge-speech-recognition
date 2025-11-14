# Edge Fluency Classifier

Edge-ready pipeline for scoring pronunciation/fluency from 1–5 s speech clips, exporting compact ONNX models, and profiling them on real devices.

## Highlights

- **Data**: Stream the Speechocean 762 fluency corpus via Hugging Face (with optional locally recorded clips) into deterministic manifests ready for augmentation.
- **Features**: MFCC/FBANK extraction with CMVN, augmentation (noise, pitch/tempo, RIR), caching, and configurable clip windows.
- **Models**: Compact MLPs (teacher/student), classical PLDA scoring, optional distillation, multi-metric evaluation, and SNR robustness sweeps.
- **Quantization**: Dynamic + static INT8 quantization (PyTorch) with accuracy tracking and ONNX export.
- **Profiling**: ADB-based ONNX Runtime benchmarking on Android tablets; host ONNX profiling with summaries saved under `experiments/` and mirrored to `results/`.

## End-to-End Workflow (Deterministic)

```bash
python -m venv .venv && source .venv/bin/activate
make setup

# 1) Data + features
make download
make features

# 2) Training
make train_teacher
make train_student
make train_plda           # optional classical baseline
make evaluate_all

# 3) Edge readiness
make quantize
make export_onnx

# 4) Profiling
make profile
# Optional device run via ADB helper / notebook
# make android_profile DEVICE="android_tablet"

# 5) Reporting
make report
```

Core parameters (override via env or CLI): `DEVICE/TARGET_DEVICE`, `CLIP_SECONDS`, `LABEL_SCHEMA`, `DATA_DIR`, `OUTPUT_DIR`.

## Reproducible Environment

All commands assume macOS/Linux with Python 3.11. Windows users can run inside WSL.

1. **Clone + auth**
   ```bash
   git clone https://github.com/<you>/csci447-final-project.git
   cd csci447-final-project
   huggingface-cli login  # required for Speechocean 762
   ```
2. **Python environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   make setup  # installs project in editable mode
   ```
3. **Seed/config control** – every stage reads from `config/default.yaml`; override via `CONFIG=path/to/custom.yaml make ...` to keep experiments versioned.

Generated artifacts (ONNX exports, profiler logs) stay under `experiments/` and are gitignored, so reruns do not pollute history.

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

## Make Targets (Automation)

| Target | Description |
| ------ | ----------- |
| `make download` | Stream Speechocean 762 splits, merge optional local recordings, stratify manifests. |
| `make features` | Run preprocessing, augmentations, MFCC/FBANK extraction, and CMVN stats. |
| `make train_teacher` / `make train_student` | Train teacher MLP and distilled student. |
| `make train_plda` | Fit a classical PLDA baseline and persist as `plda.joblib`. |
| `make evaluate_all` | Compute per-split metrics + SNR sweeps with figures under `experiments/`. |
| `make quantize` | Run dynamic/static INT8 quantization + pruning with accuracy tracking. |
| `make export_onnx` | Produce ONNX and label maps in `experiments/exports/`. |
| `make profile` | Host profiling (ONNX + energy proxy). |
| `make android_profile` | Pull profiler metrics (host + Android via `adb`, optional). |
| `make report` | Summarize metrics/latency/size into `results.md` and `results/report.md`. |

## Profiling & Deployment

### Host vs Android profiling

- `make profile` uses ONNX Runtime to measure latency/memory locally. Inputs are auto-shaped to match the exported ONNX using `experiments/exports/summary.json`.
- `scripts/adb_benchmark_onnx.sh` pushes the ONNX model plus ONNX Runtime perf binary to a connected tablet, runs the benchmark (`-I` auto-generated inputs) and falls back to generating `test_data_set_0` if needed.
- A reproducible Jupyter workflow lives in `notebooks/device_profiling_experiments.ipynb`. It automatically:
  1. Switches to the repo root even if launched elsewhere.
  2. Ensures `adb` is on `PATH` (checks `ANDROID_SDK_ROOT`, etc.).
  3. Runs export → host profile → Android profile via the helper script.
  4. Parses `experiments/profiles/adb_ort_perf.txt`, plots host vs Android latency, and stores a combined JSON summary under `results/`.

To run the perf test manually without the notebook, use `scripts/adb_benchmark_onnx.sh` (auto-pushes model + libs, handles `test_data_set_0` fallback).

## Data Notes

- Dataset: `mispeech/speechocean762` (score buckets map to `poor`, `moderate`, `good`). Authenticate with `huggingface-cli login` before `make download`.
- Custom label schemas: pass `LABEL_SCHEMA='{"bad":0,"ok":1,"great":2}'` or edit `config/default.yaml`.
- Local bilingual snippets: run `python -m src.dataset.local_recorder --output-dir data/local --label moderate` and re-run `make download`.

## Results & Reporting

- Quantization summary (`experiments/quantized/summary.json`) tracks size + accuracy deltas for FP32, dynamic, static, and pruned models.
- Host and ADB-based ONNX profiling writes CSV/JSON to `experiments/profiles/` and mirrors summaries to `results/profile_summary.json`.
- `results/jupyter_profile_summary.json` is emitted by the reproducibility notebook and includes the raw export summary + parsed benchmarking metrics.
- `make report` stitches everything into `results.md` + `results/report.md`, including INT8 accuracy drop (goal <5 %), latency (<100 ms/clip), and model size (<10 MB).

## License & Attribution

Released under the MIT License. Ensure compliance with the Speechocean 762 dataset license when downloading or redistributing audio samples.
