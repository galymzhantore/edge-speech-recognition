#!/usr/bin/env bash
set -euo pipefail

# Run ONNX Runtime perf test on a connected Android device (arm64-v8a).
#
# Usage:
#   ONNX=experiments/exports/model.onnx \
#   ORT_PERF_HOST=/path/to/onnxruntime_perf_test \
#   RUNS=50 SERIAL=ABC123 THREADS=4 EXTRA="" \
#   scripts/adb_benchmark_onnx.sh
#
# Notes:
# - The tool uses random inputs if none provided. For dynamic input shapes with
#   an unknown batch dim, it typically defaults to batch=1. If needed, pass
#   additional flags via EXTRA to specify input shapes.
# - Output is saved to experiments/profiles/adb_ort_perf.txt

ONNX=${ONNX:-experiments/exports/model.onnx}
ORT_PERF_HOST=${ORT_PERF_HOST:-}
# Optional: directory containing libonnxruntime*.so (and providers). If set, libs will be pushed and LD_LIBRARY_PATH used.
ORT_LIB_DIR=${ORT_LIB_DIR:-}
RUNS=${RUNS:-50}
SERIAL=${SERIAL:-}
THREADS=${THREADS:-4}
EXTRA=${EXTRA:-}
INPUT_NAME=${INPUT_NAME:-input}
SUMMARY_PATH_ENV=${SUMMARY_PATH:-}

if [[ -z "${ORT_PERF_HOST}" ]]; then
  # Try common locations in a local checkout
  for p in \
    "onnxruntime/build/android-arm64/Release/onnxruntime_perf_test" \
    "onnxruntime/build/*/Release/onnxruntime_perf_test" \
    "onnxruntime/build/Release/onnxruntime_perf_test"; do
    for cand in $p; do
      if [[ -f "$cand" ]]; then ORT_PERF_HOST="$cand"; break; fi
    done
    [[ -n "${ORT_PERF_HOST}" ]] && break
  done
  # Fallback to find if still not set
  if [[ -z "${ORT_PERF_HOST}" ]]; then
    ORT_PERF_HOST=$(find onnxruntime -name onnxruntime_perf_test -type f 2>/dev/null | head -n 1 || true)
  fi
fi

if [[ -z "${ORT_PERF_HOST}" || ! -f "${ORT_PERF_HOST}" ]]; then
  echo "ERROR: Could not locate 'onnxruntime_perf_test'." >&2
  echo "- Build it under ./onnxruntime (Android arm64):" >&2
  echo "    python tools/ci_build/build.py --build_dir build/android-arm64 --config Release \\" >&2
  echo "      --android --android_abi=arm64-v8a --android_api=26 \\" >&2
  echo "      --android_ndk_path \"$ANDROID_NDK_HOME\" --android_sdk_path \"$ANDROID_SDK_ROOT\" --parallel --skip_tests" >&2
  echo "- Or set ORT_PERF_HOST to the binary path and rerun." >&2
  exit 1
fi

# Default ORT_LIB_DIR to the perf binary directory if not provided
if [[ -z "${ORT_LIB_DIR}" ]]; then
  ORT_LIB_DIR=$(cd "$(dirname "${ORT_PERF_HOST}")" && pwd)
fi

if [[ ! -f "${ONNX}" ]]; then
  echo "ERROR: ONNX model not found at ${ONNX}. Run 'make export_onnx' first." >&2
  exit 1
fi

ADB=(adb)
if [[ -n "${SERIAL}" ]]; then
  ADB+=( -s "${SERIAL}" )
fi

echo "[adb] Pushing model and perf binary..."
"${ADB[@]}" push "${ONNX}" /data/local/tmp/model.onnx >/dev/null
if [[ -f "${ONNX}.data" ]]; then
  echo "[adb] Detected external data file; pushing ${ONNX}.data ..."
  "${ADB[@]}" push "${ONNX}.data" /data/local/tmp/model.onnx.data >/dev/null
fi
"${ADB[@]}" push "${ORT_PERF_HOST}" /data/local/tmp/onnxruntime_perf_test >/dev/null
"${ADB[@]}" shell chmod +x /data/local/tmp/onnxruntime_perf_test

# Optionally push shared libraries and set LD_LIBRARY_PATH
LIB_EXPORT=""
if [[ -n "${ORT_LIB_DIR}" ]]; then
  echo "[adb] Pushing ONNX Runtime shared libraries from ${ORT_LIB_DIR}..."
  "${ADB[@]}" push "${ORT_LIB_DIR}"/*.so /data/local/tmp/ >/dev/null || true
  LIB_EXPORT="LD_LIBRARY_PATH=/data/local/tmp"
fi

# Threads: intra-op threads env var (many builds respect these envs)
ENV_PREFIX=("OMP_NUM_THREADS=${THREADS}" "OMP_WAIT_POLICY=PASSIVE" "OPENBLAS_NUM_THREADS=${THREADS}")

## Note: Some perf_test builds do not support --input_shape(s). We rely on -I to auto-generate inputs.

mkdir -p experiments/profiles
LOG=experiments/profiles/adb_ort_perf.txt

run_perf() {
  local model_path=$1
  local flags=$2
  local cmd="${LIB_EXPORT} ${ENV_PREFIX[*]} /data/local/tmp/onnxruntime_perf_test -e cpu -m times -r ${RUNS} -x ${THREADS} -s ${flags} ${model_path}"
  "${ADB[@]}" shell "${cmd}"
}

echo "[adb] Running onnxruntime_perf_test for ${RUNS} runs (auto-generated inputs)..."
set +e
run_perf "/data/local/tmp/model.onnx" "-I ${EXTRA:-}" | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e

if [[ ${rc} -ne 0 ]] || grep -E -q "no test input data|failed to initialize|Unknown command line flag '-I'|there is no test data" "${LOG}" 2>/dev/null; then
  echo "[warn] Auto-generated inputs failed. Falling back to test_data_set_0..."
  TMP_INPUT_PB=$(mktemp /tmp/input_XXXX.pb)
  python - <<'PY'
import json, os
from onnx import numpy_helper
import numpy as np
data = json.load(open('experiments/exports/summary.json'))
freq, frames = int(data['freq']), int(data['frames'])
arr = np.random.rand(1, freq, frames).astype(np.float32)
t = numpy_helper.from_array(arr, name='input')
open(os.environ['TMP_INPUT_PB'], 'wb').write(t.SerializeToString())
PY
  MODEL_DIR_ON_DEVICE="/data/local/tmp/model_dir"
  "${ADB[@]}" shell "rm -rf ${MODEL_DIR_ON_DEVICE} && mkdir -p ${MODEL_DIR_ON_DEVICE}/test_data_set_0"
  "${ADB[@]}" push "${ONNX}" ${MODEL_DIR_ON_DEVICE}/model.onnx >/dev/null
  if [[ -f "${ONNX}.data" ]]; then
    "${ADB[@]}" push "${ONNX}.data" ${MODEL_DIR_ON_DEVICE}/model.onnx.data >/dev/null
  fi
  "${ADB[@]}" push "${TMP_INPUT_PB}" ${MODEL_DIR_ON_DEVICE}/test_data_set_0/input_0.pb >/dev/null
  rm -f "${TMP_INPUT_PB}"
  echo "[adb] Re-running using test_data_set_0..."
  set +e
  run_perf "${MODEL_DIR_ON_DEVICE}/model.onnx" "${EXTRA:-}" | tee "${LOG}"
  rc=${PIPESTATUS[0]}
  set -e
fi

echo "\nSaved raw output to ${LOG}"

# Parse and print a summary if present
awk '
  /Average inference time cost total:/ {avg=$NF}
  /P50 Latency:/ {p50=$3}
  /P90 Latency:/ {p90=$3}
  /P95 Latency:/ {p95=$3}
  /P99 Latency:/ {p99=$3}
  END {
    if (avg!="") printf("avg_ms: %s\n", avg);
    if (p50!="") printf("p50_s: %s\n", p50);
    if (p90!="") printf("p90_s: %s\n", p90);
    if (p95!="") printf("p95_s: %s\n", p95);
    if (p99!="") printf("p99_s: %s\n", p99);
  }
' "${LOG}" || true
