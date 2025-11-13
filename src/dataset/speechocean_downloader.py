from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import io

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

from utils.config import load_config
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def _score_to_label(score: float | None, thresholds: Dict[str, float]) -> str:
    if score is None:
        return "moderate"
    poor_max = thresholds.get("poor", 60)
    moderate_max = thresholds.get("moderate", 85)
    if score <= poor_max:
        return "poor"
    if score <= moderate_max:
        return "moderate"
    return "good"


def _read_audio_from_sample(sample_audio: Dict, sample_rate: int) -> np.ndarray:
    # sample_audio contains {"path": "...", "bytes": b"..."} when decode=False
    path = sample_audio.get("path")
    if path and Path(path).exists():
        audio, sr = librosa.load(path, sr=None, mono=True)
    else:
        audio_bytes = sample_audio.get("bytes")
        if audio_bytes is None:
            return np.zeros(1, dtype=np.float32)
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            audio = f.read(dtype="float32")
            sr = f.samplerate
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio


def process_split(dataset_name: str, split: str, root: Path, sample_rate: int, thresholds: Dict[str, float], max_clips: int | None, streaming: bool) -> List[Dict]:
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except ValueError:
        logger.warning("Split %s not found for %s; skipping", split, dataset_name)
        return []
    rows: List[Dict] = []
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    iterable = dataset if streaming else list(dataset)
    total = None if streaming else len(iterable)
    iterator = enumerate(iterable)
    for idx, sample in tqdm(iterator, desc=f"{split}", unit="clip", total=total):
        if max_clips and idx >= max_clips:
            break
        audio_info = sample.get("audio")
        if not audio_info:
            continue
        array = _read_audio_from_sample(audio_info, sample_rate)
        clip_path = split_dir / f"speechocean_{split}_{idx:06d}.wav"
        sf.write(clip_path, array, sample_rate)
        score = sample.get("final_score") or sample.get("score") or sample.get("overall_score")
        label = _score_to_label(score, thresholds)
        rows.append(
            {
                "path": str(clip_path.resolve()),
                "duration": float(len(array) / sample_rate),
                "text": sample.get("text", ""),
                "speaker_id": sample.get("speaker_id") or sample.get("speaker") or sample.get("prompt_id", "unknown"),
                "label": label,
            }
        )
    return rows


def download_speechocean_dataset(cfg: Dict, root: Path, splits: List[str] | None = None) -> List[Dict]:
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset_name", "mispeech/speechocean762")
    sample_rate = data_cfg.get("sample_rate", 16000)
    thresholds = data_cfg.get("score_thresholds", {"poor": 60, "moderate": 85})
    streaming = data_cfg.get("streaming", True)
    root = root / "speechocean"
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict] = []
    splits = splits or ["train", "validation", "test"]
    for split in splits:
        manifest_rows.extend(
            process_split(
                dataset_name,
                split,
                root,
                sample_rate,
                thresholds,
                data_cfg.get("max_clips"),
                streaming,
            )
        )
    logger.info("Collected %d clips from Speechocean 762", len(manifest_rows))
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mispeech/speechocean762 dataset")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--split", default="train+validation+test")
    parser.add_argument("--max-clips", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    splits = args.split.split("+")
    manifest_rows = download_speechocean_dataset(cfg, data_dir, splits)
    manifest_name = cfg.get("data", {}).get("manifest_name", "speechocean_manifest.csv")
    manifest_path = Path(args.output_dir) / manifest_name
    if not manifest_rows:
        raise RuntimeError("No rows collected from Speechocean dataset")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    logger.info("Speechocean manifest saved to %s (%d rows)", manifest_path, len(manifest_rows))


if __name__ == "__main__":
    main()
