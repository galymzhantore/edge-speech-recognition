from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def record_clip(duration: float, sample_rate: int) -> np.ndarray:
    logger.info("Recording %.1f seconds...", duration)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a small bilingual dataset with annotations")
    parser.add_argument("--output-dir", default="data/local")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--label", default="moderate")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "local_manifest.csv"
    rows = []
    try:
        while True:
            waveform = record_clip(args.duration, args.sample_rate)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            clip_path = out_dir / f"{args.lang}_{timestamp}.wav"
            sf.write(clip_path, waveform, args.sample_rate)
            rows.append({
                "path": str(clip_path),
                "duration": args.duration,
                "text": input("Prompt text (enter to skip): ") or "",
                "speaker_id": input("Speaker ID: ") or "local",
                "label": args.label,
            })
            cont = input("Record another clip? (y/N): ").lower().startswith("y")
            if not cont:
                break
    except KeyboardInterrupt:
        logger.info("Recording interrupted")
    if rows:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "duration", "text", "speaker_id", "label"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d local clips to %s", len(rows), manifest)


if __name__ == "__main__":
    main()
