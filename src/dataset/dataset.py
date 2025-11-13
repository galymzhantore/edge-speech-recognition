from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from audio.augment import apply_augmentations
from audio.processing import preprocess
from features.cache import FeatureCache
from features.cmvn import apply_cmvn
from features.extractor import extract_features


class SpeechDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        feature_cache: FeatureCache,
        feature_cfg: Dict,
        data_cfg: Dict,
        cmvn_stats: Optional[Dict],
        label_map: Dict[str, int],
        augmentation_cfg: Optional[Dict] = None,
        split: str = "train",
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        self.feature_cache = feature_cache
        self.feature_cfg = feature_cfg
        self.data_cfg = data_cfg
        self.cmvn_stats = cmvn_stats
        self.label_map = label_map
        self.augmentation_cfg = augmentation_cfg or {}
        self.split = split
        self.sample_rate = data_cfg.get("sample_rate", 16000)
        self.clip_range = data_cfg.get("clip_seconds", [1, 5])
        frame_shift_sec = feature_cfg.get("frame_shift_ms", 10) / 1000.0
        self.max_frames = max(1, int(self.clip_range[1] / frame_shift_sec))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        path = row["path"]
        label_str = row["label"]
        label = self.label_map.get(label_str, 0)
        feats = self.feature_cache.load(path)
        if feats is None or self.feature_cfg.get("on_the_fly", False):
            waveform = preprocess(
                path,
                self.sample_rate,
                self.clip_range[0],
                self.clip_range[1],
                self.augmentation_cfg.get("trim_db", 25),
                self.augmentation_cfg.get("vad_threshold", 0.5),
                self.augmentation_cfg.get("peak_normalize", True),
            )
            if self.split == "train" and self.augmentation_cfg.get("enabled", False):
                waveform = apply_augmentations(waveform, self.sample_rate, self.augmentation_cfg | {"noise_dir": self.data_cfg.get("noise_dir"), "rir_dir": self.data_cfg.get("rir_dir")})
            feats = extract_features(waveform, self.sample_rate, self.feature_cfg)
        if self.feature_cfg.get("cmvn") == "global" and self.cmvn_stats:
            feats = apply_cmvn(feats, "global", self.cmvn_stats)
        elif self.feature_cfg.get("cmvn") == "utterance":
            feats = apply_cmvn(feats, "utterance")
        feats = self._pad_frames(feats)
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        return {
            "features": feats_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
            "speaker_id": row.get("speaker_id", "unknown"),
        }

    def _pad_frames(self, feats: np.ndarray) -> np.ndarray:
        frames = feats.shape[1]
        if frames == self.max_frames:
            return feats
        if frames > self.max_frames:
            return feats[:, : self.max_frames]
        pad_width = self.max_frames - frames
        pad = np.zeros((feats.shape[0], pad_width), dtype=feats.dtype)
        return np.concatenate([feats, pad], axis=1)


__all__ = ["SpeechDataset"]
