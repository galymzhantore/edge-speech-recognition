from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SpeechDataset
from features.cache import FeatureCache


def load_cmvn(cmvn_path: Path) -> Optional[Dict]:
    if not cmvn_path.exists():
        return None
    data = np.load(cmvn_path)
    return {"mean": data["mean"], "var": data["var"]}


class DataModule:
    def __init__(self, cfg: Dict, feature_cfg: Dict, augmentation_cfg: Dict, label_map: Dict[str, int]) -> None:
        manifests_dir = Path(cfg.get("output_dir", "experiments")) / "manifests"
        self.manifests = {
            split: manifests_dir / f"{split}.csv"
            for split in ("train", "valid", "test")
        }
        cache = Path(feature_cfg.get("cache_dir", "data/features"))
        self.cache = FeatureCache(cache, feature_cfg.get("type", "mfcc"))
        cmvn_path = Path(cfg.get("output_dir", "experiments")) / "features" / "cmvn.npz"
        self.cmvn_stats = load_cmvn(cmvn_path)
        self.feature_cfg = feature_cfg
        self.data_cfg = cfg
        self.augmentation_cfg = augmentation_cfg
        self.label_map = label_map

    def _loader(self, split: str, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = SpeechDataset(
            self.manifests[split],
            self.cache,
            self.feature_cfg,
            self.data_cfg,
            self.cmvn_stats,
            self.label_map,
            augmentation_cfg=self.augmentation_cfg,
            split=split,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.data_cfg.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self, batch_size: int) -> DataLoader:
        return self._loader("train", batch_size, True)

    def valid_dataloader(self, batch_size: int) -> DataLoader:
        return self._loader("valid", batch_size, False)

    def test_dataloader(self, batch_size: int) -> DataLoader:
        return self._loader("test", batch_size, False)


__all__ = ["DataModule", "load_cmvn"]
