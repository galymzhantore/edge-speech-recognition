from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger

from .manifest import enrich_labels, save_splits, stratified_split, summarize
from .speechocean_downloader import download_speechocean_dataset

logger = setup_logger(__name__)


def _extend_with_local(rows: List[Dict], local_manifest: Path) -> List[Dict]:
    if not local_manifest.exists():
        return rows
    local_df = pd.read_csv(local_manifest)
    records = local_df.to_dict(orient="records")
    logger.info("Merging %d locally recorded clips from %s", len(records), local_manifest)
    rows.extend(records)
    return rows


def _save_manifest(rows: List[Dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    logger.info("Wrote manifest with %d rows to %s", len(rows), manifest_path)


def main() -> None:
    cfg = parse_config("Download Common Voice/Speechocean datasets")
    data_cfg = cfg.get("data", {})
    data_root = Path(data_cfg.get("data_dir", "data"))
    manifest_rows = download_speechocean_dataset(cfg, data_root)
    local_manifest = data_cfg.get("local_manifest")
    if local_manifest:
        manifest_rows = _extend_with_local(manifest_rows, Path(local_manifest))
    manifest_name = data_cfg.get("manifest_name", "speechocean_manifest.csv")
    output_root = Path(data_cfg.get("output_dir", "experiments"))
    manifest_path = output_root / manifest_name
    _save_manifest(manifest_rows, manifest_path)

    label_map = data_cfg.get("label_map") or parse_label_schema(data_cfg.get("label_schema", "3class"))
    df = pd.read_csv(manifest_path)
    enriched = enrich_labels(df, label_map)
    ratio = data_cfg.get("split_ratio", {"train": 0.8, "valid": 0.1, "test": 0.1})
    splits = stratified_split(enriched, ratio, data_cfg.get("stratify_by", ["label"]))
    manifest_dir = output_root / "manifests"
    save_splits(splits, manifest_dir)
    summarize(enriched, output_root)


if __name__ == "__main__":
    main()
