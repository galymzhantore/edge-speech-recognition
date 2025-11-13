from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils.config import load_config, parse_label_schema
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def stratified_split(df: pd.DataFrame, splits: Dict[str, float], strat_cols: List[str]) -> Dict[str, pd.DataFrame]:
    if not strat_cols:
        strat_cols = ["label"]
    key = df[strat_cols].astype(str).agg("-".join, axis=1)
    train_ratio = splits.get("train", 0.8)
    valid_ratio = splits.get("valid", 0.1)
    test_ratio = splits.get("test", 0.1)
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    train_idx, test_idx = next(splitter.split(df, key))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[test_idx].reset_index(drop=True)
    temp_ratio = valid_ratio / (valid_ratio + test_ratio)
    splitter_valid = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=43)
    valid_idx, test_idx = next(splitter_valid.split(df_temp, df_temp[strat_cols].astype(str).agg("-".join, axis=1)))
    df_valid = df_temp.iloc[valid_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    return {"train": df_train, "valid": df_valid, "test": df_test}


def enrich_labels(df: pd.DataFrame, schema_map: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["label_id"] = df["label"].map(schema_map)
    return df


def save_splits(splits: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = out_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        logger.info("Saved %s split with %d rows to %s", name, len(split_df), path)


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    summary_path = out_dir / "summary.md"
    label_counts = df["label"].value_counts().to_dict()
    speaker_counts = df["speaker_id"].nunique()
    lines = ["# Dataset Summary", "", f"Total clips: {len(df)}", f"Unique speakers: {speaker_counts}", "", "## Label distribution"]
    for label, count in label_counts.items():
        lines.append(f"- {label}: {count}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manifests and splits")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--valid", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest_name = args.manifest or cfg["data"].get("manifest_name", "commonvoice_manifest.csv")
    manifest_path = Path(args.output_dir) / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} not found; run download first")
    df = pd.read_csv(manifest_path)
    schema = parse_label_schema(cfg.get("data", {}).get("label_schema", "3class"))
    df = enrich_labels(df, schema)
    splits = stratified_split(df, {"train": args.train, "valid": args.valid, "test": args.test}, cfg["data"].get("stratify_by", ["label"]))
    split_dir = Path(args.output_dir) / "manifests"
    save_splits(splits, split_dir)
    summarize(df, Path(args.output_dir))


if __name__ == "__main__":
    main()
