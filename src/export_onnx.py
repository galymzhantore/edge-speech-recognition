from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from data.data_module import DataModule
from models.mlp import build_mlp
from utils.cli import parse_config
from utils.config import parse_label_schema
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _prepare_data(cfg: Dict) -> Tuple[np.ndarray, int, int, Dict[str, int]]:
    label_map = cfg["data"].get("label_map") or parse_label_schema(cfg["data"].get("label_schema", "3class"))
    data_module = DataModule(cfg["data"], cfg["features"], cfg.get("augmentation", {}), label_map)
    train_loader = data_module.train_dataloader(cfg["training"].get("batch_size", 32))
    sample = next(iter(train_loader))
    feat = sample["features"][:1]
    freq, frames = feat.shape[1:]
    return feat.numpy(), freq, frames, label_map


def _load_model(cfg: Dict, input_dim: int, num_classes: int, device: torch.device) -> nn.Module:
    model = build_mlp(cfg["training"].get("model", "mlp_small"), input_dim, num_classes, cfg["models"])
    ckpt = Path(cfg["data"].get("output_dir", "experiments")) / "checkpoints" / f"{cfg['training'].get('model', 'mlp_small')}.pt"
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(device)
    return model


def export_onnx(model: nn.Module, dummy: np.ndarray, out_path: Path, opset: int = 17) -> np.ndarray:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_tensor = torch.from_numpy(dummy)
    torch.onnx.export(
        model,
        dummy_tensor,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    logger.info("Saved ONNX model to %s", out_path)
    return dummy_tensor.numpy()


def validate_onnx(onnx_path: Path, sample_input: np.ndarray, torch_out: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: sample_input})[0]
    delta = float(np.max(np.abs(ort_out - torch_out)))
    logger.info("ONNX max deviation: %.6f", delta)
    return delta


def main() -> None:
    cfg = parse_config("Export models to ONNX")
    sample, freq, frames, label_map = _prepare_data(cfg)
    input_dim = freq * frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, input_dim, len(label_map), device)
    torch_out = model(torch.from_numpy(sample).to(device)).detach().cpu().numpy()

    export_dir = Path(cfg["data"].get("output_dir", "experiments")) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / "model.onnx"
    sample_input = export_onnx(model, sample, onnx_path, opset=int(cfg.get("export", {}).get("onnx_opset", 17)))
    onnx_delta = validate_onnx(onnx_path, sample_input, torch_out)

    label_map_path = export_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    summary = {
        "freq": freq,
        "frames": frames,
        "input_dim": input_dim,
        "label_map_path": str(label_map_path),
        "onnx_path": str(onnx_path),
        "onnx_delta": onnx_delta,
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Export summary saved to %s", export_dir / "summary.json")


if __name__ == "__main__":
    main()

