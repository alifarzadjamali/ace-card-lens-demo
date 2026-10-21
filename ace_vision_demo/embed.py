"""Turn a card image into numbers the search index can compare.

The basic encoder uses colour, layout and a perceptual hash. The other runs
DINOv3 ViT-S/16 through ONNX Runtime on the CPU. Both return normalised vectors,
so a dot product gives cosine similarity.
"""
from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)


class BaselineEncoder:
    name = "Traditional image features (Colour + pHash)"
    key = "baseline"

    def encode(self, cards_bgr: list[np.ndarray]) -> np.ndarray:
        feats = []
        for img in cards_bgr:
            img = cv2.resize(img, (210, 293), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            hist = np.sqrt(hist / (hist.sum() + 1e-6))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
            layout = cv2.resize(lab, (6, 8), interpolation=cv2.INTER_AREA).ravel()
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dct = cv2.dct(cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA))[:8, :8].ravel()
            phash = np.where(dct > np.median(dct[1:]), 1.0, -1.0)
            feats.append(np.concatenate([_l2(hist) * 1.0, _l2(layout - layout.mean()) * 1.0, _l2(phash) * 1.2]))
        return _l2(np.stack(feats))


class DinoEncoder:
    name = "DINOv3-S/16 (self-supervised ViT, ONNX)"
    key = "dinov3s"
    REPO = "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX"
    REVISION = "48988dfe73065df8d6f5ccc0edc7c8bcf307de41"

    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort
        if model_path is None:
            from huggingface_hub import snapshot_download
            from pathlib import Path
            try:
                model_dir = snapshot_download(
                    self.REPO,
                    revision=self.REVISION,
                    allow_patterns=["onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"],
                )
            except Exception as e:
                raise RuntimeError(f"could not download DINOv3 ONNX weights: {e}") from e
            model_path = str(Path(model_dir) / "onnx/model_quantized.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.outs = [o.name for o in self.sess.get_outputs()]

    @staticmethod
    def _prep(img_bgr: np.ndarray) -> np.ndarray:
        # Resize the whole card so the border and printed details stay in view.
        rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)

    def encode(self, cards_bgr: list[np.ndarray], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(cards_bgr), batch):
            x = np.stack([self._prep(c) for c in cards_bgr[i:i + batch]]).astype(np.float32)
            res = dict(zip(self.outs, self.sess.run(None, {self.inp: x})))
            tokens = res.get("last_hidden_state")
            if tokens is not None and tokens.ndim == 3:
                cls, patch = tokens[:, 0], tokens[:, 5:].mean(1)  # Skip four register tokens.
                v = np.concatenate([_l2(cls), _l2(patch)], axis=1)   # Combine CLS and average patch features.
            else:
                v = res.get("pooler_output", list(res.values())[0])
            out.append(_l2(v))
        return np.concatenate(out)


def get_encoder(key: str):
    if key == "dinov3s":
        return DinoEncoder()
    return BaselineEncoder()
