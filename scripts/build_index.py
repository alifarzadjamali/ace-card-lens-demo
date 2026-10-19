"""Download catalogue images and build both search indexes.

Run ``python scripts/build_index.py`` after changing the catalogue. The NPZ
files go in ``data/`` and can be committed with the app.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ace_vision_demo.catalogue import VectorIndex, load_catalogue  # noqa: E402
from ace_vision_demo.embed import get_encoder  # noqa: E402

if __name__ == "__main__":
    df = load_catalogue()
    for key in (sys.argv[1:] or ["dinov2s", "baseline"]):
        enc = get_encoder(key)
        idx = VectorIndex.build(df, enc, progress=lambda f: print(f"\r{key}: {100 * f:5.1f}%", end=""))
        print(f"\n{key}: {len(idx.ids)} vectors x {idx.vecs.shape[1]} dims -> {idx.path()}")
