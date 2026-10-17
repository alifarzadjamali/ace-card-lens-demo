"""Load card details and images, then search the image vectors.

The demo keeps card details in a CSV and vectors in small NPZ files. It fetches
images from their public URLs and caches them locally. Each vector keeps its
card ID so a search result can point back to the right card details.
"""
from __future__ import annotations

import io
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "img_cache"
_session = requests.Session()
_session.headers["User-Agent"] = "CardLens-demo/1.0 (KTP interview prototype)"


def load_catalogue() -> pd.DataFrame:
    df = pd.read_csv(DATA / "catalogue.csv", dtype=str).fillna("")
    df["set_name"] = df["set_name"].replace({"Base": "Base Set"})
    df["label"] = df["name"] + " · " + df["set_name"] + " " + df["number"] + "/" + df["set_total"]
    return df


def _plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower())


def read_card_text(card_bgr: np.ndarray, engine, names: list[str]) -> dict:
    result = {"name": None, "number": None, "total": None, "status": "Unreadable"}
    if engine is None:
        result["status"] = "Unavailable"
        return result
    try:
        h, w = card_bgr.shape[:2]
        top = card_bgr[:int(h * 0.18)]
        bottom = card_bgr[int(h * 0.90):, int(w * 0.48):]
        text_top = engine(top)
        text_bottom = engine(cv2.resize(bottom, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        top_lines = [str(t) for t, s in zip(text_top.txts or [], text_top.scores or []) if s >= 0.65]
        bottom_lines = [str(t) for t, s in zip(text_bottom.txts or [], text_bottom.scores or []) if s >= 0.65]
        exact = [name for name in names if any(_plain(name) == _plain(line) for line in top_lines)]
        matches = exact or [name for name in names if any(_plain(name) in _plain(line) for line in top_lines)]
        if matches:
            result["name"] = max(matches, key=len)
        for line in reversed(bottom_lines):
            found = re.search(r"(?<!\d)(\d{1,3})\s*[/|]\s*(\d{2,3})", line)
            if found:
                number, total = map(int, found.groups())
                if 0 < number <= total:
                    result["number"], result["total"] = str(number), str(total)
                    break
        if result["name"] or result["number"]:
            result["status"] = "Read"
    except Exception:
        result["status"] = "Unavailable"
    return result


def ocr_agreement(text: dict, row) -> tuple[float, str]:
    if text["status"] != "Read":
        return 0.0, "Text not readable"
    name_ok = text["name"] is not None and _plain(text["name"]) == _plain(row["name"])
    number_ok = text["number"] is not None and text["number"] == str(row["number"]) and text["total"] == str(row["set_total"])
    bonus = (0.05 if name_ok else 0.0) + (0.10 if number_ok else 0.0)
    if name_ok and number_ok:
        return bonus, "Name and number agree"
    if name_ok:
        return bonus, "Name agrees"
    if number_ok:
        return bonus, "Number agrees"
    return 0.0, "No OCR agreement"


def _decode(raw: bytes, bg=(128, 128, 128)) -> np.ndarray | None:
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] == 4:      # Put transparent corners on a plain background.
        a = arr[..., 3:4].astype(np.float32) / 255.0
        arr = (arr[..., :3] * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)
    elif arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def fetch_image(card_id: str, url: str, size: str = "small", keep_alpha: bool = False) -> np.ndarray | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{card_id}_{size}.png"
    if p.exists():
        raw = p.read_bytes()
    else:
        try:
            r = _session.get(url, timeout=20)
            r.raise_for_status()
            raw = r.content
            p.write_bytes(raw)
        except Exception:
            return None
    if keep_alpha:
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    return _decode(raw)


def fetch_many(df: pd.DataFrame, progress: Callable[[float], None] | None = None) -> list[np.ndarray | None]:
    out: list = [None] * len(df)
    jobs = list(enumerate(zip(df.card_id, df.image_small)))
    done = 0
    with ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(fetch_image, cid, url): i for i, (cid, url) in jobs}
        for f in futs:
            out[futs[f]] = f.result()
            done += 1
            if progress and done % 10 == 0:
                progress(done / len(jobs))
    return out


class VectorIndex:
    """Compare a query with every stored vector. This is fine for 565 cards."""

    def __init__(self, ids: np.ndarray, vecs: np.ndarray, encoder_key: str):
        self.ids, self.vecs, self.encoder_key = ids, vecs.astype(np.float32), encoder_key
        self.pos = {c: i for i, c in enumerate(ids)}

    @classmethod
    def build(cls, df, encoder, progress=None):
        imgs = fetch_many(df, progress=lambda f: progress(0.7 * f) if progress else None)
        keep = [i for i, im in enumerate(imgs) if im is not None]
        vecs = []
        for j in range(0, len(keep), 32):
            chunk = keep[j:j + 32]
            vecs.append(encoder.encode([imgs[i] for i in chunk]))
            if progress:
                progress(0.7 + 0.3 * min(1.0, (j + 32) / max(1, len(keep))))
        idx = cls(df.card_id.values[keep], np.concatenate(vecs), encoder.key)
        idx.save()
        return idx

    def path(self):
        return DATA / f"index_{self.encoder_key}.npz"

    def save(self):
        np.savez_compressed(self.path(), ids=self.ids, vecs=self.vecs)

    @classmethod
    def load(cls, encoder_key):
        p = DATA / f"index_{encoder_key}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        return cls(z["ids"], z["vecs"], encoder_key)

    def search(self, q: np.ndarray, allowed: set | None = None, k: int = 10):
        sims = self.vecs @ q.ravel()
        order = np.argsort(-sims)
        res = []
        for i in order:
            if allowed is not None and self.ids[i] not in allowed:
                continue
            res.append((self.ids[i], float(sims[i])))
            if len(res) >= k:
                break
        return res


# Check the layout of close matches
_orb = cv2.ORB_create(nfeatures=1500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def geometric_inliers(query_bgr: np.ndarray, cand_bgr: np.ndarray) -> int:
    """Count printed details that line up between the photo and a match."""
    a = cv2.cvtColor(cv2.resize(query_bgr, (315, 440)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(cand_bgr, (315, 440)), cv2.COLOR_BGR2GRAY)
    ka, da = _orb.detectAndCompute(a, None)
    kb, db = _orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = [m for m, n in (p for p in _bf.knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.78 * n.distance]
    if len(good) < 8:
        return len(good) // 2
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return int(inl.sum()) if inl is not None else 0


def rerank(query_bgr, hits, df_by_id, top_n=8, w_geo=0.35):
    out = []
    for rank, (cid, sim) in enumerate(hits):
        geo = None
        if rank < top_n:
            row = df_by_id.loc[cid]
            img = fetch_image(cid, row.image_small)
            geo = geometric_inliers(query_bgr, img) if img is not None else 0
        g = min(1.0, (geo or 0) / 60.0)
        score = (1 - w_geo) * sim + w_geo * g if geo is not None else sim - w_geo
        out.append((cid, sim, geo, score))
    return sorted(out, key=lambda t: -t[3])
