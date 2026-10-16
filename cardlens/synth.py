"""Synthetic 'phone photo' generator with controllable defects.

Lets interviewers test the pipeline without a physical card: take a clean
catalogue image, damage it in known ways (ground truth!), then put it on a desk
under uneven light with perspective, blur and JPEG noise.
The same idea scales into a data-augmentation / synthetic-defect pipeline for
training detectors when real labelled defects are scarce.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detect import CARD_H, CARD_W, PX_PER_MM


def _rounded_mask(radii_mm: dict) -> np.ndarray:
    m = np.zeros((CARD_H, CARD_W), np.uint8)
    cv2.rectangle(m, (0, 0), (CARD_W - 1, CARD_H - 1), 255, -1)
    for corner, r_mm in radii_mm.items():
        r = int(r_mm * PX_PER_MM)
        if r <= 0:
            continue
        x0 = 0 if "left" in corner else CARD_W - r
        y0 = 0 if "top" in corner else CARD_H - r
        m[y0:y0 + r, x0:x0 + r] = 0
        cx = r if "left" in corner else CARD_W - 1 - r
        cy = r if "top" in corner else CARD_H - 1 - r
        patch = np.zeros_like(m)
        cv2.circle(patch, (cx, cy), r, 255, -1, cv2.LINE_AA)
        sub = (slice(y0, y0 + r), slice(x0, x0 + r))
        m[sub] = np.maximum(m[sub], patch[sub])
    return m


def _whiten(card, corner, rng, strength=1.0):
    H, W = card.shape[:2]
    c = int(5 * PX_PER_MM)
    band = int(1.6 * PX_PER_MM * strength)
    x0 = 0 if "left" in corner else W - c
    y0 = 0 if "top" in corner else H - c
    yy, xx = np.mgrid[y0:y0 + c, x0:x0 + c]
    dx = xx if "left" in corner else (W - 1 - xx)
    dy = yy if "top" in corner else (H - 1 - yy)
    d = np.minimum(dx, dy) + rng.normal(0, 2.5, dx.shape)
    near_corner = np.hypot(dx, dy) < c * 0.95
    m = (d < band) & near_corner & (rng.random(dx.shape) < 0.85)
    m = cv2.dilate(m.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    roi = card[y0:y0 + c, x0:x0 + c]
    roi[m] = (0.15 * roi[m] + 0.85 * np.array([242, 244, 246])).astype(np.uint8)


def _edge_wear(card, edge, rng):
    H, W = card.shape[:2]
    for _ in range(14):
        L = rng.integers(10, 40)
        t = int(1.0 * PX_PER_MM)
        if edge in ("left", "right"):
            y = rng.integers(80, H - 80 - L); x = 0 if edge == "left" else W - t
            card[y:y + L, x:x + t] = (0.2 * card[y:y + L, x:x + t] + 0.8 * 243).astype(np.uint8)
        else:
            x = rng.integers(80, W - 80 - L); y = 0 if edge == "top" else H - t
            card[y:y + t, x:x + L] = (0.2 * card[y:y + t, x:x + L] + 0.8 * 243).astype(np.uint8)


def _stain(card, rng, size_mm=7.0):
    H, W = card.shape[:2]
    r = size_mm * PX_PER_MM / 2
    cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.2, 0.75) * H
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = np.arctan2(yy - cy, xx - cx)
    wobble = 1 + 0.18 * np.sin(3 * ang + rng.uniform(0, 6)) + 0.1 * np.sin(5 * ang + rng.uniform(0, 6))
    d = np.hypot((xx - cx) / 1.25, yy - cy) / (r * wobble)
    alpha = np.clip(1.15 - d, 0, 1) ** 0.6 * 0.75
    ring = np.exp(-((d - 0.95) ** 2) / 0.004) * 0.35          # coffee-ring darker rim
    alpha = np.clip(alpha + ring, 0, 0.9)[..., None]
    tint = np.array([60, 120, 165], np.float32)               # brown (BGR)
    out = card.astype(np.float32) * (1 - alpha) + (card.astype(np.float32) * tint / 255.0) * alpha
    card[:] = np.clip(out, 0, 255).astype(np.uint8)
    return (int(cx - r), int(cy - r), int(2 * r), int(2 * r))


def _offcentre(card, dx_mm, dy_mm):
    pad = 60
    big = cv2.copyMakeBorder(card, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dx, dy = int(dx_mm * PX_PER_MM), int(dy_mm * PX_PER_MM)
    return big[pad - dy:pad - dy + CARD_H, pad - dx:pad - dx + CARD_W].copy()


def _background(h, w, rng, style):
    if style == "wood":
        base = np.array([70, 110, 150], np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grain = np.sin(yy / 9.0 + 3 * np.sin(xx / 140.0) + rng.normal(0, 0.15, (h, w))) * 14
        img = base + grain[..., None] * np.array([0.6, 0.8, 1.0]) + rng.normal(0, 5, (h, w, 1))
    else:  # dark grading mat
        img = np.full((h, w, 3), (38, 34, 32), np.float32) + rng.normal(0, 4, (h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


def make_sample(card_bgra: np.ndarray, *, offcentre=(0.0, 0.0), whiten=(), round_corners=(), stain=False,
                edge_wear=(), seed=7, background="mat", tilt=0.6):
    rng = np.random.default_rng(seed)
    card_bgra = cv2.resize(card_bgra, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    card = card_bgra[..., :3].copy()
    if card_bgra.ndim == 3 and card_bgra.shape[2] == 4:
        # catalogue PNGs have transparent rounded corners: fill them with the border colour
        hole = (card_bgra[..., 3] < 200).astype(np.uint8) * 255
        if hole.any():
            card = cv2.inpaint(card, cv2.dilate(hole, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)
    truth = {"offcentre_mm": offcentre, "whitened": list(whiten), "rounded": list(round_corners),
             "stain": None, "edge_wear": list(edge_wear)}
    if any(offcentre):
        card = _offcentre(card, *offcentre)
    for c in whiten:
        _whiten(card, c, rng)
    for e in edge_wear:
        _edge_wear(card, e, rng)
    if stain:
        truth["stain"] = _stain(card, rng)
    radii = {c: (6.5 if c in round_corners else 3.0) for c in ("top-left", "top-right", "bottom-right", "bottom-left")}
    alpha = _rounded_mask(radii)

    # place on a desk with perspective
    S = 1400
    bg = _background(S, int(S * 0.8), rng, background)
    Hb, Wb = bg.shape[:2]
    ch = 0.68 * Hb
    cw = ch * CARD_W / CARD_H
    cx, cy = Wb / 2 + rng.uniform(-40, 40), Hb / 2 + rng.uniform(-30, 30)
    j = lambda: rng.uniform(-1, 1) * 40 * tilt
    dst = np.float32([[cx - cw / 2 + j(), cy - ch / 2 + j()], [cx + cw / 2 + j(), cy - ch / 2 + j()],
                      [cx + cw / 2 + j(), cy + ch / 2 + j()], [cx - cw / 2 + j(), cy + ch / 2 + j()]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]]), dst)
    warped = cv2.warpPerspective(card, M, (Wb, Hb), flags=cv2.INTER_LINEAR)
    wa2 = cv2.warpPerspective(alpha, M, (Wb, Hb), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    shadow = (cv2.GaussianBlur(np.roll(np.roll(wa2, 14, 0), 10, 1), (41, 41), 0) * 0.45)[..., None]
    wa = wa2[..., None]
    out = bg.astype(np.float32) * (1 - shadow)
    out = out * (1 - wa) + warped.astype(np.float32) * wa
    # uneven lighting + sensor noise + blur + JPEG
    yy, xx = np.mgrid[0:Hb, 0:Wb].astype(np.float32)
    light = 0.85 + 0.25 * np.exp(-(((xx - Wb * 0.3) / Wb) ** 2 + ((yy - Hb * 0.25) / Hb) ** 2) * 2.0)
    out = out * light[..., None] + rng.normal(0, 3, out.shape)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0.8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), truth
