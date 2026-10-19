"""Find the card in a photo and straighten it.

The output is 630 x 880 pixels, based on a 63 x 88 mm card at 10 pixels per mm.
That scale lets the later checks report widths and areas in physical units.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

CARD_W, CARD_H = 630, 880          # 63 x 88 mm at 10 px/mm
PX_PER_MM = 10.0


@dataclass
class Detection:
    card: np.ndarray            # rectified BGR card, CARD_H x CARD_W
    mask: np.ndarray            # rectified silhouette (uint8 0/255), tells us real corner shape
    quad: np.ndarray | None     # 4x2 corners in the source image (None = whole image used)
    method: str                 # how the card was found
    edge_visible: bool          # True when the card silhouette is trustworthy (photo on a background)


def _order(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _line_intersection(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    b = np.array([x2 - x1, y2 - y1], np.float64)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t, _ = np.linalg.solve(a, b)
    return np.array([x1 + t * vx1, y1 + t * vy1], np.float32)


def _refine_quad(contour: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Fit lines to the straight edges and find where they meet.

    A polygon corner can fall on the card's rounded edge. The fitted lines give
    us a sharp reference corner for measuring the actual curve.
    """
    c = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        seg = q - p
        L = np.linalg.norm(seg) + 1e-6
        u = seg / L
        n = np.array([-u[1], u[0]])
        rel = c - p
        t = rel @ u
        dist = np.abs(rel @ n)
        sel = c[(t > 0.15 * L) & (t < 0.85 * L) & (dist < 0.03 * L + 3)]
        if len(sel) < 10:
            return quad
        lines.append(cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    out = []
    for i in range(4):
        pt = _line_intersection(lines[i - 1], lines[i])
        if pt is None:
            return quad
        out.append(pt)
    out = np.array(out, np.float32)
    # Reject fitted corners that move too far from the first outline.
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.1 * np.linalg.norm(quad[0] - quad[2]):
        return quad
    return out


def _candidate_masks(img: np.ndarray):
    """Try two ways to separate the card from its background."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Find closed shapes from image edges.
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    filled = np.zeros_like(gray)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    filled = cv2.erode(filled, np.ones((5, 5), np.uint8), iterations=2)
    yield "edges", filled
    # Compare the image with the colour around its border.
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = 8
    ring = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                           lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, 0)
    dist = np.linalg.norm(lab - bg, axis=2)
    _, m = cv2.threshold(dist.astype(np.uint8) if dist.max() < 256 else np.clip(dist, 0, 255).astype(np.uint8),
                         0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    yield "background-model", m


def detect_card(bgr: np.ndarray) -> Detection:
    h0, w0 = bgr.shape[:2]
    scale = 1000.0 / max(h0, w0)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    if scale >= 1:
        scale = 1.0
    area_img = small.shape[0] * small.shape[1]

    best = None
    for name, m in _candidate_masks(small):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < 0.12 * area_img or a > 0.97 * area_img:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        approx = None
        for eps in (0.02, 0.03, 0.04, 0.05):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                approx = ap
                break
        if approx is None:
            approx = cv2.boxPoints(cv2.minAreaRect(hull))
            name += "+minAreaRect"
        quad = _order(approx)
        rect_fill = a / (cv2.contourArea(quad) + 1e-6)
        score = rect_fill  # how rectangular the blob is
        if best is None or score > best[0]:
            best = (score, name, c, quad)

    if best is None or best[0] < 0.85:
        # assume a flat scan / tightly cropped photo: the image *is* the card
        card = cv2.resize(bgr, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
        if w0 > h0 * 1.1:
            card = cv2.resize(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), (CARD_W, CARD_H))
        return Detection(card, np.full((CARD_H, CARD_W), 255, np.uint8), None, "full-frame (scan)", False)

    _, name, contour, quad = best
    quad = _refine_quad(contour, quad)
    # portrait orientation: if the TL->TR side is longer than TL->BL, rotate order
    if np.linalg.norm(quad[1] - quad[0]) > np.linalg.norm(quad[3] - quad[0]):
        quad = np.roll(quad, -1, axis=0)
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], np.float32)
    quad_full = quad / scale
    H = cv2.getPerspectiveTransform(quad_full, dst)
    card = cv2.warpPerspective(bgr, H, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    sil = np.zeros(small.shape[:2], np.uint8)
    cv2.drawContours(sil, [contour], -1, 255, cv2.FILLED)
    sil = cv2.resize(sil, (w0, h0), interpolation=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(sil, H, (CARD_W, CARD_H), flags=cv2.INTER_LINEAR)
    mask = (mask > 127).astype(np.uint8) * 255
    return Detection(card, mask, quad_full, name, True)


def draw_quad(bgr: np.ndarray, quad: np.ndarray | None, color=(60, 200, 255)) -> np.ndarray:
    out = bgr.copy()
    if quad is None:
        return out
    t = max(2, int(0.004 * max(out.shape[:2])))
    cv2.polylines(out, [quad.astype(np.int32)], True, color, t, cv2.LINE_AA)
    for p in quad.astype(int):
        cv2.circle(out, tuple(p), 3 * t, color, -1, cv2.LINE_AA)
    return out
