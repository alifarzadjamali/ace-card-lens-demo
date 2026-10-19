"""Mark possible wear on a straightened card photo.

The checks measure border widths, pale spots on corners and edges, corner shape
and surface differences. Results point a grader to places worth looking at;
they are estimates from an image, not a card grade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import CARD_H, CARD_W, PX_PER_MM

# Published Ace front-centering bands, using the wider side of the ratio.
ACE_CENTERING_BANDS = [(60, "10 band (≤60/40)"), (65, "9 band (≤65/35)"),
                       (70, "8 band (≤70/30)"), (75, "7 band (≤75/25)"),
                       (80, "6–4 band (≤80/20)"), (85, "3–1 band (≤85/15)")]
NOMINAL_CORNER_RADIUS_MM = 3.0   # Approximate radius of a newly cut card corner.


@dataclass
class Finding:
    check: str
    location: str
    value: str
    severity: str          # ok | minor | major | n/a
    detail: str = ""
    box: tuple | None = None   # x, y, w, h on the rectified card


@dataclass
class DefectReport:
    findings: list[Finding] = field(default_factory=list)
    centering: dict = field(default_factory=dict)
    overlay: np.ndarray | None = None
    surface_heatmap: np.ndarray | None = None
    surface_mode: str = ""


# Centering
def _border_width(card_lab: np.ndarray, side: str) -> tuple[float, float]:
    """Measure one border from the point where its colour changes."""
    H, W = card_lab.shape[:2]
    if side in ("left", "right"):
        band = card_lab[int(0.3 * H):int(0.7 * H)]
        prof = band.mean(0) if side == "left" else band.mean(0)[::-1]
        span = W
    else:
        band = card_lab[:, int(0.3 * W):int(0.7 * W)]
        prof = band.mean(1) if side == "top" else band.mean(1)[::-1]
        span = H
    prof = prof[: int(0.08 * span)]
    skip = int(0.012 * span)                      # Skip the cut edge and nearby background bleed.
    ref = np.median(prof[skip: skip + max(4, int(0.012 * span))], 0)
    d = np.linalg.norm(prof - ref, axis=1)
    d = np.convolve(d, np.ones(3) / 3, mode="same")
    thr = max(12.0, 0.45 * d[skip:].max())
    run = 0
    for i in range(skip, len(d)):
        run = run + 1 if d[i] > thr else 0
        if run >= 4:
            edge = i - 3
            strength = float(np.clip((d[edge:edge + 6].mean() - thr) / 30 + 0.5, 0, 1))
            if strength >= 0.3:
                return float(edge), strength
    return float("nan"), 0.0


def measure_centering(card_bgr: np.ndarray) -> dict:
    lab = cv2.cvtColor(cv2.GaussianBlur(card_bgr, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    w = {s: _border_width(lab, s) for s in ("left", "right", "top", "bottom")}
    out = {s + "_mm": (v[0] / PX_PER_MM) for s, v in w.items()}
    conf = min(v[1] for v in w.values())

    def ratio(a, b):
        if np.isnan(a) or np.isnan(b) or a + b == 0:
            return float("nan")
        return 100.0 * a / (a + b)

    lr, tb = ratio(w["left"][0], w["right"][0]), ratio(w["top"][0], w["bottom"][0])
    worst = max(lr, 100 - lr, tb, 100 - tb) if not (np.isnan(lr) or np.isnan(tb)) else float("nan")
    band = "Not assessable"
    if not np.isnan(worst):
        band = next((g for lim, g in ACE_CENTERING_BANDS if worst < lim + 1e-9), "Outside listed bands")
    out.update(lr=lr, tb=tb, worst=worst, band=band, confidence=conf, px=w)
    return out


def _fmt_ratio(r):
    return "Not assessable" if np.isnan(r) else f"{r:.0f}/{100 - r:.0f}"


# Corners and edges
def _border_colour(card_hsv: np.ndarray) -> np.ndarray:
    H, W = card_hsv.shape[:2]
    s = int(0.012 * W), int(0.03 * W)
    strips = [card_hsv[int(.3 * H):int(.7 * H), s[0]:s[1]], card_hsv[int(.3 * H):int(.7 * H), W - s[1]:W - s[0]],
              card_hsv[s[0]:s[1], int(.3 * W):int(.7 * W)], card_hsv[H - s[1]:H - s[0], int(.3 * W):int(.7 * W)]]
    return np.median(np.concatenate([x.reshape(-1, 3) for x in strips]), 0)


def _white_mask(card_bgr, border_hsv):
    """Find bright, low-colour pixels where the border is usually coloured."""
    hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
    s_thr = min(50.0, 0.35 * float(border_hsv[1]))
    return ((hsv[..., 2] >= 170) & (hsv[..., 1] <= s_thr)).astype(np.uint8)


def check_corners_edges(card_bgr: np.ndarray, mask: np.ndarray, edge_visible: bool) -> list[Finding]:
    H, W = card_bgr.shape[:2]
    hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
    border = _border_colour(hsv)
    border_is_white = border[1] < 60 and border[2] > 160
    white = _white_mask(card_bgr, border)
    inside = (mask > 0).astype(np.uint8)
    findings: list[Finding] = []

    # Check the outer 1.8 mm of the card.
    band_px = int(1.8 * PX_PER_MM)
    edge_band = np.zeros((H, W), np.uint8)
    edge_band[:band_px] = edge_band[-band_px:] = 1
    edge_band[:, :band_px] = edge_band[:, -band_px:] = 1
    edge_band &= inside

    c = int(6 * PX_PER_MM)  # 6 mm corner patch
    corners = {"top-left": (0, 0), "top-right": (W - c, 0), "bottom-right": (W - c, H - c), "bottom-left": (0, H - c)}
    r_nom_px = NOMINAL_CORNER_RADIUS_MM * PX_PER_MM
    for name, (x, y) in corners.items():
        patch_band = edge_band[y:y + c, x:x + c]
        patch_white = white[y:y + c, x:x + c] & patch_band
        frac = patch_white.sum() / max(1, patch_band.sum())
        if border_is_white or not edge_visible or patch_band.sum() < 0.1 * patch_band.size:
            sev, detail = "n/a", "The border is not clear enough to assess whitening."
        else:
            sev = "ok" if frac < 0.04 else ("minor" if frac < 0.12 else "major")
            detail = f"{100 * frac:.1f}% of border pixels read white"
        findings.append(Finding("Corner whitening", name, "Not assessable" if sev == "n/a" else f"{100 * frac:.1f}%", sev, detail, (x, y, c, c)))

        if edge_visible:
            missing = c * c - inside[y:y + c, x:x + c].sum()
            r_eq = np.sqrt(max(missing, 0) / (1 - np.pi / 4))  # radius of a quarter-circle with this missing area
            r_mm = r_eq / PX_PER_MM
            excess = r_mm - NOMINAL_CORNER_RADIUS_MM
            sev = "ok" if excess < 0.6 else ("minor" if excess < 1.4 else "major")
            findings.append(Finding("Corner rounding", name, f"r = {r_mm:.1f} mm", sev,
                                    f"A fresh corner is roughly {NOMINAL_CORNER_RADIUS_MM:.1f} mm", (x, y, c, c)))
        else:
            findings.append(Finding("Corner rounding", name, "n/a", "n/a",
                                    "Card outline not visible. Try a photo on a contrasting background."))

    # Leave corners out of the edge check.
    edges = {"top": (c, 0, W - 2 * c, band_px), "bottom": (c, H - band_px, W - 2 * c, band_px),
             "left": (0, c, band_px, H - 2 * c), "right": (W - band_px, c, band_px, H - 2 * c)}
    for name, (x, y, w, h) in edges.items():
        b = edge_band[y:y + h, x:x + w]
        frac = (white[y:y + h, x:x + w] & b).sum() / max(1, b.sum())
        if border_is_white or not edge_visible or b.sum() < 0.5 * b.size:
            sev = "n/a"
        else:
            sev = "ok" if frac < 0.02 else ("minor" if frac < 0.06 else "major")
        findings.append(Finding("Edge wear", name, "Not assessable" if sev == "n/a" else f"{100 * frac:.1f}%", sev, "white chipping along the cut edge", (x, y, w, h)))
    return findings


# Surface marks
def _colour_transfer(src_lab, ref_lab, valid):
    out = src_lab.copy()
    for ch in range(3):
        s, r = src_lab[..., ch][valid], ref_lab[..., ch][valid]
        out[..., ch] = (src_lab[..., ch] - s.mean()) / (s.std() + 1e-6) * r.std() + r.mean()
    return out


def _align_to_reference(card_bgr, ref_bgr):
    """Line up the card photo with its catalogue image using ECC."""
    warp = np.eye(3, dtype=np.float32)
    ok = False
    for f in (4, 2):
        a = cv2.cvtColor(cv2.resize(ref_bgr, (CARD_W // f, CARD_H // f)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(cv2.resize(card_bgr, (CARD_W // f, CARD_H // f)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        S = np.diag([1.0 / f, 1.0 / f, 1.0]).astype(np.float32)
        w = S @ warp @ np.linalg.inv(S)
        try:
            _, w = cv2.findTransformECC(a, b, w.astype(np.float32), cv2.MOTION_HOMOGRAPHY,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6), None, 3)
            warp = (np.linalg.inv(S) @ w @ S).astype(np.float32)
            ok = True
        except cv2.error:
            break
    if not ok:
        return card_bgr, False
    return cv2.warpPerspective(card_bgr, warp, (CARD_W, CARD_H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE), True


def _tolerant_diff(q, r, shift=3):
    """Allow small shifts before counting a pixel as a surface difference."""
    best = None
    for dy in range(-shift, shift + 1, max(1, shift // 2) or 1):
        for dx in range(-shift, shift + 1, max(1, shift // 2) or 1):
            rs = np.roll(np.roll(r, dy, 0), dx, 1)
            d = np.abs(q - rs)
            d = 0.5 * d[..., 0] + 1.2 * d[..., 1] + 1.2 * d[..., 2]   # stains shift chroma more than luminance
            best = d if best is None else np.minimum(best, d)
    return best


def check_surface(card_bgr: np.ndarray, ref_bgr: np.ndarray | None, holo_hint: bool = False,
                  mask: np.ndarray | None = None):
    H, W = card_bgr.shape[:2]
    margin = int(2.5 * PX_PER_MM)
    valid = np.zeros((H, W), bool)
    valid[margin:-margin, margin:-margin] = True
    if mask is not None:   # Ignore background visible around damaged corners.
        valid &= cv2.erode(mask, np.ones((25, 25), np.uint8)) > 0

    if ref_bgr is not None:
        ref = cv2.resize(ref_bgr, (W, H), interpolation=cv2.INTER_AREA)
        aligned, ok = _align_to_reference(card_bgr, ref)
        q = cv2.cvtColor(cv2.GaussianBlur(aligned, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        r = cv2.cvtColor(cv2.GaussianBlur(ref, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        q = _colour_transfer(q, r, valid)
        diff = cv2.GaussianBlur(_tolerant_diff(q, r, 4), (9, 9), 0)
        mode = "Compared with catalogue image" + (" (aligned)" if ok else " (unaligned)")
    else:
        lab = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg = cv2.medianBlur(card_bgr, 31)
        bgl = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.abs(lab - bgl)
        diff = cv2.GaussianBlur(0.4 * d[..., 0] + 1.2 * d[..., 1] + 1.2 * d[..., 2], (9, 9), 0)
        mode = "Local colour check (no catalogue reference)"

    diff[~valid] = 0
    vals = diff[valid]
    med, mad = np.median(vals), np.median(np.abs(vals - np.median(vals))) + 1e-6
    thr = max(med + 6.0 * 1.4826 * mad, 28.0 if ref_bgr is not None else 22.0)
    m = (diff > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m)
    findings = []
    min_area = int(1.0 * PX_PER_MM * PX_PER_MM)      # ignore < 1 mm^2
    blobs = sorted([stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area],
                   key=lambda s: -s[cv2.CC_STAT_AREA])[:6]
    for s in blobs if ref_bgr is not None else []:
        x, y, w, h, a = (int(v) for v in s)
        area_mm2 = a / PX_PER_MM ** 2
        sev = "minor" if area_mm2 < 8 else "major"
        loc = f"{'upper' if y + h / 2 < H / 2 else 'lower'} {'left' if x + w / 2 < W / 2 else 'right'}"
        findings.append(Finding("Surface mark / stain", loc, f"{area_mm2:.1f} mm²", sev,
                                "differs from reference" if ref_bgr is not None else "local colour anomaly", (x, y, w, h)))
    if not findings:
        findings.append(Finding("Surface mark / stain", "whole card", "Not assessable" if ref_bgr is None else "none found",
                                "n/a" if ref_bgr is None else "ok", mode))
    if holo_hint and ref_bgr is not None:
        for f in findings:
            f.detail += ". Foil glare can look like a mark."
    heat = np.clip(diff / (thr * 2) * 255, 0, 255).astype(np.uint8)
    return findings, cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), mode


# Run the checks and draw the marked card
SEV_COLOUR = {"ok": (80, 200, 120), "minor": (40, 180, 250), "major": (60, 60, 235), "n/a": (160, 160, 160)}


def analyse(card_bgr, mask, edge_visible, ref_bgr=None, holo_hint=False) -> DefectReport:
    rep = DefectReport()
    cen = measure_centering(card_bgr)
    rep.centering = cen
    def centering_severity(value):
        return "n/a" if np.isnan(value) else ("ok" if max(value, 100 - value) < 60 else ("minor" if max(value, 100 - value) < 70 else "major"))

    rep.findings.append(Finding("Centering", "left/right", _fmt_ratio(cen["lr"]), centering_severity(cen["lr"]),
                                f"L {cen['left_mm']:.1f} mm · R {cen['right_mm']:.1f} mm"))
    rep.findings.append(Finding("Centering", "top/bottom", _fmt_ratio(cen["tb"]), centering_severity(cen["tb"]),
                                f"T {cen['top_mm']:.1f} mm · B {cen['bottom_mm']:.1f} mm"))
    rep.findings += check_corners_edges(card_bgr, mask, edge_visible)
    surf, heat, mode = check_surface(card_bgr, ref_bgr, holo_hint, mask)
    rep.findings += surf
    rep.surface_heatmap, rep.surface_mode = heat, mode
    rep.overlay = draw_overlay(card_bgr, rep)
    return rep


def draw_overlay(card_bgr, rep: DefectReport) -> np.ndarray:
    out = card_bgr.copy()
    H, W = out.shape[:2]
    px = rep.centering.get("px", {})
    col = (255, 210, 60)
    for side, (v, _) in px.items():
        if np.isnan(v):
            continue
        v = int(v)
        if side == "left":
            cv2.line(out, (v, 0), (v, H), col, 2, cv2.LINE_AA)
        elif side == "right":
            cv2.line(out, (W - 1 - v, 0), (W - 1 - v, H), col, 2, cv2.LINE_AA)
        elif side == "top":
            cv2.line(out, (0, v), (W, v), col, 2, cv2.LINE_AA)
        else:
            cv2.line(out, (0, H - 1 - v), (W, H - 1 - v), col, 2, cv2.LINE_AA)
    rank = {"n/a": 0, "ok": 1, "minor": 2, "major": 3}
    # Draw the stronger flag when two checks cover the same corner.
    boxes = {}
    for f in rep.findings:
        if f.box is None or f.check == "Centering":
            continue
        k = (f.box, f.check.split()[0])
        if k not in boxes or rank[f.severity] > rank[boxes[k].severity]:
            boxes[k] = f
    for (box, _), f in boxes.items():
        if f.severity in ("ok", "n/a") and f.check != "Surface mark / stain":
            continue
        if f.severity == "ok":
            continue
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), SEV_COLOUR[f.severity], 3, cv2.LINE_AA)
    return out
