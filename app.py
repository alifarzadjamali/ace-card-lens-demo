"""I built AceVision to explore how a photo can help identify a card and highlight details worth checking by eye."""
from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ace_vision_demo.catalogue import VectorIndex, fetch_image, load_catalogue, ocr_agreement, read_card_text, rerank
from ace_vision_demo.defects import analyse
from ace_vision_demo.detect import draw_quad, detect_card
from ace_vision_demo.embed import BaselineEncoder, get_encoder
from ace_vision_demo.synth import make_sample

st.set_page_config(page_title="AceVision | Inspection studio", page_icon="🔎", layout="wide",
                   initial_sidebar_state="collapsed")

from ace_vision_demo import ui

# Streamlit Cloud can rerun app.py after a deploy while keeping imported modules in memory.
ui = importlib.reload(ui)

ui.apply_style()

SAMPLES = {
    "Charizard · Base Set (1999)": "base1-4",
    "Blastoise · Base Set (1999)": "base1-2",
    "Pikachu · Base Set (1999)": "base1-58",
    "Gengar · Fossil (1999)": "base3-5",
    "Snorlax · Jungle (1999)": "base2-11",
}
DEFECTS = ["Off-centre", "Corner whitening", "Rounded corner", "Stain", "Edge wear"]
SAMPLE_DEFAULTS = {
    "base1-4": ["Off-centre", "Rounded corner", "Stain"],
    "base1-2": ["Corner whitening"],
    "base1-58": ["Rounded corner"],
    "base3-5": ["Stain"],
    "base2-11": ["Edge wear"],
}


# Cached data and models
@st.cache_resource(show_spinner=False)
def catalogue():
    df = load_catalogue()
    return df, df.set_index("card_id", drop=False)


@st.cache_resource(show_spinner=False)
def encoder(key: str):
    try:
        return get_encoder(key), None
    except Exception as e:  # Keep the basic search available if model loading fails.
        return BaselineEncoder(), str(e)


@st.cache_resource(show_spinner=False)
def index_for(key: str):
    idx = VectorIndex.load(key)
    if idx is not None:
        return idx
    df, _ = catalogue()
    enc, _ = encoder(key)
    bar = st.progress(0.0, text=f"Getting {len(df)} card images and building the {enc.name} search index...")
    idx = VectorIndex.build(df, enc, progress=lambda f: bar.progress(min(1.0, f)))
    bar.empty()
    return idx


@st.cache_resource(show_spinner=False)
def text_reader():
    try:
        from rapidocr import RapidOCR
        return RapidOCR()
    except Exception:
        return None


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_upload(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:  # Pillow handles formats OpenCV may miss, including HEIC and WebP.
            from PIL import Image, ImageOps
            from pillow_heif import register_heif_opener
            import io
            register_heif_opener()
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    h, w = img.shape[:2]
    if max(h, w) > 2400:
        scale = 2400 / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# Search options
df, by_id = catalogue()
with st.sidebar:
    st.subheader("Search settings")
    enc_key = st.radio("Image search", ["dinov3s", "baseline"],
                       format_func=lambda k: {"dinov3s": "DINOv3 image model",
                                              "baseline": "Traditional image features (Colour + pHash)"}[k],
                       help="Try the image model or the simpler colour and pattern search.")
    use_rerank = st.toggle("Geometric re-ranking (ORB + RANSAC)", value=True,
                           help="Checks whether the printed layout lines up with each close match.")
    k = st.slider("Results to show", 3, 12, 6)
    st.markdown("Filter the catalogue")
    f_sets = st.multiselect("Set", sorted(df.set_name.unique()))
    f_rarity = st.multiselect("Rarity", sorted(r for r in df.rarity.unique() if r))
    f_name = st.text_input("Name contains", "")
    st.divider()
    st.caption(f"{len(df)} cards from {df.set_name.nunique()} sets. Card details come from PokemonTCG.io. "
               "Images © Nintendo / Creatures / GAME FREAK / The Pokémon Company. "
               "A personal project by Ali (Farzad) Jamali; no Ace Grading affiliation.")

# Page header
ui.hero(len(df), df.set_name.nunique())
ui.section("01", "Bring a card into focus", "Choose a sample, upload a photo, or use your camera.")

# Photo or sample
tab_sample, tab_upload, tab_cam = st.tabs(["Try a sample", "Upload a photo", "Use camera"])
with tab_sample:
    c1, c2, preview = st.columns([1.25, 1, .65], gap="large")
    with c1:
        pick = st.selectbox("Card", list(SAMPLES.keys()))
        chosen = st.multiselect("Add sample defects", DEFECTS, default=SAMPLE_DEFAULTS[SAMPLES[pick]],
                                key=f"defects-{SAMPLES[pick]}")
    with c2:
        bg = st.radio("Background", ["mat", "wood"], horizontal=True,
                      format_func=lambda b: {"mat": "Grading mat", "wood": "Desk"}[b])
        go = st.button("Generate photo & analyse", type="primary", width="stretch")
    with preview:
        sample_row = by_id.loc[SAMPLES[pick]]
        sample_image = fetch_image(SAMPLES[pick], sample_row.image_small)
        if sample_image is not None:
            st.image(to_rgb(sample_image), width=120)
        st.caption("CATALOGUE REFERENCE")
    if go:
        cid = SAMPLES[pick]
        row = by_id.loc[cid]
        src = fetch_image(cid, row.image_large, size="large", keep_alpha=True)
        if src is None:
            st.session_state.query = None
            st.error("Couldn't get that card image. Try another sample or upload a photo.")
        else:
            seed = 7
            rng = np.random.default_rng(seed)
            kw = dict(background=bg, seed=seed)
            if "Off-centre" in chosen:
                kw["offcentre"] = (float(rng.choice([-1, 1]) * rng.uniform(1.0, 1.6)), float(rng.uniform(-0.8, 0.8)))
            if "Corner whitening" in chosen:
                kw["whiten"] = list(rng.choice(["top-left", "top-right", "bottom-right", "bottom-left"], 2, replace=False))
            if "Rounded corner" in chosen:
                kw["round_corners"] = [str(rng.choice(["top-left", "top-right", "bottom-right", "bottom-left"]))]
            if "Stain" in chosen:
                kw["stain"] = True
            if "Edge wear" in chosen:
                kw["edge_wear"] = [str(rng.choice(["left", "right", "top", "bottom"]))]
            photo, truth = make_sample(src, **kw)
            st.session_state.update(query=photo, truth=truth, source=f"synthetic · {pick}", true_id=cid)
with tab_upload:
    up = st.file_uploader("Upload the front of one card. A flat card on a plain background works best. Avoid sleeve glare.",
                          type=["jpg", "jpeg", "png", "webp", "heic"])
    if up is not None and st.session_state.get("upload_name") != up.name + str(up.size):
        img = decode_upload(up.getvalue())
        if img is None:
            st.session_state.query = None
            st.error("Couldn't read that image.")
        else:
            st.session_state.update(query=img, truth=None, source=f"upload · {up.name}", true_id=None,
                                    upload_name=up.name + str(up.size))
with tab_cam:
    shot = st.camera_input("Hold the card flat, fill most of the frame")
    if shot is not None and st.session_state.get("cam_id") != hashlib.md5(shot.getvalue()).hexdigest():
        st.session_state.update(query=decode_upload(shot.getvalue()), truth=None, source="camera", true_id=None,
                                cam_id=hashlib.md5(shot.getvalue()).hexdigest())

if "query" not in st.session_state or st.session_state.query is None:
    ui.empty_state()
    st.stop()

# Analyse the card
query = st.session_state.query
det = detect_card(query)

if not det.edge_visible:
    st.warning("We couldn't detect the card outline. Please try another photo.")
    st.markdown("Place the card flat on a plain, contrasting background with all four corners visible. "
                "Keep your hands outside the card's edges, avoid glare, and leave a little space around the card.")
    st.caption("Analysis stopped. Upload a new photo or take another one using the camera tab above.")
    st.stop()

enc, enc_err = encoder(enc_key)
if enc_err:
    st.warning(f"DINOv3 couldn't load, so we're using the basic image search. Details: {enc_err[:120]}")
index = index_for(enc.key)

qvec = enc.encode([det.card])[0]

mask = pd.Series(True, index=df.index)
if f_sets:
    mask &= df.set_name.isin(f_sets)
if f_rarity:
    mask &= df.rarity.isin(f_rarity)
if f_name:
    mask &= df["name"].str.contains(f_name, case=False, regex=False)
allowed = set(df.card_id[mask]) if (f_sets or f_rarity or f_name) else None

hits = index.search(qvec, allowed=allowed, k=max(k, 10))
if use_rerank and hits:
    ranked = rerank(det.card, hits, by_id)
else:
    ranked = [(c, s, None, s) for c, s in hits]

card_text = read_card_text(det.card, text_reader(), df.name.unique().tolist())
ocr_evidence = {}
if card_text["status"] == "Read":
    checked = []
    for cid, sim, geo, score in ranked:
        bonus, evidence = ocr_agreement(card_text, by_id.loc[cid])
        ocr_evidence[cid] = evidence
        checked.append((cid, sim, geo, min(1.0, score + bonus)))
    ranked = sorted(checked, key=lambda item: -item[3])

top = ranked[0] if ranked else None
ref = None
# Image vectors can score highly for different cards. Only compare surfaces when
# the printed details also line up with the catalogue image.
if top and top[1] >= 0.55 and (top[2] or 0) >= 25:
    r = by_id.loc[top[0]]
    ref = fetch_image(top[0], r.image_large, size="large")
holo = bool(top) and "Holo" in by_id.loc[top[0]].rarity
rep = analyse(det.card, det.mask, det.edge_visible, ref_bgr=ref, holo_hint=holo)

# Results
ui.section("02", "The inspection desk", "Check the detected outline and the areas marked for review.")
flags = [f for f in rep.findings if f.severity in ("minor", "major")]
review_key = hashlib.sha1(cv2.imencode(".jpg", query)[1].tobytes()).hexdigest()
match_review = st.session_state.get(f"{review_key}-match", "Not reviewed yet")
review_complete = (match_review in ("Yes", "No, I corrected it")
                   and (match_review != "No, I corrected it" or st.session_state.get(f"{review_key}-correction"))
                   and all(st.session_state.get(f"{review_key}-dec{i}", "Not reviewed yet") != "Not reviewed yet"
                           for i in range(len(flags))))
cen = rep.centering
m1, m2, m3, m4 = st.columns(4)
m1.metric("Closest catalogue match", by_id.loc[top[0]]["name"] if top else "n/a",
          f"{top[3]:.2f} match score" if top else None, delta_color="off")
lr = "Not assessable" if np.isnan(cen["lr"]) else f"{cen['lr']:.0f}/{100 - cen['lr']:.0f}"
tb = "Not assessable" if np.isnan(cen["tb"]) else f"{cen['tb']:.0f}/{100 - cen['tb']:.0f}"
m2.metric("Centering L/R · T/B", f"{lr} · {tb}")
m3.metric("Estimated centring band", cen["band"].split(" (")[0], "Photo-based estimate", delta_color="off")
m4.metric("Findings to review", f"{len(flags)}", f"{sum(f.severity == 'major' for f in flags)} High-priority",
          delta_color="inverse" if flags else "off")

c1, c2, c3 = st.columns([0.85, 1.3, 0.85])
with c1, st.container(key="inspection_capture"):
    st.markdown('<div class="cl-panel-label"><span>01 / CAPTURE</span> Detected outline</div>', unsafe_allow_html=True)
    st.image(to_rgb(draw_quad(query, det.quad)), caption="Input photo + detected card outline", width="stretch")
with c2, st.container(key="inspection_rectified"):
    st.markdown('<div class="cl-panel-label"><span>02 / INSPECT</span> Rectified card</div>', unsafe_allow_html=True)
    st.image(to_rgb(rep.overlay), caption="Straightened card with border lines and marked areas",
             width="stretch")
with c3, st.container(key="inspection_surface"):
    st.markdown('<div class="cl-panel-label"><span>03 / COMPARE</span> Surface differences</div>', unsafe_allow_html=True)
    st.image(to_rgb(rep.surface_heatmap), caption=f"Surface difference map ({rep.surface_mode})",
             width="stretch")

# More detail
ui.section("03", "Evidence & review", "Compare the suggested matches and add your review below.")
t_sim, t_def, t_rev, t_how = st.tabs(["Similar cards", "Defect report",
                                     "Grader review complete" if review_complete else "Awaiting grader review", "How it works"])

with t_sim:
    if not ranked:
        st.warning("No catalogue cards match the current filters.")
    else:
        s1, s2 = ranked[0][3], (ranked[1][3] if len(ranked) > 1 else 0)
        conf = "High" if s1 >= 0.80 and s1 - s2 >= 0.03 else ("Medium" if s1 >= 0.65 else "Low")
        same_name = len(ranked) > 1 and by_id.loc[ranked[0][0]]["name"] == by_id.loc[ranked[1][0]]["name"]
        st.markdown(f"Match signal (heuristic): **{conf}** · top match {s1:.3f}, gap to next {s1 - s2:+.3f}")
        if same_name and s1 - s2 < 0.05:
            st.warning("These printings look very similar. Check the set symbol and card number in the strips below.")
        cols = st.columns(min(k, 6))
        for i, (cid, sim, geo, score) in enumerate(ranked[:k]):
            r = by_id.loc[cid]
            with cols[i % len(cols)]:
                thumb = fetch_image(cid, r.image_small)
                st.image(to_rgb(thumb) if thumb is not None else r.image_small, width="stretch")
                st.markdown(f"**{i + 1}. {r['name']}**")
                st.progress(float(np.clip(score, 0, 1)))
                details = [
                    f"Match score: {score:.3f}",
                    f"Cosine: {sim:.3f}",
                ]
                if geo is not None:
                    details.append(f"Geometrically consistent matches: {geo}")
                details.extend([
                    f"Set: {r.set_name}",
                    f"Card number: {r.number}/{r.set_total}",
                    f"Rarity: {r.rarity or 'n/a'}",
                    f"Release year: {r.release_date[:4]}",
                    f"<span class='cl-small'>Illustrator: {r.artist or 'n/a'}</span>",
                    f"<span class='cl-small'>Catalogue ID: {cid}</span>",
                ])
                if cid in ocr_evidence:
                    details.append(f"<span class='cl-small'>OCR: {ocr_evidence[cid]}</span>")
                st.markdown('<div class="cl-card-meta">' + '<br>'.join(details) + '</div>',
                            unsafe_allow_html=True)
        st.markdown("##### Check the small print: set symbol, number and copyright line")
        strip_cols = st.columns(4)
        strip = lambda img: to_rgb(cv2.resize(img, (630, 880))[770:880])
        strip_cols[0].image(strip(det.card), caption="Query", width="stretch")
        for j, (cid, *_rest) in enumerate(ranked[:3]):
            im = fetch_image(cid, by_id.loc[cid].image_small)
            if im is not None:
                strip_cols[j + 1].image(strip(im), caption=f"#{j + 1} {by_id.loc[cid].set_name}", width="stretch")

with t_def:
    st.markdown("Things worth checking on the card. These measurements come from the photo and aren't a grade.")
    if top is None or ref is None:
        st.warning("The card match wasn't clear enough for a reference comparison. Surface marks are not assessable from local colour alone; the heatmap may include artwork or lighting.")
    rows = [{"Check": f.check, "Location": f.location, "Measured": f.value,
             "Severity": {"major": "High-priority", "minor": "Review", "ok": "No flag", "n/a": "Not assessable"}[f.severity],
             "Detail": f.detail}
            for f in rep.findings]
    tbl = pd.DataFrame(rows)
    colours = {"No flag": "#1f6f43", "Review": "#b7791f", "High-priority": "#b42318", "Not assessable": "#555"}
    st.dataframe(tbl.style.map(lambda v: f"background-color:{colours.get(v, '')}; color:white" if v in colours else "",
                               subset=["Severity"]), width="stretch", hide_index=True)
    truth = st.session_state.get("truth")
    if truth:
        st.caption("Added to this sample photo: " + json.dumps(
            {k_: v for k_, v in truth.items() if v not in (None, [], (0.0, 0.0))}, default=str))
    st.caption("Estimated centering bands are photo-based guides, not final grades. Bands follow Ace's published front-centering limits (10 ≤ 60/40, 9 ≤ 65/35, 8 ≤ 70/30, "
               "7 ≤ 75/25). Corner rounding is the equivalent radius of material missing vs. a sharp corner; "
               "a new card's die-cut radius is ≈3 mm.")

with t_rev:
    st.markdown("Review each finding and the catalogue match, then download your decisions as JSON. A grader makes the final call.")
    decisions = {}
    for i, f in enumerate(flags):
        decisions[i] = st.radio(f"{f.check} · {f.location} · {f.value}",
                                ["Not reviewed yet", "Confirm", "Reject (false alarm)", "Unsure"],
                                horizontal=True, key=f"{review_key}-dec{i}")
    if not flags:
        st.success("Nothing was flagged. Still check the whole card before making a decision.")
    match_ok = st.radio("Catalogue match correct?", ["Not reviewed yet", "Yes", "No, I corrected it", "Unsure"],
                        horizontal=True, key=f"{review_key}-match")
    corrected_id = None
    if match_ok == "No, I corrected it":
        choices = df.card_id.tolist()
        corrected_id = st.selectbox("Correct catalogue card", choices, index=None,
                                    format_func=lambda cid: f"{by_id.loc[cid]['name']} · {by_id.loc[cid].set_name} · #{by_id.loc[cid].number}/{by_id.loc[cid].set_total} · {cid}",
                                    placeholder="Select the correct card", key=f"{review_key}-correction")
        if corrected_id is None:
            st.info("Select the correct catalogue card to complete this correction.")
    note = st.text_input("Grader note", "")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_sha1": review_key,
        "model": {"encoder": enc.name, "rerank": use_rerank, "pipeline": "AceVision-0.2"},
        "match": {"card_id": top[0] if top else None, "cosine": round(top[1], 4) if top else None,
                  "match_score": round(top[3], 4) if top else None, "grader": match_ok,
                  "corrected_card_id": corrected_id, "ocr_evidence": ocr_evidence.get(top[0]) if top else None},
        "ocr": card_text,
        "centering": {"lr": None if np.isnan(cen["lr"]) else round(cen["lr"], 1),
                      "tb": None if np.isnan(cen["tb"]) else round(cen["tb"], 1), "estimated_band": cen["band"]},
        "findings": [{"check": f.check, "location": f.location, "value": f.value, "severity": f.severity,
                      "grader": decisions.get(i)} for i, f in enumerate(flags)],
        "note": note,
    }
    st.download_button("Download audit record (JSON)", json.dumps(record, indent=2), "AceVision_review.json",
                       "application/json", disabled=match_ok == "No, I corrected it" and corrected_id is None)

with t_how:
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#171C25",
 fontcolor="white", color="#D5B26C", fontname="Helvetica", fontsize=11]; edge [color="#999"];
 photo [label="Photo / scan"]; det [label="Detect + rectify\\n63×88 mm, 10 px/mm"];
 emb [label="Embedding\\nDINOv3 or traditional features"]; ann [label="Vector index\\n(exact cosine search)"];
 ocr [label="OCR\\nname + card number"];
 meta [label="Card details\\nset · number · rarity", shape=cylinder]; rr [label="Re-rank\\ngeometric check"];
 ref [label="Reference image\\nof matched card"]; dfx [label="Defect checks\\ncentring · corners · edges · surface"];
 ui [label="Review\\nconfirm / reject"]; log [label="Download\\nreview JSON", shape=cylinder];
 photo->det->emb->ann->rr; det->ocr->rr; meta->ann [style=dashed, label=" filters"]; rr->ref->dfx; det->dfx; rr->ui; dfx->ui->log;
}""")
    st.markdown("""
OpenCV finds the card outline and straightens the photo to 630 × 880 pixels. Measurements assume a 63 × 88 mm card, giving a scale of 10 pixels per millimetre.

DINOv3 turns the card image into a vector for similarity search. The traditional option uses colour, layout and a perceptual hash. The app compares those vectors with the catalogue and applies any set, rarity or name filters you chose. For the closest matches, ORB and RANSAC check whether the printed details line up. OCR reads the name and card number when legible and gives matching catalogue details a small score bonus. If OCR cannot read the text, the image matches are still available.

The condition checks measure border widths, look for pale marks near corners and edges, and estimate corner shape. The app compares surfaces with a catalogue image only when the printed details line up; ECC then aligns the images. Without a reliable reference, surface marks are not assessable. Estimated centring bands are guides, not final grades. A grader reviews the findings and catalogue choice before exporting decisions as JSON.

This small catalogue uses exact cosine search. A larger collection could use pgvector or FAISS with HNSW, with card IDs shared across the search index and card details.
""")
