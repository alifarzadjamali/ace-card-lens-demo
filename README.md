# CardLens — AI-assisted defect check & similarity search for trading cards

Concept demo by **Ali (Farzad) Jamali** for the Ace Grading × University of Salford KTP interview
(*Artificial Intelligence and Computer Vision Engineer*). Not affiliated with or endorsed by Ace Grading.
It proposes points of interest for a human grader; it does **not** assign or validate a card grade.

**Live demo:** https://ace-card-lens-demo.streamlit.app/ · [QR code for the presentation](assets/ace-card-lens-demo-qr.png)

Upload or photograph a card, or generate a synthetic "phone photo" with known defects. The app then:

1. **Detects and rectifies** the card (contours, fitted edge lines, homography) to 63 × 88 mm at 10 px/mm.
2. **Embeds** it with **DINOv2 ViT-S/14** (ONNX, CPU) or a classical colour + pHash baseline.
3. **Searches** a catalogue of 565 Pokémon cards (Base, Jungle, Fossil, Base Set 2, 151) by exact cosine similarity,
   with metadata filters (hybrid search).
4. **Re-ranks** the shortlist with ORB + RANSAC geometric verification.
5. **Checks possible defects**: centring ratio against Ace's front-centring bands, corner whitening, corner rounding (mm),
   edge wear, and stains (difference against the matched catalogue reference after ECC alignment).
6. **Keeps a human in the loop**: the grader confirms or rejects each flag, and the decisions are exported as an
   audit record or training label.

## Run locally (Windows PowerShell)
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```
Open the local URL shown by Streamlit. Select **Try a sample** and press **Generate photo & analyse**. The sample starts from a public catalogue image and adds synthetic defects; it is not a real damaged-card photograph. You can also upload or photograph a single card on a plain, contrasting background. The application processes images in memory and does not save uploaded photos; the audit record is downloaded only when the visitor requests it.

The repository includes two small prebuilt vector indexes so visitors do not have to build them on first use. If rebuilding after changing the catalogue, run:
```bash
.venv/Scripts/python.exe scripts/build_index.py
```

The app fetches card thumbnails and reference images from their public URLs when needed, and downloads DINOv2 ONNX weights from Hugging Face on the first deep-model run. The classical baseline is available if model download fails. Internet access is required for the sample and card thumbnails.

## Deploy on Streamlit Community Cloud
1. Connect this GitHub repository at https://share.streamlit.io and select `app.py` on `main`.
2. Select Python 3.12 and choose a public app URL. Verify that the app opens in a private browser window.
3. Run the default sample, then test an upload from a phone. Add the confirmed URL and a QR code to the presentation slide. Keep the clickable URL visible for interviewers attending on one device.
4. Open the app shortly before the interview and keep screenshots or a short recording ready in case the live service or image host is slow.

## Change the catalogue
```bash
python scripts/build_catalogue.py base1 base2 base3 base4 sv3pt5 swsh7   # any PokemonTCG set ids
```

## Code map
| File | Purpose |
|---|---|
| `cardlens/detect.py` | card localisation, edge-line refinement, perspective rectification |
| `cardlens/defects.py` | centring, corners (whitening + rounding), edges, surface/stains, overlay |
| `cardlens/embed.py` | baseline and DINOv2-ONNX encoders |
| `cardlens/catalogue.py` | metadata, image cache, vector index, geometric re-ranking |
| `cardlens/synth.py` | synthetic photo generator with ground-truth defects |
| `app.py` | Streamlit UI |

Card metadata comes from the public [PokemonTCG data repository](https://github.com/PokemonTCG/pokemon-tcg-data). Card images are publicly accessible through PokemonTCG.io but are **not open-licensed**: artwork © Nintendo / Creatures / GAME FREAK / The Pokémon Company. They are used here for a non-commercial interview demonstration and are not committed to this repository. Public availability does not grant reuse rights; replace the catalogue images if permission is required for wider use. Uploaded images should be ones the visitor is entitled to submit.
