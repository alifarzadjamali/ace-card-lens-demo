# AceVision

Take a photo of a trading card. AceVision finds close matches in a small Pokémon card catalogue and marks spots you might want to check, such as white corners, worn edges or uneven centring. You can review each flag and download your notes.

I built this for my Ace Grading and University of Salford KTP interview. It's a working demo, not an Ace Grading product or a card grading service. The checks can be fooled by glare, sleeves and a bad photo. A grader still makes the call.

[Open the demo](https://ace-vision-demo.streamlit.app/) or use the [QR code](assets/ace-vision-demo-qr.png).

## Give it a go

Open the demo and click **Generate photo & analyse**. It starts with a catalogue card, adds some simulated wear and puts it on a desk background. You can choose a different card or change the defects before you run it.

You can also upload a photo or use your camera. Keep all four corners visible, with some plain background around the card. If the app cannot find the outline, it stops and asks for another photo. Tightly cropped scans may also be rejected.

After analysis, you'll see the detected outline, a straightened view, a surface difference map and possible catalogue matches. The **Awaiting grader review** tab lets you confirm or reject findings, correct the catalogue card and download a JSON record. Uploaded photos stay in the current session; the app doesn't save them to a database.

The five sample presets cover off-centre printing, corner whitening, corner rounding, stains and edge wear. Tests check the five card outlines and the whitening, rounding, stain and edge-wear examples. The off-centre shift is subtle and may not be flagged. A reference image must match closely before the app flags a surface mark. Whitening on pale borders and measurements without a clear card edge are shown as not assessable. These tests use simulated damage; they don't establish accuracy on real cards.

## How it works

OpenCV finds the card and straightens it to a 630 × 880 pixel image. The app then compares it with 565 cards from Base Set, Jungle, Fossil, Base Set 2 and 151. DINOv3 uses learned visual features; the traditional option uses colour, layout and a perceptual hash. Set, rarity and name filters narrow the search. ORB and RANSAC check whether the printed layout of the closest cards lines up with the photo. RapidOCR reads the name and card number when legible, adding a small agreement bonus to matching catalogue entries. If text cannot be read, image search works as before.

The condition checks compare border widths for centring, look for pale pixels near corners and edges, and measure visible corner shape. Measurements in millimetres assume a 63 × 88 mm card. The surface check uses a catalogue image only when enough printed details line up. Without a reliable reference, surface marks are not assessable. The centring band is an estimate, not a final grade. A grader reviews each finding and can correct the catalogue choice in the JSON record.

The search uses a small, exact cosine index. There are two prebuilt indexes in `data/`, so visitors don't have to build them. For a much larger catalogue, an approximate index such as HNSW would be worth testing.

## Run it on your computer

From this folder in Windows PowerShell:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```

Open the local address that Streamlit prints. The app needs internet access for card images. On its first DINOv3 run, it also downloads the model from Hugging Face. If that fails, you can use the baseline encoder.

If you change the catalogue, rebuild the indexes with:

```bash
.venv/Scripts/python.exe scripts/build_index.py
```

To make a catalogue from other Pokémon TCG sets:

```bash
python scripts/build_catalogue.py base1 base2 base3 base4 sv3pt5 swsh7   # any PokemonTCG set ids
```

## Put up your own copy

This demo runs on [Streamlit Community Cloud](https://share.streamlit.io). For this project, connect `alifarzadjamali/ace-vision-demo`, choose `main` and `app.py`, select Python 3.12, and set the custom subdomain to `ace-vision-demo`. For your own copy, connect your fork and choose an available subdomain. Check the new link in a private browser window, then try the sample and an upload. Open it before a presentation in case the app has gone to sleep.

## Where things live

| File | What it does |
| --- | --- |
| `app.py` | The Streamlit screen and review flow |
| `ace_vision_demo/ui.py` | The graphite and gold theme and page components |
| `ace_vision_demo/detect.py` | Finds and straightens the card |
| `ace_vision_demo/embed.py` | Turns the card image into a search vector |
| `ace_vision_demo/catalogue.py` | Loads card details and images, searches the index and checks shortlisted matches |
| `ace_vision_demo/defects.py` | Measures centring, corners, edges and surface marks |
| `ace_vision_demo/synth.py` | Makes sample photos with known defects |
| `scripts/` | Rebuilds the catalogue and indexes |

The card details come from the public [PokemonTCG data repository](https://github.com/PokemonTCG/pokemon-tcg-data). The card images are served from PokemonTCG.io and belong to Nintendo, Creatures, GAME FREAK and The Pokémon Company. They're used here for a non-commercial interview demo and aren't stored in this GitHub repo. Public access to an image isn't a licence to reuse it elsewhere. Please upload only photos you're allowed to share.
