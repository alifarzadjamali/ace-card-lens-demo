# CardLens

Take a photo of a trading card. CardLens finds close matches in a small Pokémon card catalogue and marks spots you might want to check, such as white corners, worn edges or uneven centring. You can review each flag and download your notes.

I built this for my Ace Grading and University of Salford KTP interview. It's a working demo, not an Ace Grading product or a card grading service. The checks can be fooled by glare, sleeves and a bad photo. A grader still makes the call.

[Open the demo](https://ace-card-lens-demo.streamlit.app/) or use the [QR code](assets/ace-card-lens-demo-qr.png).

## Give it a go

Open the demo and click **Generate photo & analyse**. It starts with a catalogue card, adds some simulated wear and puts it on a desk background. You can choose a different card or change the defects before you run it.

Got a card handy? Upload a photo or use your camera. One card on a plain background works best. You'll see the detected outline, a straightened view, a surface difference map and a shortlist of possible matches. The **Grader review** tab lets you confirm or reject the flags and download a JSON record. The app keeps uploaded photos in the current session; it doesn't save them to a database.

The five sample presets cover off-centre printing, corner whitening, corner rounding, stains and edge wear. Tests check the five card outlines and the whitening, rounding, stain and edge-wear examples. The off-centre shift is subtle and may not be flagged; a positive detection is not forced. A reference image must match closely before the app flags a surface mark. Whitening on pale borders and measurements without a clear card edge are shown as not assessable. Real wear and damage may look different.

## How it works

OpenCV finds the card and straightens it to a 630 × 880 pixel image. The app then compares it with 565 cards from Base Set, Jungle, Fossil, Base Set 2 and 151. You can search with DINOv2 or switch to a simple colour and perceptual-hash baseline. Set, rarity and name filters narrow the search. ORB and RANSAC check whether the printed layout of the closest cards lines up with the photo.

The condition checks use image measurements. They compare border widths for centring, look for pale pixels near corners and edges, and measure visible corner shape. The surface check uses a catalogue image only when enough printed details line up. Otherwise, it looks for unusual colours in the photo. Each result is a place to look, not a grade. A poor photo can still make artwork or lighting look like damage.

The search uses a small, exact cosine index. There are two prebuilt indexes in `data/`, so visitors don't have to build them. For a much larger catalogue, an approximate index such as HNSW would be worth testing.

## Run it on your computer

From this folder in Windows PowerShell:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m streamlit run app.py
```

Open the local address that Streamlit prints. The app needs internet access for card images. On its first DINOv2 run, it also downloads the model from Hugging Face. If that fails, you can use the baseline encoder.

If you change the catalogue, rebuild the indexes with:

```bash
.venv/Scripts/python.exe scripts/build_index.py
```

To make a catalogue from other Pokémon TCG sets:

```bash
python scripts/build_catalogue.py base1 base2 base3 base4 sv3pt5 swsh7   # any PokemonTCG set ids
```

## Put up your own copy

This demo runs on [Streamlit Community Cloud](https://share.streamlit.io). Connect your GitHub repo, choose `main` and `app.py`, and select Python 3.12. Check the new link in a private browser window, then try the sample and an upload. Open it before a presentation in case the app has gone to sleep.

## Where things live

| File | What it does |
| --- | --- |
| `app.py` | The Streamlit screen and review flow |
| `cardlens/detect.py` | Finds and straightens the card |
| `cardlens/embed.py` | Turns the card image into a search vector |
| `cardlens/catalogue.py` | Loads card details and images, searches the index and checks shortlisted matches |
| `cardlens/defects.py` | Measures centring, corners, edges and surface marks |
| `cardlens/synth.py` | Makes sample photos with known defects |
| `scripts/` | Rebuilds the catalogue and indexes |

The card details come from the public [PokemonTCG data repository](https://github.com/PokemonTCG/pokemon-tcg-data). The card images are served from PokemonTCG.io and belong to Nintendo, Creatures, GAME FREAK and The Pokémon Company. They're used here for a non-commercial interview demo and aren't stored in this GitHub repo. Public access to an image isn't a licence to reuse it elsewhere. Please upload only photos you're allowed to share.
