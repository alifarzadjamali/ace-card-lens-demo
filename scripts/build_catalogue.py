"""Build data/catalogue.csv from the open PokemonTCG data repo (metadata only).

Usage:  python scripts/build_catalogue.py base1 base2 base3 base4 sv3pt5
"""
import csv, json, sys, urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/PokemonTCG/pokemon-tcg-data/master"
DEFAULT_SETS = ["base1", "base2", "base3", "base4", "sv3pt5"]


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main(set_ids):
    sets = {s["id"]: s for s in fetch(f"{RAW}/sets/en.json")}
    rows = []
    for sid in set_ids:
        s = sets[sid]
        for c in fetch(f"{RAW}/cards/en/{sid}.json"):
            rows.append({
                "card_id": c["id"],
                "name": c["name"],
                "set_id": sid,
                "set_name": s["name"],
                "series": s["series"],
                "release_date": s["releaseDate"],
                "number": c.get("number", ""),
                "set_total": s.get("printedTotal", s.get("total", "")),
                "rarity": c.get("rarity", ""),
                "supertype": c.get("supertype", ""),
                "subtypes": "|".join(c.get("subtypes", []) or []),
                "types": "|".join(c.get("types", []) or []),
                "hp": c.get("hp", ""),
                "artist": c.get("artist", ""),
                "image_small": c["images"]["small"],
                "image_large": c["images"]["large"],
            })
    out = Path(__file__).resolve().parents[1] / "data" / "catalogue.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} cards from {len(set_ids)} sets -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SETS)
