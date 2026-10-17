import unittest

import cv2
import numpy as np

from cardlens.catalogue import fetch_image, load_catalogue
from cardlens.defects import analyse, measure_centering
from cardlens.detect import CARD_H, CARD_W, detect_card
from cardlens.synth import make_sample


class CardLensRegressionTests(unittest.TestCase):
    def test_pikachu_outer_border_is_used(self):
        row = load_catalogue().set_index("card_id").loc["base1-58"]
        image = fetch_image("base1-58", row.image_small)
        if image is None:
            self.skipTest("Catalogue image unavailable")
        card = cv2.resize(image, (CARD_W, CARD_H))
        result = measure_centering(card)
        self.assertLess(result["top_mm"], 5)
        self.assertLess(result["bottom_mm"], 5)
        self.assertLess(max(result["tb"], 100 - result["tb"]), 65)

    def test_centering_keeps_side_order_and_severity_separate(self):
        card = np.full((CARD_H, CARD_W, 3), (20, 190, 230), np.uint8)
        card[25:CARD_H - 30, 43:CARD_W - 18] = (130, 80, 60)
        result = analyse(card, np.full((CARD_H, CARD_W), 255, np.uint8), False)
        self.assertGreater(result.centering["lr"], 65)
        self.assertLess(result.centering["tb"], 50)
        self.assertEqual(result.findings[0].severity, "major")
        self.assertEqual(result.findings[1].severity, "ok")

    def test_uncertain_border_and_unreferenced_surface_are_not_assessable(self):
        card = np.full((CARD_H, CARD_W, 3), 100, np.uint8)
        result = analyse(card, np.full((CARD_H, CARD_W), 255, np.uint8), False)
        self.assertEqual(result.centering["band"], "Not assessable")
        self.assertEqual(result.findings[-1].severity, "n/a")
        self.assertEqual(result.findings[-1].value, "Not assessable")

    def test_five_sample_cards_generate_detectable_outlines(self):
        cards = ["base1-4", "base1-2", "base1-58", "base3-5", "base2-11"]
        rows = load_catalogue().set_index("card_id")
        for card_id in cards:
            with self.subTest(card_id=card_id):
                source = fetch_image(card_id, rows.loc[card_id].image_large, size="large", keep_alpha=True)
                if source is None:
                    self.skipTest("Catalogue image unavailable")
                photo, _ = make_sample(source, seed=7)
                detected = detect_card(photo)
                self.assertTrue(detected.edge_visible)
                self.assertIsNotNone(detected.quad)

    def test_sample_defect_outputs(self):
        cases = [
            ("base1-2", {"whiten": ["top-left", "bottom-right"]}, "Corner whitening", "top-left"),
            ("base1-58", {"round_corners": ["top-right"]}, "Corner rounding", "top-right"),
            ("base3-5", {"stain": True}, "Surface mark / stain", None),
            ("base2-11", {"edge_wear": ["left"]}, "Edge wear", "left"),
        ]
        rows = load_catalogue().set_index("card_id")
        for card_id, defect, check, location in cases:
            with self.subTest(card_id=card_id):
                row = rows.loc[card_id]
                source = fetch_image(card_id, row.image_large, size="large", keep_alpha=True)
                reference = fetch_image(card_id, row.image_large, size="large")
                if source is None or reference is None:
                    self.skipTest("Catalogue image unavailable")
                photo, _ = make_sample(source, seed=7, **defect)
                detected = detect_card(photo)
                report = analyse(detected.card, detected.mask, detected.edge_visible, ref_bgr=reference)
                self.assertTrue(any(f.check == check and (location is None or f.location == location)
                                    and f.severity in ("minor", "major") for f in report.findings))


if __name__ == "__main__":
    unittest.main()
