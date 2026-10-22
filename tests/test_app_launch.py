import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppLaunchTests(unittest.TestCase):
    def test_initial_page_renders(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.button[0].label, "Generate photo & analyse")


if __name__ == "__main__":
    unittest.main()
