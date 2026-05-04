from __future__ import annotations

import json
import os

import dotenv
dotenv.load_dotenv()


def load_data():
    # Resolve cfg next to qbrain package (not cwd) so main.py runs from repo root.

    root = os.path.dirname(os.path.abspath(qbrain.__file__))
    path = os.path.join(root, "test_out.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())



