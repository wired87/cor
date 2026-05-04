import json
import os
import pprint

import numpy as np
from PIL import Image


def get_energy_workflow_optimized(image_path):
    # 1. Bild laden
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        width, height = img.size
        data = np.array(img)

    # 2. Einzigartige Farben finden (schon effizient durch numpy)
    all_pixels = data.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(all_pixels, axis=0, return_inverse=True)

    luminances = 0.299 * unique_colors[:, 0] + 0.587 * unique_colors[:, 1] + 0.114 * unique_colors[:, 2]

    # Sortier-Indizes: von hell nach dunkel
    sort_idx = np.argsort(-luminances)

    # Erstelle ein Mapping von "altem Index aus np.unique" zu "neuem Energy-Rang"
    rank_map = np.empty_like(sort_idx)
    rank_map[sort_idx] = np.arange(len(unique_colors))

    # 4. Color Codes & Energie-Werte für alle Pixel berechnen
    # Das ist der magische Teil: Wir weisen allen Pixeln gleichzeitig ihren Rang zu
    pixel_ranks = rank_map[inverse_indices].reshape(height, width)

    # Energie-Werte für die e_map (Mittelwert der RGB-Kanäle der einzigartigen Farben)
    energy_values_unique = np.mean(unique_colors, axis=1)
    pixel_energies = energy_values_unique[inverse_indices].reshape(height, width)

    pos_to_eval = {}
    pos_to_color_code = {}

    for y in range(height):
        for x in range(width):
            pos = (x, y)
            pos_to_eval[pos] = pixel_energies[y, x]
            pos_to_color_code[pos] = pixel_ranks[y, x]
    return pos_to_eval, pos_to_color_code


def convert_img_to_energy_map() -> tuple[dict[str, dict[tuple[int, int], tuple[int, int]]], int]:
    """with open("brainmaster.json", "r") as f:
        data = json.load(f)"""
    data = [((0,0,0), [[1,2,3,4,5,6,7,8,],[1,2,3,4,5,6,7,8,]])]
    max_len = max(len(v[1][0]) for v in data)
    print("injection map created max len inj", max_len)
    return {"PHOTON": data}, max_len

if __name__ == "__main__":
    convert_img_to_energy_map()
