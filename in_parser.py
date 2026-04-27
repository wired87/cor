import os

import numpy as np
from PIL import Image

import numpy as np
from PIL import Image

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

    # 3. Sortierung & Energie-Berechnung (Vektorisiert)
    # Wir berechnen die Luminanz für alle einzigartigen Farben gleichzeitig
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

    # 5. Strukturaufbau (nur wenn Dictionaries zwingend erforderlich sind)
    # ACHTUNG: Das Erstellen der Dicts ist immer noch langsam.
    # Falls du die Daten weiterverarbeitest, nutze lieber die numpy arrays 'pixel_ranks' und 'pixel_energies'.
    pos_to_eval = {}
    pos_to_color_code = {}

    # Falls du die Dicts wirklich brauchst, hier die schnellere Variante:
    for y in range(height):
        for x in range(width):
            pos = (x, y)
            pos_to_eval[pos] = pixel_energies[y, x]
            pos_to_color_code[pos] = pixel_ranks[y, x]

    return pos_to_eval, pos_to_color_code


def convert_img_to_energy_map():
    inection_map = []
    source_path = 'input'
    for file in os.listdir(source_path):
        path = os.path.join(source_path, file)
        pos_to_eval, pos_to_color_code = get_energy_workflow_optimized(path)
        inection_map.append({"photon": pos_to_eval})
    return inection_map

if __name__ == "__main__":
    convert_img_to_energy_map()
