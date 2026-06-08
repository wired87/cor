"""
Prompt (2026-06): Demo SOA input for offline visualization smoke tests.
"""
from __future__ import annotations
from typing import Any
import numpy as np
def _demo_input_data(timesteps: int = 42) -> dict[str, list[Any]]:
    t = np.linspace(0, 6 * np.pi, timesteps)
    return {
        "sensor_A": [np.array([np.sin(v), np.cos(v), np.sin(v * 0.5)]) for v in t],
        "sensor_B": [float(1.8 + 0.9 * np.sin(v * 0.8)) for v in t],
        "sensor_C": [
            np.array([complex(np.sin(v + i * 0.2), np.cos(v * 0.6 - i * 0.15)) for i in range(5)])
            for v in t
        ],
        "sensor_D": [
            {
                "list_array_3": np.array([np.sin(v * 0.3), np.cos(v * 0.4), np.sin(v * 0.5)]),
                "float": float(v / np.pi),
                "array_complex_5": np.array([complex(np.cos(v + i), np.sin(v - i)) for i in range(5)]),
                "int_field": int((v * 100) % 17),
            }
            for v in t
        ],
    }


