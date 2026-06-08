"""
Prompt (2026-06): Output dashboard — flat `output/` artifacts, runtime specs, main animation preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from nicegui import ui

from gui.output_monitor import OutputSnapshot, flat_artifact_rows, scan_output


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def register_dashboard_panel(
    *,
    get_output_dir: Callable[[], str],
) -> dict:
    with ui.card().classes("w-full"):
        ui.label("Output Dashboard").classes("text-h6")
        ui.separator()

        runtime_box = ui.column().classes("w-full gap-1")
        engine_box = ui.column().classes("w-full gap-1")
        table = ui.table(
            columns=[
                {"name": "name", "label": "File", "field": "name", "align": "left"},
                {"name": "bytes", "label": "Size", "field": "bytes", "align": "right"},
            ],
            rows=[],
            row_key="name",
        ).classes("w-full")
        gif_slot = ui.column().classes("w-full items-center")

    def refresh() -> OutputSnapshot:
        out = Path(get_output_dir())
        snap = scan_output(out)

        runtime_box.clear()
        with runtime_box:
            ui.label("Runtime (runtime.json)").classes("text-subtitle2")
            if snap.runtime:
                for k, v in snap.runtime.items():
                    ui.label(f"{k}: {v}").classes("text-xs font-mono")
            else:
                ui.label("No runtime.json yet").classes("text-xs text-gray-500")

        engine_box.clear()
        with engine_box:
            ui.label("Engine state").classes("text-subtitle2")
            if snap.engine_state_bytes:
                ui.label(f"engine_state.json: {_fmt_bytes(snap.engine_state_bytes)}").classes("text-xs")
                ui.label(
                    f"serialized_raw_out (b64 chars): {snap.engine_raw_b64_len:,}"
                ).classes("text-xs")
                ui.label(
                    f"serialized_f_out (b64 chars): {snap.engine_f_b64_len:,}"
                ).classes("text-xs")
            else:
                ui.label("No engine_state.json yet").classes("text-xs text-gray-500")

        rows = flat_artifact_rows(snap)
        for r in rows:
            r["bytes"] = _fmt_bytes(int(r["bytes"]))
        table.rows = rows
        table.update()

        gif_slot.clear()
        with gif_slot:
            if snap.has_main_gif and snap.main_gif_path is not None:
                ui.label("Main animation").classes("text-subtitle2")
                ui.image(str(snap.main_gif_path)).classes("max-w-md border rounded")

        return snap

    return {"refresh": refresh}
