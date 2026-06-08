"""
Prompt (2026-06): Simulation control panel — parameter inputs and engine start (no core edits).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from nicegui import ui

from gui.engine_bridge import EngineRunner
from gui.output_monitor import default_output_dir


def register_control_panel(
    runner: EngineRunner,
    *,
    on_run_started: Callable[[], None],
    repo_root: Path,
) -> dict:
    """Build left-column controls; returns widget refs for app-level sync."""
    out_default = str(default_output_dir(repo_root))

    with ui.card().classes("w-full"):
        ui.label("Engine Control").classes("text-h6")
        ui.separator()

        amount_nodes = ui.number(
            label="AMOUNT_NODES",
            value=3,
            min=1,
            max=32,
            step=1,
        ).classes("w-full")
        sim_time = ui.number(
            label="SIM_TIME",
            value=3,
            min=1,
            max=64,
            step=1,
        ).classes("w-full")
        dims = ui.number(
            label="DIMS",
            value=3,
            min=1,
            max=3,
            step=1,
        ).classes("w-full")
        output_dir = ui.input(
            label="Output directory",
            value=out_default,
        ).classes("w-full")
        run_viz = ui.checkbox("Run visualization (light preset)", value=True)
        status_label = ui.label("Status: idle").classes("text-sm text-gray-600")

        def _start_run() -> None:
            if runner.is_running():
                ui.notify("Engine already running", type="warning")
                return
            ok = runner.start(
                amount_nodes=int(amount_nodes.value or 3),
                sim_time=int(sim_time.value or 3),
                dims=int(dims.value or 3),
                output_dir=str(output_dir.value or out_default),
                run_visualization=bool(run_viz.value),
            )
            if ok:
                status_label.set_text("Status: running")
                on_run_started()
                ui.notify("Engine started", type="positive")
            else:
                ui.notify("Could not start engine", type="negative")

        ui.button("Run simulation", on_click=_start_run, icon="play_arrow").classes("w-full")

    return {
        "amount_nodes": amount_nodes,
        "sim_time": sim_time,
        "dims": dims,
        "output_dir": output_dir,
        "run_viz": run_viz,
        "status_label": status_label,
    }
