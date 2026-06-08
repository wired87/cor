"""
COR local engine control terminal.

Prompt (2026-06): Create modular GUI control terminal for local engine — imports project
functions only via gui.engine_bridge; does not modify core pipeline modules.

Framework: NiceGUI — WebSocket push for live logs/dashboard refresh without full-page
reruns (better than Streamlit for long-running JAX simulations).
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from gui.engine_bridge import EngineRunner
from gui.panels.control import register_control_panel
from gui.panels.dashboard import register_dashboard_panel
from gui.panels.terminal import register_terminal_panel

_REPO_ROOT = Path(__file__).resolve().parent.parent


def build_app() -> EngineRunner:
    runner = EngineRunner()
    control_refs: dict = {}

    ui.page_title("COR Engine Control")
    with ui.header().classes("bg-slate-800 text-white"):
        ui.label("COR — Local Engine Control Terminal").classes("text-h5")
        ui.space()
        ui.label("modular · isolated · real-time").classes("text-caption opacity-80")

    with ui.row().classes("w-full gap-4 p-4 items-start"):
        with ui.column().classes("w-80 gap-4"):
            control_refs = register_control_panel(
                runner,
                on_run_started=lambda: None,
                repo_root=_REPO_ROOT,
            )

        with ui.column().classes("flex-grow gap-4"):
            terminal_refs = register_terminal_panel(runner)
            dashboard_refs = register_dashboard_panel(
                get_output_dir=lambda: str(control_refs["output_dir"].value),
            )

    def _tick() -> None:
        terminal_refs["append_logs"]()
        st = runner.status
        control_refs["status_label"].set_text(f"Status: {st}")
        if st in ("done", "error", "running"):
            dashboard_refs["refresh"]()

    ui.timer(0.35, _tick)
    dashboard_refs["refresh"]()
    return runner


def main() -> None:
    build_app()
    ui.run(
        title="COR Engine Control",
        reload=False,
        port=8765,
        show=True,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
