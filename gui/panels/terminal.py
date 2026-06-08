"""
Prompt (2026-06): Live terminal panel — streams engine stdout captured by LogCapture.
"""

from __future__ import annotations

from nicegui import ui

from gui.engine_bridge import EngineRunner


def register_terminal_panel(runner: EngineRunner) -> dict:
    with ui.card().classes("w-full h-full flex flex-col"):
        ui.label("Engine Terminal").classes("text-h6")
        ui.separator()
        log_area = (
            ui.textarea(value="")
            .props("readonly outlined autogrow")
            .classes("w-full font-mono text-xs")
            .style("min-height: 280px; max-height: 420px; overflow-y: auto;")
        )
        clear_btn = ui.button("Clear", icon="delete_outline").props("flat dense")

        def _clear() -> None:
            log_area.set_value("")

        clear_btn.on_click(_clear)

    def append_logs() -> None:
        chunk = runner.drain_logs()
        if not chunk:
            return
        current = log_area.value or ""
        merged = (current + chunk)[-120_000:]
        log_area.set_value(merged)

    return {"log_area": log_area, "append_logs": append_logs}
