"""
Prompt (2026-06): Read flat `output/` artifacts for the GUI dashboard — no engine imports.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OutputSnapshot:
    output_dir: Path
    runtime: Dict[str, Any] = field(default_factory=dict)
    manifest_files: Dict[str, int] = field(default_factory=dict)
    engine_state_bytes: int = 0
    engine_raw_b64_len: int = 0
    engine_f_b64_len: int = 0
    has_main_gif: bool = False
    main_gif_path: Optional[Path] = None
    mtime_latest: float = 0.0


def default_output_dir(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent
    return root / "output"


def scan_output(output_dir: Path) -> OutputSnapshot:
    snap = OutputSnapshot(output_dir=output_dir.resolve())
    if not snap.output_dir.is_dir():
        return snap

    rt_path = snap.output_dir / "runtime.json"
    if rt_path.is_file():
        try:
            snap.runtime = json.loads(rt_path.read_text(encoding="utf-8"))
            snap.mtime_latest = max(snap.mtime_latest, rt_path.stat().st_mtime)
        except Exception:
            pass

    manifest_path = snap.output_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snap.manifest_files = dict(manifest.get("files") or {})
            snap.mtime_latest = max(snap.mtime_latest, manifest_path.stat().st_mtime)
        except Exception:
            pass

    engine_path = snap.output_dir / "engine_state.json"
    if engine_path.is_file():
        snap.engine_state_bytes = engine_path.stat().st_size
        snap.mtime_latest = max(snap.mtime_latest, engine_path.stat().st_mtime)
        try:
            # gien: metadata only — avoid decoding multi-MB float64 payloads on every poll
            with open(engine_path, "r", encoding="utf-8") as f:
                head = f.read(4096)
            if '"serialized_raw_out"' in head:
                data = json.loads(engine_path.read_text(encoding="utf-8"))
                snap.engine_raw_b64_len = len(str(data.get("serialized_raw_out", "")))
                snap.engine_f_b64_len = len(str(data.get("serialized_f_out", "")))
        except Exception:
            pass

    gif_path = snap.output_dir / "environment_3d.gif"
    if gif_path.is_file():
        snap.has_main_gif = True
        snap.main_gif_path = gif_path
        snap.mtime_latest = max(snap.mtime_latest, gif_path.stat().st_mtime)

    return snap


def flat_artifact_rows(snap: OutputSnapshot) -> List[Dict[str, Any]]:
    """Top-level files only (flat layout) for dashboard table."""
    rows: List[Dict[str, Any]] = []
    if not snap.output_dir.is_dir():
        return rows
    for name in sorted(os.listdir(snap.output_dir)):
        full = snap.output_dir / name
        if full.is_file():
            rows.append({
                "name": name,
                "bytes": full.stat().st_size,
                "path": str(full),
            })
    return rows
