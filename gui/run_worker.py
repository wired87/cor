"""
Prompt (2026-06): Isolated subprocess entry — calls `run_main_process` with JSON params on stdin.
Keeps GUI import surface minimal; worker runs with repo root on sys.path like `python main.py`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _bootstrap() -> None:
    os.chdir(_REPO_ROOT)
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    for sub in ("cor", "color_master"):
        p = str(_REPO_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    _bootstrap()
    raw = sys.stdin.read()
    params = json.loads(raw) if raw.strip() else {}
    from main import run_main_process  # noqa: WPS433 — intentional repo-root import after bootstrap

    run_main_process(
        amount_nodes=int(params.get("amount_nodes", 3)),
        sim_time=int(params.get("sim_time", 3)),
        dims=int(params.get("dims", 3)),
        output_dir=params.get("output_dir"),
        user_id=params.get("user_id", "public"),
        run_visualization=bool(params.get("run_visualization", True)),
        inj_cfg=params.get("inj_cfg"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
