"""
Prompt (2026-06): Modular bridge to repo engine — spawns `gui.run_worker` subprocess so the GUI
never imports JAX/heavy deps at startup; stdout streams into the control terminal live.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import json
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


class EngineRunner:
    """Thread-safe local engine controller for the GUI."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._log_queue: Queue[str] = Queue()
        self._status: str = "idle"
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._result

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def is_running(self) -> bool:
        return self.status == "running"

    def drain_logs(self, max_chunks: int = 200) -> str:
        chunks: list[str] = []
        for _ in range(max_chunks):
            try:
                chunks.append(self._log_queue.get_nowait())
            except Empty:
                break
        return "".join(chunks)

    def start(
        self,
        *,
        amount_nodes: int,
        sim_time: int,
        dims: int,
        output_dir: str,
        run_visualization: bool,
        user_id: str = "public",
        inj_cfg: Any = None,
    ) -> bool:
        if self.is_running():
            return False
        with self._lock:
            self._status = "running"
            self._result = None
            self._error = None
        self._log_queue = Queue()
        payload = {
            "amount_nodes": amount_nodes,
            "sim_time": sim_time,
            "dims": dims,
            "output_dir": output_dir,
            "run_visualization": run_visualization,
            "user_id": user_id,
            "inj_cfg": inj_cfg,
        }
        self._thread = threading.Thread(
            target=self._run_worker,
            args=(payload,),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run_worker(self, payload: Dict[str, Any]) -> None:
        cmd = [sys.executable, "-m", "gui.run_worker"]
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(_REPO_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(payload))
            self._proc.stdin.close()

            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                self._log_queue.put(line)

            code = self._proc.wait()
            with self._lock:
                if code == 0:
                    self._status = "done"
                    self._result = {"exit_code": 0, "output_dir": payload.get("output_dir")}
                else:
                    self._status = "error"
                    self._error = f"worker exit code {code}"
        except Exception as exc:
            self._log_queue.put(f"\n[gui] engine error: {type(exc).__name__}: {exc}\n")
            with self._lock:
                self._error = str(exc)
                self._status = "error"
        finally:
            self._proc = None
