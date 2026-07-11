"""Resource-limited Python tool session.

Drop-in replacement for SecureLightweightPythonSession with the same
`execute(code) -> stdout` contract and persistent variables across calls, but the
sandbox runs in a subprocess with:
  - an RLIMIT_AS memory cap (runaway allocation -> caught MemoryError, not system OOM),
  - the inner session's SIGALRM timeout (works: subprocess main thread),
  - a parent-enforced wall-clock timeout that SIGKILLs a hung child (uncatchable backstop,
    works off the main thread where SIGALRM never fires).

This is what makes high-concurrency tool use safe. LLM-facing behaviour is unchanged:
the model still sends code and gets stdout; on a kill (timeout/OOM) the call returns an
error string and the subprocess is respawned for the next call.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

DRIVER = Path(__file__).resolve().parent / "_exec_driver.py"


class SafePythonSession:
    def __init__(self, *, timeout: float = 20.0, mem_mb: int = 4096,
                 inner_timeout: float | None = None, python: str = sys.executable):
        self.timeout = timeout                      # parent wall-clock SIGKILL backstop
        self.mem_mb = mem_mb                        # RLIMIT_AS cap per subprocess
        # inner SIGALRM timeout fires first for pure-Python loops; parent kills C-ext hangs
        self.inner_timeout = inner_timeout if inner_timeout is not None else max(1.0, timeout - 5)
        self.python = python
        self._p: subprocess.Popen | None = None

    def _spawn(self) -> None:
        env = dict(os.environ)
        env["PYTHON_TOOL_MEM_BYTES"] = str(self.mem_mb * 1024 * 1024)
        env["PYTHON_TOOL_TIMEOUT"] = str(self.inner_timeout)
        self._p = subprocess.Popen(
            [self.python, "-u", str(DRIVER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, text=True, bufsize=1)

    def execute(self, code: str) -> str:
        if self._p is None or self._p.poll() is not None:
            self._spawn()
        try:
            self._p.stdin.write(json.dumps({"code": code}) + "\n")
            self._p.stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill()
            return "[sandbox died before execution; respawning next call]"
        r, _, _ = select.select([self._p.stdout], [], [], self.timeout)
        if not r:
            self._kill()
            return f"[execution killed: exceeded {self.timeout:.0f}s wall limit]"
        line = self._p.stdout.readline()
        if not line:
            self._kill()
            return "[sandbox process died (likely hit memory limit); respawning next call]"
        try:
            return json.loads(line).get("output", "")
        except Exception:  # noqa: BLE001
            return "[sandbox protocol error]"

    def _kill(self) -> None:
        if self._p is not None:
            try:
                self._p.kill()
                self._p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._p = None

    def close(self) -> None:
        self._kill()

    def __del__(self):
        self.close()
