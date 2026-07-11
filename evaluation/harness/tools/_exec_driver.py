"""Subprocess driver: memory-capped, import-restricted Python tool executor.

Sets RLIMIT_AS from PYTHON_TOOL_MEM_BYTES *before* importing heavy libs, then runs a
SecureLightweightPythonSession (blocks file/net/subprocess) reading {"code": ...} lines
on stdin and writing {"output": ...} lines on stdout. Because this is the subprocess's
*main* thread, the session's SIGALRM timeout works again; the RLIMIT_AS cap turns a
runaway allocation into a caught MemoryError instead of a system OOM. The parent
(SafePythonSession) adds a wall-clock SIGKILL backstop. LLM-facing behaviour is identical
to the in-process session.
"""
import json
import os
import resource
import sys

_mem = int(os.environ.get("PYTHON_TOOL_MEM_BYTES", str(4 * 1024 ** 3)))
try:
    resource.setrlimit(resource.RLIMIT_AS, (_mem, _mem))
except (ValueError, OSError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from python_tool import SecureLightweightPythonSession  # noqa: E402

_sess = SecureLightweightPythonSession(timeout=float(os.environ.get("PYTHON_TOOL_TIMEOUT", "15")))

for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _req = json.loads(_line)
    except Exception:
        continue
    _code = _req.get("code")
    if _code is None:
        break
    try:
        _out = _sess.execute(_code)
    except MemoryError:
        _out = "[MemoryError: exceeded sandbox memory limit]"
    except BaseException as _e:  # noqa: BLE001 - incl. KeyboardInterrupt from SIGALRM
        _out = f"[sandbox error: {type(_e).__name__}: {_e}]"
    try:
        sys.stdout.write(json.dumps({"output": _out}) + "\n")
        sys.stdout.flush()
    except Exception:
        break
