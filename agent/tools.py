"""
Tool implementations as LangChain @tool functions.
Used by LangGraph ToolNode for automatic dispatch.
"""

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from config import WORKSPACE_DIR

# Active session subdirectory — set by WebCodingAgent before each run
_session_subdir: str = ""


def set_session_subdir(subdir: str) -> None:
    """Set the active session subdirectory (e.g. 'abc123'). Empty string = workspace root."""
    global _session_subdir
    _session_subdir = subdir


def _workspace() -> Path:
    """Return the effective workspace path for the current session."""
    base = Path(WORKSPACE_DIR)
    if _session_subdir:
        return base / _session_subdir
    return base


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def write_file(filename: str, content: str) -> dict:
    """Write or overwrite a file in the workspace with the given content.

    Args:
        filename: Relative path inside workspace/ (e.g. 'main.py' or 'src/utils.py').
        content: Full file content to write — always provide the COMPLETE file, never a partial patch.
    """
    workspace = _workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    filepath = workspace / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    return {"success": True, "message": f"Written: {filename}", "filename": filename, "content": content}


@tool
def read_file(filename: str) -> dict:
    """Read and return the content of a file in the workspace.

    Args:
        filename: Relative path inside workspace/.
    """
    filepath = _workspace() / filename
    if not filepath.exists():
        return {"success": False, "error": f"File not found: {filename}"}
    return {"success": True, "content": filepath.read_text(encoding="utf-8"), "filename": filename}


@tool
def execute_code(filename: str) -> dict:
    """Execute a Python file that already exists in the workspace and return stdout/stderr.

    Args:
        filename: Python file to run (relative to workspace/). The file must already exist.
    """
    filepath = _workspace() / filename
    if not filepath.exists():
        return {"success": False, "error": f"File not found: {filename}"}
    try:
        proc = subprocess.run(
            ["python", str(filepath)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "success": True,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Execution timed out (30s)"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@tool
def list_files() -> dict:
    """List all files currently present in the workspace directory."""
    workspace = _workspace()
    if not workspace.exists():
        return {"success": True, "files": []}
    files = [
        str(p.relative_to(workspace)).replace("\\", "/")
        for p in workspace.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    ]
    return {"success": True, "files": sorted(files)}


@tool
def run_command(command: str) -> dict:
    """Run a whitelisted shell command (e.g. 'pip install requests').

    Args:
        command: Shell command to execute. Only pip/python inspection commands are allowed.
    """
    _ALLOWLIST = (
        "pip install ",
        "pip list",
        "pip show ",
        "pip freeze",
        "python --version",
        "python -m pip ",
    )
    if not any(command.strip().startswith(p) for p in _ALLOWLIST):
        return {
            "success": False,
            "error": f"Command not in allowlist: {command!r}. Allowed prefixes: {_ALLOWLIST}",
        }
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out (60s)"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ── Registry ──────────────────────────────────────────────────────────────────

TOOLS = [write_file, read_file, execute_code, list_files, run_command]

