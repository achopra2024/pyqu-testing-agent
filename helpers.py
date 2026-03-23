"""Low-level utilities: venv paths, subprocess runner, dependency detection."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import List, Optional

# ── Platform helpers ─────────────────────────────────────────────────────────


def is_windows() -> bool:
    return platform.system() == "Windows"


def venv_python(venv_path: Path) -> Path:
    if is_windows():
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def venv_pip(venv_path: Path) -> List[str]:
    """Return the pip command as a list.  Uses ``python -m pip`` on Windows
    to avoid WinError 5 when pip tries to overwrite its own .exe."""
    python = venv_python(venv_path)
    return [str(python), "-m", "pip"]


def venv_executable(venv_path: Path, name: str) -> Path:
    if is_windows():
        return venv_path / "Scripts" / f"{name}.exe"
    return venv_path / "bin" / name


# ── Subprocess runner ────────────────────────────────────────────────────────


def run(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output in real-time.

    When *capture* is True, stdout/stderr are captured and returned on
    the CompletedProcess object instead of being streamed to the console.
    """
    if not capture:
        print(f"\n{'-' * 60}")
        print(f"  >> {' '.join(str(c) for c in cmd)}")
        print(f"{'-' * 60}\n")
    result = subprocess.run(
        cmd, cwd=cwd, env=env, text=True,
        capture_output=capture,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}"
        )
    return result


# ── Dependency auto-detection ────────────────────────────────────────────────

_KNOWN_DEPENDENCY_FILES = [
    "requirements.txt",
    "requirments.txt",
    "requirements-dev.txt",
    "requirements_dev.txt",
    "dev-requirements.txt",
]


def find_dependency_sources(project_path: Path) -> dict:
    """Scan a project directory for common dependency manifests.

    Returns a dict with keys:
        req_files  – list of requirements*.txt paths found
        setup_py   – Path or None
        pyproject  – Path or None
        setup_cfg  – Path or None
    """
    result = {"req_files": [], "setup_py": None, "pyproject": None, "setup_cfg": None}

    for name in _KNOWN_DEPENDENCY_FILES:
        p = project_path / name
        if p.is_file():
            result["req_files"].append(p)

    for name, key in [
        ("setup.py", "setup_py"),
        ("pyproject.toml", "pyproject"),
        ("setup.cfg", "setup_cfg"),
    ]:
        p = project_path / name
        if p.is_file():
            result[key] = p

    return result
