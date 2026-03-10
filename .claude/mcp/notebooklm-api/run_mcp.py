#!/usr/bin/env python3
"""Wrapper to run notebooklm-mcp with the correct site-packages.

Handles Docker (Python 3.13) vs WSL2 (Python 3.11) by finding
the matching site-packages for the running Python version.
"""
import subprocess
import sys
import os
from pathlib import Path

venv_lib = Path(__file__).parent / ".venv" / "lib"

# Check if current Python matches a venv site-packages
major_minor = f"python{sys.version_info.major}.{sys.version_info.minor}"
matching_sp = venv_lib / major_minor / "site-packages"

if matching_sp.exists():
    # Current Python matches venv - run directly
    sys.path.insert(0, str(matching_sp))
    from notebooklm_tools.mcp.server import main
    sys.exit(main())
else:
    # Find what Python version the venv has
    for sp in sorted(venv_lib.glob("python*/site-packages")):
        venv_python_ver = sp.parent.name  # e.g. "python3.11"
        # Try to find that Python on the system
        for candidate in [
            f"/usr/bin/{venv_python_ver}",
            str(Path.home() / ".local" / "bin" / venv_python_ver),
            str(Path.home() / ".local" / "share" / "uv" / "python" / f"cpython-{venv_python_ver.replace('python', '')}*" / "bin" / venv_python_ver),
        ]:
            # Handle glob in path
            from glob import glob
            matches = glob(candidate)
            if matches and Path(matches[0]).exists():
                # Re-exec with the correct Python, using venv site-packages
                env = os.environ.copy()
                env["PYTHONPATH"] = str(sp)
                os.execve(matches[0], [matches[0], __file__], env)
        break

    print(f"Error: Could not find {venv_python_ver} interpreter", file=sys.stderr)
    sys.exit(1)
