"""Project-root conftest.

Adds the ``backend/`` directory to ``sys.path`` so that test files
(and the application itself) can import ``apps.*`` regardless of
which directory pytest is launched from.
"""

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent / "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
