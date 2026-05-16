"""Xuanzhi Streamlit entry point.

Run from the repo root:

    streamlit run streamlit_app.py

This is a thin shim: it puts ``src/`` on the path so the ``xuanzhi``
package imports cleanly without an editable install, then hands off to
``xuanzhi.app.main``.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Importing the module runs the app (Streamlit executes top-to-bottom).
import xuanzhi.app.main  # noqa: E402,F401
