"""Xuanzhi Streamlit entry point.

Run from the repo root:

    streamlit run streamlit_app.py

This is a thin shim: it puts ``src/`` on the path so the ``xuanzhi``
package imports cleanly without an editable install, then hands off to
``xuanzhi.app.main``.

Note: we use runpy.run_path() instead of a plain import so that
main.py's module-level routing code re-executes on every Streamlit
re-run. A bare ``import xuanzhi.app.main`` is cached by Python's module
system and becomes a no-op after the first run, which makes every
radio-button click render a blank page.
"""

import runpy
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

runpy.run_path(str(_SRC / "xuanzhi" / "app" / "main.py"))
