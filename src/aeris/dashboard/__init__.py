"""Interactive run dashboard (optional 'dashboard' extra).

Panels in ``panels/`` turn typed run data into Plotly figures and tables and
never import Streamlit; ``app.py`` is the only Streamlit code, so the shell can
be replaced without touching the panels. Start it with ``aeris dashboard``.
"""

from pathlib import Path

APP_PATH = Path(__file__).with_name("app.py")
