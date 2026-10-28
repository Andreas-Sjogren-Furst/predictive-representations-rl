"""Run the Streamlit shell headlessly against real run directories."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from test_panels import add_representations, populate

from aeris.dashboard import APP_PATH
from conftest import synthetic_suite_report


def test_app_renders_built_in_layouts_and_saves_edits(tmp_path, four_rooms_run, monkeypatch):
    add_representations(four_rooms_run("first"))
    four_rooms_run("first").write_report(synthetic_suite_report())
    populate(four_rooms_run("first"))
    populate(four_rooms_run("second"))
    monkeypatch.setenv("AERIS_RUNS_ROOT", str(tmp_path))
    monkeypatch.setenv("AERIS_DASHBOARDS", str(tmp_path / "layouts"))
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    assert at.get("plotly_chart"), "the run overview layout draws charts"

    at.sidebar.selectbox[0].set_value("Zero-shot suite").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 8, (
        "suite charts: scores, distribution, goals, lowest-scoring fates, 4 task policies"
    )

    at.sidebar.selectbox[0].set_value("Representations").run()  # a second switch in a row
    assert not at.exception
    charts = at.get("plotly_chart")
    assert len(charts) == 6, "every representation panel draws"
    assert all(chart.key.startswith("Representations|") for chart in charts)

    at.sidebar.multiselect[0].set_value(["first", "second"]).run()
    at.sidebar.selectbox[0].set_value("Compare runs").run()
    assert not at.exception
    # The group summary groups by experiment name, even though the previous layout had a
    # different panel at the same position (widget state must not leak between layouts).
    assert "name=fb_demo (n=2)" in at.dataframe[0].value.columns

    at.sidebar.toggle[0].set_value(True).run()  # edit mode
    [b for b in at.button if b.label == "✕ remove"][0].click().run()
    [t for t in at.sidebar.text_input if t.label == "Save as"][0].set_value("Mine").run()
    [b for b in at.sidebar.button if b.label == "Save"][0].click().run()
    assert not at.exception
    assert (tmp_path / "layouts" / "mine.json").is_file()
