"""UI smoke test using Streamlit's AppTest with a fake backend (no CLIP download)."""
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    def __init__(self, folder):
        self.filenames = [f"{i}.jpg" for i in range(8)]
        for n in self.filenames:
            Image.new("RGB", (40, 40), "green").save(folder / n)
        self.folder = folder

    def search(self, query, top_k=5):
        if not query.strip():
            raise ValueError("Query must be a non-empty string")
        return [{"rank": i + 1, "filename": self.filenames[i],
                 "path": str(self.folder / self.filenames[i]), "score": 0.3 - i * 0.01}
                for i in range(min(top_k, len(self.filenames)))]


@pytest.fixture
def at(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    fake_app = types.ModuleType("app")
    fake_app.create_engine = lambda: FakeEngine(tmp_path)
    monkeypatch.setitem(sys.modules, "app", fake_app)
    return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()


def test_initial_state_shows_prompt(at):
    assert not at.exception
    assert any("Type a query" in i.value for i in at.info)


def test_query_renders_results(at):
    at.text_input(key="query").set_value("a dog").run()
    assert not at.exception
    assert len(at.get("image")) == 5


def test_topk_slider_changes_count(at):
    at.text_input(key="query").set_value("a dog").run()
    at.slider[0].set_value(8).run()
    assert len(at.get("image")) == 8


def test_example_button_fills_query(at):
    at.button[0].click().run()
    assert at.text_input(key="query").value == "a dog playing in the park"
    assert len(at.get("image")) == 5