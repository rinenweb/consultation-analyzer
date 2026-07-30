import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeStreamlitWidget:
    """Stand-in for whatever st.progress()/st.empty() return."""

    def progress(self, *args, **kwargs):
        return self

    def empty(self, *args, **kwargs):
        return self

    def write(self, *args, **kwargs):
        return self


class FakeResponse:
    """Stand-in for requests.Response, covering the .text/.json()/.headers used in this project."""

    def __init__(self, text: str = "", json_data=None, headers: dict = None):
        self.text = text
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


@pytest.fixture
def fake_st(monkeypatch):
    """
    Patches the `st` module used inside analysis_utils and new_opengov_api so that
    st.session_state.abort / st.progress() / st.empty() work without a real
    Streamlit script-run context.
    """
    import analysis_utils
    import new_opengov_api

    session_state = SimpleNamespace(abort=False)

    for module in (analysis_utils, new_opengov_api):
        monkeypatch.setattr(module.st, "session_state", session_state, raising=False)
        monkeypatch.setattr(module.st, "progress", lambda *a, **k: FakeStreamlitWidget(), raising=False)
        monkeypatch.setattr(module.st, "empty", lambda *a, **k: FakeStreamlitWidget(), raising=False)

    return session_state
