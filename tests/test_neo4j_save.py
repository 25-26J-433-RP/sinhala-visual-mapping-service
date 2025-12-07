import os
import sys
import json
import importlib.util

import pytest

# If Flask isn't installed in the environment running the tests, skip this module.
try:
    import flask  # noqa: F401
except Exception:
    pytest.skip("Flask is not installed; skipping neo4j save tests.", allow_module_level=True)


# Load the app module from the project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

APP_PATH = os.path.join(ROOT, 'app.py')
spec = importlib.util.spec_from_file_location('app_module', APP_PATH)
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
flask_app = app_module.app


class MockSession:
    def __init__(self):
        self.calls = []

    def execute_write(self, fn, arg):
        # record the argument passed to the write function
        self.calls.append((getattr(fn, '__name__', repr(fn)), arg))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockDriver:
    def __init__(self, session_obj):
        self._session = session_obj

    def session(self, database=None):
        return self._session


def test_generate_saves_to_neo4j_clean(monkeypatch):
    """Ensure that when `neo4j_driver` is present the app attempts to save nodes and edges."""

    mock_session = MockSession()
    mock_driver = MockDriver(mock_session)

    # Patch the driver on the app module
    monkeypatch.setattr(app_module, 'neo4j_driver', mock_driver, raising=False)

    client = flask_app.test_client()

    payload = {
        'essay_id': 'test-essay-clean',
        'text': 'පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි. එය ක්‍රියාත්මක වේ.'
    }

    resp = client.post('/api/mindmap/generate', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert data.get('essay_id') == 'test-essay-clean'

    # At least one write call should have been recorded
    assert len(mock_session.calls) >= 1

    # Confirm at least one node merge was attempted
    assert any(isinstance(arg, dict) and 'label' in arg and 'id' in arg for _, arg in mock_session.calls)
