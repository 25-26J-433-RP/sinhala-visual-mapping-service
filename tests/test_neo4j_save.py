import json
from unittest.mock import Mock, MagicMock

import pytest

from app import app as flask_app


def test_generate_mindmap_saves_to_neo4j(monkeypatch):
    """Ensure that when a neo4j driver is present, the app attempts to save nodes and edges."""

    # Prepare a fake driver whose .session(...) returns a context manager
    driver_mock = Mock()
    cm = MagicMock()
    session_mock = Mock()
    # context manager __enter__ returns our session mock
    cm.__enter__.return_value = session_mock
    driver_mock.session.return_value = cm

    # Inject the mock driver into the app module
    import app as app_module
    app_module.neo4j_driver = driver_mock

    client = flask_app.test_client()

    payload = {
        "essay_id": "test-1",
        "text": "පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි."
    }

    resp = client.post('/api/mindmap/generate', json=payload)
    assert resp.status_code == 200

    data = resp.get_json()
    assert data.get('success') is True
    assert data.get('essay_id') == 'test-1'

    # Should have opened a session
    assert driver_mock.session.called, "neo4j_driver.session was not called"

    # Determine expected number of save calls: nodes + edges
    nodes = data['data'].get('nodes', [])
    edges = data['data'].get('edges', [])

    expected_calls = len(nodes) + len(edges)

    # execute_write should be called once per node/edge
    assert session_mock.execute_write.call_count == expected_calls
import json

import pytest

from app import app as flask_app


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


def test_generate_saves_to_neo4j(monkeypatch):
    """Ensure that when `neo4j_driver` is present the app attempts to save nodes and edges."""

    # Prepare a mock session that records execute_write calls
    mock_session = MockSession()

    # Patch the app's neo4j_driver to our mock driver
    monkeypatch.setattr('app.neo4j_driver', MockDriver(mock_session))

    client = flask_app.test_client()

    payload = {
        "essay_id": "test-essay-1",
        "text": "පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි. එය ක්‍රියාත්මක වේ."
    }

    resp = client.post(
        '/api/mindmap/generate',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert data.get('essay_id') == 'test-essay-1'

    # The mock_session should have recorded multiple calls (nodes and edges)
    assert len(mock_session.calls) >= 1

    # Check that at least one node-like call and optionally edge-like call were attempted
    node_call_found = False
    edge_call_found = False
    for fname, arg in mock_session.calls:
        # node merges pass a dict with 'id' and 'label'
        if isinstance(arg, dict) and 'label' in arg and 'id' in arg:
            node_call_found = True
        # edge merges pass a dict with 'source' and 'target'
        if isinstance(arg, dict) and 'source' in arg and 'target' in arg:
            edge_call_found = True

    assert node_call_found, 'No node merge was attempted'
    # edges may be empty for short texts; don't fail if no edges found, only log
