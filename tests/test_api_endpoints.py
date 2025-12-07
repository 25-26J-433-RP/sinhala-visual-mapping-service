import os
import sys
import json
import importlib.util

import pytest

# If Flask isn't installed in the environment running the tests, skip this module.
try:
    import flask  # noqa: F401
except Exception:
    pytest.skip("Flask is not installed; skipping endpoint tests.", allow_module_level=True)

# Load the app module from the project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import app as flask_app


def test_health_check():
    client = flask_app.test_client()
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('status') == 'healthy'


def test_generate_direct_text(monkeypatch):
    client = flask_app.test_client()

    payload = {
        'essay_id': 'endpoint-test-1',
        'text': 'ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. එය සුන්දරයි.'
    }

    # Ensure neo4j_driver is not set to avoid DB calls
    import app as app_module
    monkeypatch.setattr(app_module, 'neo4j_driver', None, raising=False)

    resp = client.post('/api/mindmap/generate', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert data.get('essay_id') == 'endpoint-test-1'
    assert 'data' in data and 'nodes' in data['data']


def test_generate_external_api(monkeypatch):
    client = flask_app.test_client()

    # Mock requests.get to return a response with cleaned_text
    class MockResp:
        status_code = 200

        def json(self):
            return {'cleaned_text': 'පරිගණකය යනු උපකරණයකි.'}

    import requests

    monkeypatch.setattr(requests, 'get', lambda *a, **k: MockResp())

    payload = {'external_api_url': 'http://example.test/cleaned'}

    # Ensure neo4j_driver disabled
    import app as app_module
    monkeypatch.setattr(app_module, 'neo4j_driver', None, raising=False)

    resp = client.post('/api/mindmap/generate', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert 'data' in data and 'nodes' in data['data']


def test_batch_endpoint(monkeypatch):
    client = flask_app.test_client()

    payload = {
        'texts': [
            {'essay_id': 'b1', 'text': 'පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි.'},
            {'essay_id': 'b2', 'text': 'ශ්‍රී ලංකාව සංචාරක ගමනාන්තයකි.'}
        ]
    }

    import app as app_module
    monkeypatch.setattr(app_module, 'neo4j_driver', None, raising=False)

    resp = client.post('/api/mindmap/batch', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert isinstance(data.get('data'), list)
    assert len(data['data']) == 2


def test_get_mindmap_by_essay_no_neo4j():
    client = flask_app.test_client()

    # Ensure neo4j_driver is None
    import app as app_module
    app_module.neo4j_driver = None

    resp = client.get('/api/mindmap/essay/test-not-configured')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data.get('success') is False
