import json
import pytest
from datetime import datetime
from flask import url_for

import app as smart_app

@pytest.fixture
def client():
    smart_app.app.config['TESTING'] = True
    # 避免 session 依赖环境变量
    smart_app.app.config['SECRET_KEY'] = 'test-secret'
    with smart_app.app.test_client() as client:
        yield client

# --- helper 模拟数据 ---
class DummyCursor:
    def __init__(self, rows):
        self._rows = rows
    def execute(self, *_):
        pass
    def fetchall(self):
        return self._rows
    def close(self):
        pass

class DummyConn:
    def __init__(self, rows):
        self._rows = rows
    def cursor(self, *args, **kwargs):
        return DummyCursor(self._rows)
    def close(self):
        pass

# --- 测试 /api/login ---
def test_login_success(client):
    ts = int(datetime.now().timestamp()*1000)
    resp = client.post(
        '/api/login',
        json={'username': 'admin', 'password': 'admin', 'timestamp': ts}
    )
    data = resp.get_json()
    assert resp.status_code == 200
    assert data['success'] is True
    assert 'session' in resp.headers.get('Set-Cookie')

def test_login_expired_timestamp(client):
    ts = 0  # 明显过期
    resp = client.post(
        '/api/login',
        json={'username': 'admin', 'password': 'admin', 'timestamp': ts}
    )
    assert resp.status_code == 401
    assert resp.get_json()['message'].startswith('请求已过期')

def test_login_failure(client):
    ts = int(datetime.now().timestamp()*1000)
    resp = client.post(
        '/api/login',
        json={'username': 'foo', 'password': 'bar', 'timestamp': ts}
    )
    assert resp.status_code == 401
    assert resp.get_json()['success'] is False

# --- 测试 /api/realtime-data ---
@pytest.fixture(autouse=True)
def stub_connect_db_real(monkeypatch):
    # 模拟实时数据表返回两行
    sample = [
        {'timestamp': datetime(2025,5,10,10,0,0), 'water_level':1, 'temperature':2, 'humidity':3,'windpower':4}
    ]
    monkeypatch.setattr(smart_app, 'connect_db', lambda: DummyConn(sample))
    yield

def login_and_get_cookies(client):
    ts = int(datetime.now().timestamp()*1000)
    login = client.post('/api/login', json={'username':'admin','password':'admin','timestamp':ts})
    return dict(login.headers.get_all('Set-Cookie'))

def test_realtime_data(client):
    cookies = login_and_get_cookies(client)
    resp = client.get('/api/realtime-data', headers={'Cookie': cookies})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert data[0]['water_level'] == 1
    # timestamp 应为字符串
    assert isinstance(data[0]['timestamp'], str)

# --- 测试 /api/history-data ---
@pytest.fixture
def stub_connect_db_hist(monkeypatch):
    sample = [
        {'timestamp': datetime(2025,5,9,9,0,0), 'water_level':5, 'temperature':6, 'humidity':7, 'windpower':8, 'winddirection':90, 'rains':0}
    ]
    monkeypatch.setattr(smart_app, 'connect_db', lambda: DummyConn(sample))
    yield

def test_history_missing_params(client, stub_connect_db_hist):
    cookies = login_and_get_cookies(client)
    resp = client.get('/api/history-data', headers={'Cookie': cookies})
    assert resp.status_code == 400
    assert '缺少开始或结束日期' in resp.get_json()['error']

def test_history_success(client, stub_connect_db_hist):
    cookies = login_and_get_cookies(client)
    resp = client.get(
        '/api/history-data?start=2025-05-01&end=2025-05-11',
        headers={'Cookie': cookies}
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data[0]['rains'] == 0
    assert 'timestamp' in data[0]

# --- 测试 /api/deepseek ---
import requests

@pytest.fixture
def stub_deepseek(monkeypatch):
    class DummyResp:
        status_code = 200
        def json(self): return {'response': '模拟回答'}
    # 模拟数据库返回空列表
    monkeypatch.setattr(smart_app, 'connect_db', lambda: DummyConn([]))
    monkeypatch.setenv('DEEPSEEK_URL', 'http://fake-deepseek')
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: DummyResp())
    yield

def test_deepseek_no_question(client, stub_deepseek):
    cookies = login_and_get_cookies(client)
    resp = client.post('/api/deepseek', json={}, headers={'Cookie': cookies})
    assert resp.status_code == 400
    assert '请提供问题' in resp.get_json()['error']

def test_deepseek_success(client, stub_deepseek):
    cookies = login_and_get_cookies(client)
    resp = client.post(
        '/api/deepseek',
        json={'question': '水库水位多少？'},
        headers={'Cookie': cookies}
    )
    assert resp.status_code == 200
    assert resp.get_json()['response'] == '模拟回答'
