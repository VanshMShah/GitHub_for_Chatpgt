import importlib
import time
from types import SimpleNamespace
from fastapi.testclient import TestClient
from src.ats.agent import BackgroundAgent
from src.ats.jobs import JobManager

def setup_api(monkeypatch,engine):
    api=importlib.import_module('src.ats.api')
    monkeypatch.setattr(api,'store',engine.store);monkeypatch.setattr(api,'engine',engine)
    monkeypatch.setattr(api,'agent',BackgroundAgent(engine));monkeypatch.setattr(api,'jobs',JobManager(engine.store))
    monkeypatch.setattr(api,'settings',SimpleNamespace(**{**api.settings.__dict__,'auto_start_agent':False,'api_token':''}))
    monkeypatch.setattr(api,'task',None);return api

# Share the behavior-test engine fixture explicitly.
from test_completion import engine

def test_api_reports_empty_states_and_serves_assets(monkeypatch,engine):
    api=setup_api(monkeypatch,engine)
    with TestClient(api.app) as client:
        for endpoint in ['/','/assets/app.js','/assets/style.css','/api/status','/api/forecasts','/api/performance','/api/models','/api/paper','/api/settings','/api/diagnostics']:
            assert client.get(endpoint).status_code==200,endpoint
        assert client.get('/api/forecasts').json()==[]
        assert client.get('/api/paper').json()['mode']=='PAPER_ONLY'
        assert client.get('/api/performance').json()['validated_profitability'] is False
        assert client.post('/api/paper/start').status_code==400

def test_api_rejects_cross_origin_and_requires_optional_token(monkeypatch,engine):
    api=setup_api(monkeypatch,engine)
    with TestClient(api.app) as client:
        assert client.post('/api/risk',json={'level':'high'},headers={'Origin':'https://unrelated.example'}).status_code==403
        monkeypatch.setattr(api,'settings',SimpleNamespace(**{**api.settings.__dict__,'api_token':'local-secret'}))
        assert client.get('/api/status').status_code==401
        assert client.get('/api/settings',headers={'Authorization':'Bearer local-secret'}).status_code==200
        assert 'local-secret' not in client.get('/api/settings',headers={'Authorization':'Bearer local-secret'}).text

def test_job_is_nonblocking_and_finishes_in_subprocess(monkeypatch,engine):
    api=setup_api(monkeypatch,engine)
    with TestClient(api.app) as client:
        response=client.post('/api/jobs',json={'kind':'settle'})
        assert response.status_code==202
        for _ in range(150):
            job=client.get('/api/jobs/current').json()
            if job.get('status') in ('SUCCEEDED','FAILED'):break
            time.sleep(.1)
        assert job['status']=='SUCCEEDED',job
        assert job['result']==0

def test_settings_export_backup_and_kill_switch(monkeypatch,engine,tmp_path):
    api=setup_api(monkeypatch,engine)
    monkeypatch.setenv('ATS_PAPER_KILL_SWITCH',str(tmp_path/'kill'))
    with TestClient(api.app) as client:
        assert client.post('/api/settings',json={'risk_level':'low','default_view':'history'}).status_code==200
        assert engine.store.get_state('preferences')['default_view']=='history'
        assert client.post('/api/paper/kill',json={'halt':True}).json()['halted']
        assert client.get('/api/paper').json()['halted']
        assert not client.post('/api/paper/kill',json={'halt':False}).json()['halted']
        assert client.get('/api/export/forecasts').status_code==200
        assert client.post('/api/backup').status_code==200
