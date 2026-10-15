import json
import os
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from filelock import Timeout

from repairbench.data import Task
from repairbench.jobs import JobStore, TERMINAL
from repairbench.service import create_app


def wait_job(client, job_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        item = client.get('/api/jobs/' + job_id).json()
        if item['status'] in TERMINAL:
            return item
        time.sleep(.02)
    raise AssertionError('Job did not terminate')


@pytest.fixture
def service(tmp_path, monkeypatch):
    task = Task('gcd', 'def gcd(a,b): return -1', {})
    monkeypatch.setattr('repairbench.jobs.load_task', lambda *a: task)
    monkeypatch.setattr('repairbench.service.load_task', lambda *a: task)
    def repair(*args, **options):
        return {'decision': 'no_visible_pass', 'accepted_candidate': None, 'attempts': [], 'ranked_attempts': []}
    monkeypatch.setattr('repairbench.jobs.repair_task', repair)
    store = JobStore(tmp_path, tmp_path/'jobs', None, SimpleNamespace(adapter_record=None), {'test_fixture': True})
    with TestClient(create_app(store), base_url='http://127.0.0.1:8018') as client:
        token = client.get('/api/config').json()['token']
        client.headers['X-Repair-Token'] = token
        yield client, store


def test_http_job_evidence_export_and_restart(service):
    client, store = service
    response = client.post('/api/jobs', json={'task': 'gcd', 'budget': 1})
    assert response.status_code == 202
    item = wait_job(client, response.json()['id'])
    assert item['status'] == 'completed' and item['result']['accepted_candidate'] is None
    assert item['original_source'] and item['identity']['test_fixture']
    assert client.get('/api/jobs/'+item['id']+'/export').json() == item
    assert json.loads((store.output/(item['id']+'.json')).read_text()) == item
    assert client.get('/api/jobs').json()[0]['id'] == item['id']
    store.close()
    reopened = JobStore(store.root, store.output, None, store.generator, {})
    assert reopened.get(item['id']) == item
    reopened.close()


@pytest.mark.parametrize('body', [{'task':'../gcd'}, {'task':'unapproved'}, {'task':'gcd','budget':5},
    {'task':'gcd','budget':True}, {'task':'gcd','budget':'4'}, {'task':'gcd','seconds':601},
    {'task':'gcd','source':'print(1)'}, {'task':'gcd','feedback':'true'}])
def test_request_validation_rejects_before_work(service, body):
    client, store = service
    assert client.post('/api/jobs', json=body).status_code == 422
    assert not store.records


def test_local_origin_token_and_body_controls(service):
    client, store = service
    assert client.get('/api/config', headers={'host':'attacker.example'}).status_code == 403
    assert client.get('/api/config', headers={'origin':'https://attacker.example'}).status_code == 403
    assert client.post('/api/jobs',json={'task':'gcd'},headers={'X-Repair-Token':'wrong'}).status_code == 403
    assert client.post('/api/jobs',content=b'x'*1025).status_code == 413
    assert client.post('/api/jobs',content=iter([b'x'*600,b'x'*600])).status_code == 413
    assert not store.records
    assert 'frame-ancestors' in client.get('/api/config').headers['content-security-policy']


def test_single_worker_cancel_wins_and_no_queued_work(service, monkeypatch):
    client, store = service
    entered, finish = threading.Event(), threading.Event()
    def repair(*args, **options):
        entered.set(); assert finish.wait(5)
        return {'decision':'accepted_visible_only','accepted_candidate':{'source':'must not accept'}}
    monkeypatch.setattr('repairbench.jobs.repair_task',repair)
    try:
        job = client.post('/api/jobs',json={'task':'gcd'}).json(); assert entered.wait(3)
        assert client.post('/api/jobs',json={'task':'gcd'}).status_code == 429
        assert client.post('/api/jobs/'+job['id']+'/cancel',json={}).json()['status']=='cancelling'
    finally:
        finish.set()
    final=wait_job(client,job['id'])
    assert final['status']=='cancelled' and final['result']['accepted_candidate'] is None
    assert len(store.records)==1 and client.get('/api/health').json()['active_job'] is None
    assert client.post('/api/jobs/'+job['id']+'/cancel',json={}).json()==final


def test_failure_and_capacity_preserve_evidence(service, monkeypatch):
    client, store = service
    store.capacity=1
    def broken(*a,**k):raise RuntimeError('isolated runtime unavailable')
    monkeypatch.setattr('repairbench.jobs.repair_task',broken)
    job=client.post('/api/jobs',json={'task':'gcd'}).json()
    final=wait_job(client,job['id'])
    assert final['status']=='failed' and final['result'] is None and 'runtime unavailable' in final['error']
    assert client.post('/api/jobs',json={'task':'gcd'}).status_code==429
    assert client.get('/api/jobs/missing').status_code==404
    assert client.post('/api/jobs/missing/cancel',json={}).status_code==404


def test_directory_exclusion_and_interrupted_recovery(tmp_path):
    generator=SimpleNamespace(adapter_record=None)
    store=JobStore(tmp_path,tmp_path/'jobs',None,generator,{})
    with pytest.raises(Timeout):JobStore(tmp_path,store.output,None,generator,{})
    store.close()
    job_id=str(uuid.uuid4())
    (store.output/(job_id+'.json')).write_text(json.dumps({'id':job_id,'status':'running','result':{'accepted_candidate':{}}}))
    reopened=JobStore(tmp_path,store.output,None,generator,{})
    assert reopened.get(job_id)['status']=='interrupted' and reopened.get(job_id)['result'] is None
    reopened.close()


def test_history_remains_newest_first_after_restart(tmp_path):
    output=tmp_path/'history';output.mkdir()
    older='ffffffff-ffff-4fff-8fff-ffffffffffff'
    newer='00000000-0000-4000-8000-000000000000'
    for job_id,date in [(older,'2026-01-01'),(newer,'2026-01-02')]:
        (output/(job_id+'.json')).write_text(json.dumps({'id':job_id,'status':'completed','created_at':date}))
    store=JobStore(tmp_path,output,None,None,{})
    assert [item['id'] for item in store.list()]==[newer,older]
    store.close()


def test_shutdown_requests_cancel_and_waits_for_cleanup(service, monkeypatch):
    client, store=service
    entered=threading.Event()
    def repair(*args,**options):
        entered.set()
        deadline=time.monotonic()+5
        while not options['cancelled']() and time.monotonic()<deadline:time.sleep(.01)
        assert options['cancelled']()
        return {'decision':'cancelled','accepted_candidate':None}
    monkeypatch.setattr('repairbench.jobs.repair_task',repair)
    job=client.post('/api/jobs',json={'task':'gcd'}).json();assert entered.wait(2)
    store.close()
    assert not store.worker.is_alive() and store.get(job['id'])['status']=='cancelled'
    assert client.post('/api/jobs',json={'task':'gcd'}).status_code==429


@pytest.mark.docker
def test_http_runs_published_candidate_only_in_isolated_runtime(tmp_path):
    from repairbench.sandbox import DockerRunner
    image=os.environ.get('REPAIR_IMAGE_FILE')
    if not image:pytest.skip('Set REPAIR_IMAGE_FILE for real service isolation integration')
    source=json.loads(Path('results/feedback-v1/tasks/gcd.json').read_text())['arms']['independent']['attempts'][0]['response']
    class Replay:
        adapter_record=None
        def generate(self,*a,**k):return source,{'recorded_response_replay':True}
    runner=DockerRunner(Path(image).read_text().strip());identity={'runtime':runner.preflight(),'recorded_response_replay':True}
    store=JobStore(Path('data/quixbugs'),tmp_path/'http',runner,Replay(),identity)
    with TestClient(create_app(store),base_url='http://127.0.0.1:8018') as client:
        client.headers['X-Repair-Token']=client.get('/api/config').json()['token']
        item=client.post('/api/jobs',json={'task':'gcd','budget':1}).json()
        result=wait_job(client,item['id'])
        assert result['result']['decision']=='accepted_with_regressions'
        assert result['result']['buggy']['status']=='failed'
        assert result['result']['regression']['status']=='passed'
        assert any(e['phase']=='generated' for e in result['events'])
        assert len([e for e in result['events'] if e['phase']=='test_result'])==5
