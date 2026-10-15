"""Real-model HTTP integration probe; requires a running local workbench."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def verify(port, output):
    if output.exists():
        raise ValueError('Choose a new evidence file')
    base = f'http://127.0.0.1:{port}'
    token = None
    def request(path, body=None):
        raw = None if body is None else json.dumps(body).encode()
        headers = {} if token is None else {'X-Repair-Token': token, 'Content-Type': 'application/json'}
        with urllib.request.urlopen(urllib.request.Request(base+path, data=raw, headers=headers), timeout=10) as response:
            return json.load(response)
    config = request('/api/config')
    token = config.pop('token')
    def wait(job_id, predicate):
        deadline = time.monotonic()+240
        while time.monotonic()<deadline:
            item = request('/api/jobs/'+job_id)
            if predicate(item):
                return item
            time.sleep(.1)
        request('/api/jobs/'+job_id+'/cancel', {})
        raise TimeoutError('Integration wait expired; cancellation requested')
    terminal = lambda item: item['status'] in {'completed', 'cancelled', 'failed'}
    first = request('/api/jobs', {'task':'gcd', 'budget':2, 'feedback':True, 'seconds':120})
    repaired = wait(first['id'], terminal)
    second = request('/api/jobs', {'task':'quicksort', 'budget':4, 'feedback':True, 'seconds':120})
    overlap = None
    try:
        request('/api/jobs', {'task':'gcd'})
    except urllib.error.HTTPError as error:
        overlap = error.code
    generating = wait(second['id'], lambda item: any(e['phase']=='generating' for e in item['events']) or terminal(item))
    cancellation_started = time.monotonic()
    request('/api/jobs/'+second['id']+'/cancel', {})
    cancelled = wait(second['id'], terminal)
    report = {'kind':'Real local model integration, not a model-success benchmark',
              'config':config, 'fresh_job':repaired, 'cancelled_job':cancelled,
              'cancel_requested_while_running':generating['status']=='running',
              'cancel_settle_seconds':round(time.monotonic()-cancellation_started,6),
              'overlap_status':overlap, 'active_job_after':request('/api/health')['active_job']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    assert repaired['result']['decision']=='accepted_with_regressions', 'Fresh probe did not pass; inspect saved evidence'
    assert repaired['result']['buggy']['status']=='failed'
    assert len(repaired['result']['attempts'])==2
    assert overlap==429 and report['cancel_requested_while_running']
    assert cancelled['status']=='cancelled' and cancelled['result']['accepted_candidate'] is None
    assert report['active_job_after'] is None
    return {key:report[key] for key in ['kind','cancel_settle_seconds','overlap_status','active_job_after']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8018)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.port,args.output),indent=2))
