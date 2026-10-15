"""Browser-test fixture only; never used by the actual model service."""
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import uvicorn
import repairbench.jobs as jobs
import repairbench.service as service
from repairbench.data import Task


def fixture_repair(*args, **options):
    for _ in range(20):
        if options['cancelled']():
            return {'decision':'cancelled','attempts':[{'number':1,'visible_passed':False,'rejection':'execution_not_started_cancelled','edit_size':{'token_edits':2,'changed_lines':2}}],'ranked_attempts':[],'accepted_candidate':None}
        time.sleep(.05)
    run={'status':'failed','evidence':{'passed':0,'failed':1,'errors':0,'skipped':0,'output':'Fixture failure <script>window.injected=true</script>'}}
    return {'decision':'no_visible_pass','eligibility':'eligible','buggy':run,'attempts':[{'number':1,'visible_passed':False,'run':run,'rejection':'fixture_rejected','response':'<img src=x onerror="window.injected=true">'}],
            'ranked_attempts':[1],'accepted_candidate':None,'selected_candidate':None}


if __name__=='__main__':
    jobs.load_task=service.load_task=lambda root,name:Task(name,'def gcd(a,b): return -1\n',{})
    jobs.repair_task=fixture_repair
    with tempfile.TemporaryDirectory(prefix='repair-ui-') as directory:
        store=jobs.JobStore(Path(directory),Path(directory)/'jobs',None,SimpleNamespace(adapter_record=None),{'browser_test_fixture':True})
        uvicorn.run(service.create_app(store,port=8019),host='127.0.0.1',port=8019,access_log=False)
