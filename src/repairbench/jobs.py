"""Single-worker local jobs with durable evidence and cooperative cancellation."""
import copy
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from .data import load_task
from .feedback import repair_task
from .regressions import CASES

TERMINAL = {'completed', 'cancelled', 'failed', 'interrupted'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class BusyError(ValueError):
    pass


class JobStore:
    def __init__(self, root, output, runner, generator, identity, *, capacity=20):
        if not 1 <= capacity <= 20:
            raise ValueError('Job capacity must be between 1 and 20')
        self.root, self.output = Path(root), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.file_lock = FileLock(self.output / '.service.lock', timeout=0)
        self.file_lock.acquire()
        self.runner, self.generator, self.identity = runner, generator, identity
        self.capacity, self.records = capacity, {}
        self.mutex = threading.RLock()
        self.stop_event = threading.Event()
        self.worker = None
        self.active = None
        self.closed = False
        try:
            for path in sorted(self.output.glob('*.json')):
                if path.stat().st_size > 5_000_000 or len(self.records) >= capacity:
                    raise ValueError('Evidence directory exceeds the bounded job capacity')
                item = json.loads(path.read_text(encoding='utf-8'))
                if str(uuid.UUID(item['id'])) != path.stem:
                    raise ValueError('Invalid evidence filename')
                if item['status'] not in TERMINAL:
                    item.update(status='interrupted', finished_at=utc_now(), error='Service stopped before a final decision; no patch accepted.')
                    item['result'] = None
                self.records[item['id']] = item
                self._save(item)
        except BaseException:
            self.file_lock.release()
            raise

    def _save(self, item):
        path = self.output / (item['id'] + '.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(item, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)

    def get(self, job_id):
        with self.mutex:
            return copy.deepcopy(self.records[job_id])

    def list(self):
        with self.mutex:
            return [{k: item.get(k) for k in ('id', 'task', 'status', 'created_at', 'finished_at')}
                    for item in sorted(self.records.values(), key=lambda item: item.get('created_at', ''), reverse=True)]

    def submit(self, name, budget=4, feedback=True, seconds=300):
        if name not in CASES or type(budget) is not int or not 1 <= budget <= 4 or type(feedback) is not bool or type(seconds) is not int or not 10 <= seconds <= 600:
            raise ValueError('Unsupported curated task or repair limits')
        with self.mutex:
            if self.closed or self.active is not None:
                raise BusyError('One repair can run at a time. Wait for it to finish or cancel it.')
            if len(self.records) >= self.capacity:
                raise BusyError('This evidence directory is full. Restart with a new output directory; existing evidence is preserved.')
            task = load_task(self.root, name)
            job_id = str(uuid.uuid4())
            item = {'id': job_id, 'task': name, 'status': 'running', 'created_at': utc_now(),
                    'request': {'budget': budget, 'feedback': feedback, 'seconds': seconds},
                    'original_source': task.source, 'identity': self.identity, 'events': [], 'result': None}
            self.records[job_id] = item
            self.active = job_id
            self.stop_event.clear()
            self._save(item)
            self.worker = threading.Thread(target=self._run, args=(job_id,), name='repair-worker')
            self.worker.start()
            return copy.deepcopy(item)

    def _event(self, job_id, phase, **evidence):
        with self.mutex:
            item = self.records[job_id]
            item['events'].append({'at': utc_now(), 'phase': phase, **evidence})
            self._save(item)

    def _run(self, job_id):
        started = time.monotonic()
        owner = self
        class ObservedRunner:
            def run(self, files, test, **options):
                owner._event(job_id, 'testing', test=test, reference=options.get('reference', False))
                result = owner.runner.run(files, test, **options)
                owner._event(job_id, 'test_result', test=test, reference=options.get('reference', False), evidence=result)
                return result
        class ObservedGenerator:
            adapter_record = owner.generator.adapter_record
            def generate(self, *args, **options):
                owner._event(job_id, 'generating')
                response, stats = owner.generator.generate(*args, **options)
                owner._event(job_id, 'generated', response=response, generation=stats)
                return response, stats
        try:
            item = self.get(job_id)
            request = item['request']
            result = repair_task(self.root, item['task'], ObservedRunner(), ObservedGenerator(),
                                 request['budget'], feedback=request['feedback'],
                                 max_seconds=request['seconds'], cancelled=self.stop_event.is_set)
            with self.mutex:
                # A cancellation acknowledged after the workflow's final check still wins.
                if self.stop_event.is_set():
                    result.update(decision='cancelled', accepted_candidate=None)
                self.records[job_id].update(result=result, status='cancelled' if result['decision'] == 'cancelled' else 'completed')
        except Exception as error:
            with self.mutex:
                self.records[job_id].update(status='failed', result=None,
                    error=f'{type(error).__name__}: {str(error)[:2000]}')
        finally:
            with self.mutex:
                item = self.records[job_id]
                item.update(finished_at=utc_now(), seconds=round(time.monotonic()-started, 6))
                try:
                    self._save(item)
                finally:
                    self.active = None

    def cancel(self, job_id):
        with self.mutex:
            item = self.records[job_id]
            if item['status'] not in TERMINAL:
                self.stop_event.set()
                item.update(status='cancelling', cancel_requested_at=utc_now())
                self._save(item)
            return copy.deepcopy(item)

    def close(self):
        with self.mutex:
            self.closed = True
            if self.active:
                self.cancel(self.active)
            worker = self.worker
        if worker:
            worker.join()  # Keep the lock until the bounded active operation cleans up.
        self.file_lock.release()
