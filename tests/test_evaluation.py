from pathlib import Path

import pytest

from repairbench.data import Task
from repairbench.evaluate import summarize, task_result


class RecordedRunner:
    def __init__(self, records):
        self.records = iter(records)
        self.payloads = []

    def run(self, files, test_file, *, reference=False):
        self.payloads.append((files, reference))
        return next(self.records)


def record(status, passed=0, skipped=0):
    return {'status': status, 'seconds': 0.1, 'evidence': {'passed': passed, 'skipped': skipped}}


@pytest.fixture
def task(monkeypatch):
    def load(root, name, *, reference=False):
        files = {'python_programs/example.py': b'def f(x): return x < 1\n'}
        if reference:
            files['correct_python_programs/example.py'] = b'ORACLE'
        return Task(name, files['python_programs/example.py'].decode(), files)
    monkeypatch.setattr('repairbench.evaluate.load_task', load)


def test_acceptance_requires_full_test_count_and_no_oracle(task):
    runner = RecordedRunner([record('passed', 3), record('failed'), record('passed', 2), record('passed', 3)])
    result = task_result(Path('.'), 'example', runner, 2)
    assert result['accepted_candidate']['number'] == 2
    assert len(result['attempts']) == 2
    assert all('correct_python_programs/example.py' not in files for files, reference in runner.payloads if not reference)
    summary = summarize([result], 0.4, 2)
    assert summary['resolved_visible_tests'] == 1
    assert summary['no_change_resolved'] == 0


@pytest.mark.parametrize('fixed,buggy,reason', [('failed','failed','reference_not_passing'), ('passed','passed','bug_not_exposed'), ('passed','runner_error','baseline_runtime_error')])
def test_invalid_tasks_are_excluded_before_search(task, fixed, buggy, reason):
    runner = RecordedRunner([record(fixed, 3), record(buggy, 3)])
    result = task_result(Path('.'), 'example', runner, 2)
    assert result['eligibility'] == reason
    assert result['attempts'] == []
    assert summarize([result], 0.2, 2)['resolved_fraction'] is None
