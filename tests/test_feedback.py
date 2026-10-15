import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from repairbench.data import Task, digest, load_task
from repairbench.feedback import edit_size, repair_task, visible_feedback
from repairbench.regressions import CASES, run_regressions
from repairbench.sandbox import DockerRunner, RuntimeUnavailable


def result(passed=True, count=1):
    return {'status': 'passed' if passed else 'failed', 'seconds': 0.1,
            'evidence': {'passed': count if passed else 0, 'failed': 0 if passed else 1, 'errors': 0, 'skipped': 0, 'output': 'visible result'}}


class Generator:
    adapter_record = {'verified': True}
    def __init__(self, sources):
        self.sources, self.calls = sources, []
    def generate(self, task, context, seed, sampled, **options):
        self.calls.append((task.source, context, seed, sampled, options))
        return self.sources[len(self.calls) - 1], {'seconds': 0.1, 'peak_gpu_allocated_bytes': 0}


@pytest.fixture
def fixture_task(monkeypatch):
    task = Task('sample', 'def f(x): return x - 1\n', {'python_programs/sample.py': b'def f(x): return x - 1\n'})
    monkeypatch.setattr('repairbench.feedback.load_task', lambda *a, **k: task)
    return task


class Runner:
    def __init__(self): self.calls = []
    def run(self, files, test, reference=False):
        self.calls.append((dict(files), test, reference))
        return result(reference or b'return x + 1' in next(iter(files.values())))


def block(source):
    return '```python\n' + source + '\n```'


def test_feedback_budget_rejections_duplicates_and_ranking(fixture_task):
    generator = Generator(['invalid', block('def f(x): return x + 1'), block('def f(x): return x + 1'), block('def f(x):\n    unused = 100\n    return x + 1')])
    runner = Runner()
    report = repair_task(Path('.'), 'sample', runner, generator, budget=4)
    assert len(generator.calls) == 4 and len(report['attempts']) == 4
    assert report['attempts'][0]['rejection'] and report['attempts'][2]['rejection'] == 'duplicate_candidate'
    assert report['accepted_candidate']['number'] == 2
    assert report['decision'] == 'accepted_visible_only'
    assert report['ranked_attempts'] == [2, 4]
    assert len(runner.calls) == 4  # Reference, buggy, two unique candidates.
    assert generator.calls[0][-1]['feedback'] is None
    assert generator.calls[1][-1]['feedback']['rejection']
    assert generator.calls[2][-1]['feedback']['passed'] == 1
    assert all(call[0] == fixture_task.source and not call[1] for call in generator.calls)
    assert os.environ['LANGSMITH_TRACING'] == 'false'


def test_regression_veto_has_no_retry_fallback_or_prompt_leak(fixture_task, monkeypatch):
    marker = 'SECRET_REGRESSION_ONLY'
    checks = []
    def regression(task, runner, **kwargs):
        checks.append(kwargs)
        return {**result(kwargs.get('reference', False)), 'private_marker': marker}
    monkeypatch.setattr('repairbench.feedback.run_regressions', regression)
    generator = Generator([block('def f(x): return x + 1'), block('def f(x):\n    unused = 100\n    return x + 1')])
    report = repair_task(Path('.'), 'sample', Runner(), generator, budget=2)
    assert report['decision'] == 'regression_rejected' and report['accepted_candidate'] is None
    assert report['selected_candidate']['number'] == 1 and len(checks) == 2
    assert len(generator.calls) == 2 and marker not in json.dumps(generator.calls)


def test_invalid_regression_reference_is_not_candidate_failure(fixture_task, monkeypatch):
    monkeypatch.setattr('repairbench.feedback.run_regressions', lambda *a, **k: result(False))
    report = repair_task(Path('.'), 'sample', Runner(), Generator([block('def f(x): return x + 1')]), budget=1)
    assert report['decision'] == 'regression_reference_not_passing' and report['accepted_candidate'] is None
    assert 'regression' not in report


def test_independent_proposals_receive_no_feedback_and_same_seeds(fixture_task):
    outputs = ['invalid'] * 4
    left, right = Generator(outputs), Generator(outputs)
    repair_task(Path('.'), 'sample', Runner(), left, feedback=False)
    repair_task(Path('.'), 'sample', Runner(), right, feedback=True)
    assert [c[2:4] for c in left.calls] == [c[2:4] for c in right.calls]
    assert all(c[-1]['feedback'] is None for c in left.calls)


def test_cancel_before_work_and_between_generation_and_execution(fixture_task):
    runner, generator = Runner(), Generator([block('def f(x): return x + 1')])
    report = repair_task(Path('.'), 'sample', runner, generator, cancelled=lambda: True)
    assert report['decision'] == 'cancelled' and not runner.calls and not generator.calls
    report = repair_task(Path('.'), 'sample', runner, generator, cancelled=lambda: bool(generator.calls))
    assert report['decision'] == 'cancelled' and report['accepted_candidate'] is None
    assert len(runner.calls) == 2 and len(generator.calls) == 1


def test_visible_failure_and_runtime_error_fail_closed(fixture_task):
    runner = SimpleNamespace(run=lambda *a, **k: result(False))
    generator = Generator([])
    assert repair_task(Path('.'), 'sample', runner, generator)['eligibility'] == 'reference_not_passing'
    assert not generator.calls
    def broken(*args, **kwargs): raise RuntimeUnavailable('runtime offline')
    with pytest.raises(RuntimeUnavailable):
        repair_task(Path('.'), 'sample', SimpleNamespace(run=broken), generator)
    for budget in (0, 9):
        with pytest.raises(ValueError): repair_task(Path('.'), 'sample', Runner(), generator, budget=budget)


def test_time_limit_stops_before_execution_and_graph_never_uses_network(fixture_task, monkeypatch):
    import socket
    from repairbench import feedback as module
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected network access')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setenv('LANGSMITH_TRACING', 'true')
    generator = Generator([block('def f(x): return x + 1')])
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: 20 if generator.calls else 0))
    runner = Runner()
    report = repair_task(Path('.'), 'sample', runner, generator, max_seconds=10)
    assert report['decision'] == 'time_limit' and len(runner.calls) == 2
    assert report['accepted_candidate'] is None and len(generator.calls) == 1


def test_feedback_is_bounded_and_omits_regression_fields():
    feedback = visible_feedback({'response': 'a'*9000, 'run': {'status': 'failed', 'evidence': {'output': 'b'*9000}}, 'regression': 'secret'})
    assert len(feedback['previous_response']) == 6000 and feedback['response_truncated']
    assert len(feedback['output_tail']) == 1600 and 'secret' not in json.dumps(feedback)
    small = edit_size('def f(x): return x - 1\n', 'def f(x): return x + 1\n')
    comments = edit_size('# header\ndef f(x): return x - 1\n', 'def f(x): return x + 1\n')
    assert small['token_edits'] == comments['token_edits'] == 2


@pytest.mark.parametrize('stage', ['reference', 'candidate'])
@pytest.mark.parametrize('reason', ['cancelled', 'time_limit'])
def test_stop_during_regression_cannot_accept_patch(fixture_task, monkeypatch, stage, reason):
    from repairbench import feedback as module
    stopped = []
    calls = []
    def regression(task, runner, **options):
        current = 'reference' if options.get('reference') else 'candidate'
        calls.append(current)
        if current == stage:
            stopped.append(True)
        return result()
    monkeypatch.setattr(module, 'run_regressions', regression)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: 20 if stopped and reason == 'time_limit' else 0))
    generator = Generator([block('def f(x): return x + 1')])
    report = repair_task(Path('.'), 'sample', Runner(), generator, budget=1, max_seconds=10,
                         cancelled=lambda: bool(stopped) and reason == 'cancelled')
    assert report['decision'] == reason and report['accepted_candidate'] is None
    assert report['selected_candidate'] and len(generator.calls) == 1
    assert calls == (['reference'] if stage == 'reference' else ['reference', 'candidate'])


def test_evaluation_resume_verifies_config_and_task_bytes(tmp_path, monkeypatch):
    from repairbench import feedback_evaluation as module
    monkeypatch.setattr(module, 'verify', lambda p: {'tasks': ['gcd'], 'revision': 'r', 'archive_sha256': 'a'})
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda n: 'pinned')
    calls = []
    def repair(*args, **kwargs):
        calls.append(kwargs)
        return {'eligibility': 'eligible', 'attempts': [], 'selected_candidate': None, 'accepted_candidate': None, 'decision': 'no_visible_pass'}
    monkeypatch.setattr(module, 'repair_task', repair)
    generator = SimpleNamespace(record={}, adapter_record={}, device='cpu', max_new_tokens=768)
    runner = SimpleNamespace(preflight=lambda: {'image_id': 'fixed'})
    out = tmp_path/'run'
    first = module.evaluate_feedback(tmp_path, out, runner, generator, tasks=['gcd'])
    assert first['completed'] and len(calls) == 2
    module.evaluate_feedback(tmp_path, out, runner, generator, tasks=['gcd'], resume=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='configuration'):
        module.evaluate_feedback(tmp_path, out, runner, generator, budget=2, tasks=['gcd'], resume=True)
    (out/'tasks/gcd.json').write_text('{}')
    with pytest.raises(ValueError, match='evidence'):
        module.evaluate_feedback(tmp_path, out, runner, generator, tasks=['gcd'], resume=True)


@pytest.fixture
def docker_runner():
    path = os.environ.get('REPAIR_IMAGE_FILE')
    if not path:
        pytest.skip('Set REPAIR_IMAGE_FILE for actual container checks')
    runner = DockerRunner(Path(path).read_text().strip())
    runner.preflight()
    return runner


@pytest.mark.docker
@pytest.mark.parametrize('name', sorted(CASES))
def test_authored_suites_pass_audited_references(docker_runner, name):
    reference = load_task(Path('data/quixbugs'), name, reference=True)
    run = run_regressions(reference, docker_runner, reference=True)
    assert run['status'] == 'passed', run


@pytest.mark.docker
def test_actual_regression_gate_rejects_visible_test_overfit(docker_runner):
    source = '''def gcd(a, b):
    import math
    visible = {(17, 0), (13, 13), (37, 600), (20, 100), (624129, 2061517), (3, 12)}
    return math.gcd(a, b) if (a, b) in visible else -1
'''
    report = repair_task(Path('data/quixbugs'), 'gcd', docker_runner, Generator([block(source)]), budget=1)
    assert report['selected_candidate'] and report['attempts'][0]['visible_passed']
    assert report['decision'] == 'regression_rejected' and report['accepted_candidate'] is None
    assert report['regression']['status'] == 'failed'
