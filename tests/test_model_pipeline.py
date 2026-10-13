from pathlib import Path
import json
import os

import pytest

from repairbench.comparison import compare_task, summary
from repairbench.data import Task
from repairbench.inference import messages
from repairbench.models import file_digest, verify_model
from repairbench.patching import apply_response
from repairbench.retrieval import context_chunks
from repairbench.sandbox import DockerRunner


@pytest.fixture
def task():
    return Task('example', 'def f(x):\n    return x < 1\n', {
        'python_programs/example.py': b'def f(x):\n    return x < 1\n',
        'python_testcases/test_example.py': b'assert f(1)\n',
        'json_testcases/example.json': b'[[[1], true]]',
        'python_programs/node.py': b'class Node: pass\n',
    })


def test_patch_preserves_tests_and_helpers_without_executing(task, tmp_path):
    marker = tmp_path / 'never-created'
    response = f'```python\nfrom pathlib import Path\nPath({str(marker)!r}).write_text("bad")\ndef f(x):\n    return x <= 1\n```'
    source, payload = apply_response(task, response)
    assert not marker.exists()
    assert payload['python_programs/example.py'] == source.encode()
    for name in task.files.keys() - {'python_programs/example.py'}:
        assert payload[name] == task.files[name]
    assert task.files['python_programs/example.py'] != payload['python_programs/example.py']


@pytest.mark.parametrize('response', [
    'def f(x): return True',
    '```python\ndef f(x): return x <= 1\n```\n```python\npass\n```',
    '```python\ndef f(y): return True\n```',
    '```python\ndef f(x): return x < 1\n```',
    '```python\ndef f(x):\n!\n```',
    '```python\ndef g(x): return True\n```',
    'x' * 65537,
], ids=['unfenced', 'multiple_blocks', 'signature', 'unchanged', 'syntax', 'missing_function', 'oversize'])
def test_ambiguous_invalid_unchanged_or_api_breaking_outputs_rejected(task, response):
    with pytest.raises(ValueError):
        apply_response(task, response)


def test_context_is_task_local_and_never_contains_reference_programs(task):
    polluted = Task(task.name, task.source, {**task.files,
        'correct_python_programs/example.py': b'ORACLE',
        'python_programs/other.py': b'OTHER_TASK',
        '../secret.txt': b'SECRET',
        'python_testcases/test_other.py': b'OTHER_TEST',
    })
    chunks = context_chunks(polluted)
    combined = '\n'.join(c.text for c in chunks)
    assert all(word not in combined for word in ['ORACLE','OTHER_TASK','OTHER_TEST','SECRET'])
    assert 'assert f(1)' in combined
    assert all(len(c.text) <= 1200 for c in chunks)
    prompt = messages(task, [c.evidence(0.5) for c in chunks])
    assert task.source in prompt[1]['content']


def test_large_fixture_lines_are_split_and_hashed(task):
    expanded = Task(task.name, task.source, {**task.files, 'json_testcases/example.json': b'a' * 2500})
    chunks = [c for c in context_chunks(expanded) if c.path.endswith('.json')]
    assert [len(c.text) for c in chunks] == [1200, 1200, 100]
    assert len({c.evidence(1)['sha256'] for c in chunks}) == 2


def test_model_identity_and_downloaded_content_must_match(tmp_path, monkeypatch):
    from repairbench import models
    spec = {'id': 'fixture/model', 'revision': 'a' * 40, 'files': ['config.json']}
    monkeypatch.setitem(models.MODELS, 'fixture', spec)
    directory = tmp_path / 'fixture'
    directory.mkdir()
    config = directory / 'config.json'
    config.write_text('{}')
    record = {**spec, 'sha256': {'config.json': file_digest(config)}}
    manifest = directory / 'verified-model.json'
    manifest.write_text(json.dumps(record))
    assert verify_model(tmp_path, 'fixture')[0] == directory
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='file changed'):
        verify_model(tmp_path, 'fixture')
    manifest.write_text(json.dumps({**record, 'revision': 'b' * 40}))
    with pytest.raises(ValueError, match='identity'):
        verify_model(tmp_path, 'fixture')


def test_comparison_counts_invalid_and_duplicates_and_checks_reference_counts(task, monkeypatch):
    def load(root, name, reference=False):
        return Task(task.name, task.source, {**task.files, **({'correct_python_programs/example.py': b'ORACLE'} if reference else {})})
    monkeypatch.setattr('repairbench.comparison.load_task', load)
    monkeypatch.setattr('repairbench.comparison.propose', lambda source, budget: [])

    class Runner:
        def __init__(self): self.payloads = []
        def run(self, files, test, reference=False):
            self.payloads.append((files, reference))
            return {'status': 'passed' if reference or len(self.payloads) > 2 else 'failed', 'evidence': {'passed': 2 if reference else 1, 'skipped': 0}, 'seconds': 0.1}

    class Generator:
        def __init__(self): self.contexts = []
        def generate(self, passed_task, context, seed, sampled):
            assert not any(k.startswith('correct_') for k in passed_task.files)
            self.contexts.append(context)
            return ('invalid' if len(self.contexts) in (1, 5) else '```python\ndef f(x): return x <= 1\n```'), {'seconds': 0.1}

    class Retriever:
        def retrieve(self, passed_task):
            assert not any(k.startswith('correct_') for k in passed_task.files)
            return [{'text': 'visible'}]

    runner, generator = Runner(), Generator()
    result = compare_task(Path('.'), 'example', runner, generator, Retriever(), 4)
    assert result['eligibility'] == 'eligible'
    for arm in ('unadapted', 'retrieval'):
        assert not result['arms'][arm]['accepted_candidate']
        assert len(result['arms'][arm]['attempts']) == 4
        assert result['arms'][arm]['attempts'][2]['rejection'] == 'duplicate_candidate'
    assert all(not c for c in generator.contexts[:4])
    assert all(c for c in generator.contexts[4:])
    assert all(not any(k.startswith('correct_') for k in files) for files, ref in runner.payloads if not ref)
    report = summary([result], 4, 1)
    assert report['arms']['retrieval']['attempts'] == 4
    assert report['arms']['retrieval']['executed_candidates'] == 1
    assert report['arms']['retrieval']['resolved'] == 0


@pytest.mark.docker
def test_published_model_patch_replays_in_isolated_runtime():
    image_file = os.environ.get('REPAIR_IMAGE_FILE')
    if not image_file:
        pytest.skip('Set REPAIR_IMAGE_FILE to run actual Docker tests')
    from repairbench.data import load_task

    report = json.loads(Path('results/model-v1/tasks/bitcount.json').read_text())
    task = load_task(Path('data/quixbugs'), 'bitcount')
    arm = report['arms']['retrieval']
    candidate = arm['accepted_candidate']
    assert candidate is not None
    response = arm['attempts'][candidate['number'] - 1]['response']
    source, payload = apply_response(task, response)
    assert source == candidate['source']
    runner = DockerRunner(Path(image_file).read_text().strip())
    runner.preflight()
    result = runner.run(payload, 'python_testcases/test_bitcount.py')
    assert result['status'] == 'passed'
    assert result['evidence']['passed'] == report['reference']['evidence']['passed']
    assert result['evidence']['skipped'] == report['reference']['evidence']['skipped']
