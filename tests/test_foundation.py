import ast
import io
import json
import tarfile
from pathlib import Path

import pytest

from repairbench.candidates import patch, propose
from repairbench.data import load_task, safe_path, verify
from repairbench.sandbox import DockerRunner, Limits, archive_files


@pytest.mark.parametrize('value', ['../x', '/x', 'C:/x', 'a\\x', 'a/../b', './a', 'a//b', ''])
def test_workspace_paths_reject_escapes(value):
    assert not safe_path(value)
    with pytest.raises(ValueError):
        archive_files({value: b'x'})


def test_archive_is_bounded_read_only_and_has_no_links():
    content = archive_files({'package/code.py': b'print(1)'})
    with tarfile.open(fileobj=io.BytesIO(content)) as tar:
        assert [p.name for p in tar] == ['package', 'package/code.py']
        assert tar.getmember('package').mode == 0o555
        assert tar.getmember('package/code.py').mode == 0o444
        assert all(not p.issym() and not p.islnk() for p in tar)
    with pytest.raises(ValueError):
        archive_files({'x': b'x' * 1_000_001})


def test_mutations_are_reproducible_bounded_and_preserve_unicode():
    source = '# untouched\ndef f(x):\n    café = 1; return x < café\n'
    candidates = propose(source, 2)
    assert candidates == propose(source, 2)
    assert len(candidates) == 2
    assert len({x.sha256 for x in candidates}) == 2
    for candidate in candidates:
        ast.parse(candidate.source)
        assert candidate.source.startswith('# untouched\ndef f(x):\n')
        assert 'café' in candidate.source
        assert patch(source, candidate.source, 'code.py').startswith('--- a/code.py')


@pytest.mark.parametrize('budget', [0, 129])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError):
        propose('x=1', budget)


def test_image_and_limits_fail_closed():
    with pytest.raises(ValueError):
        DockerRunner('python:latest')
    with pytest.raises(ValueError):
        Limits(wall_seconds=0)
    runner = DockerRunner('sha256:' + 'a' * 64)
    args = runner.create_args('test-container', 'tests/test_x.py', False)
    for required in ['--network', 'none', '--read-only', '--cap-drop', 'ALL', '--pids-limit', '--memory-swap', '--user', '65532:65532']:
        assert required in args
    assert '--volume' not in args and '--mount' not in args


def test_manifest_cannot_be_replaced_with_local_hashes(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'files': {}}))
    with pytest.raises(ValueError, match='packaged'):
        verify(tmp_path)


def test_loaded_tasks_do_not_expose_reference_code():
    root = Path('data/quixbugs')
    if not root.exists():
        pytest.skip('Download the pinned benchmark to exercise its inventory')
    manifest = verify(root)
    assert len(manifest['tasks']) == 40
    task = load_task(root, 'gcd')
    assert task.source == task.files['python_programs/gcd.py'].decode()
    assert not any(name.startswith('correct_python_programs/') for name in task.files)
    assert any(name.startswith('correct_python_programs/') for name in load_task(root, 'gcd', reference=True).files)


def test_delayed_attach_client_does_not_turn_completed_tests_into_timeout(monkeypatch):
    runner = DockerRunner('sha256:' + 'a' * 64)
    calls = []
    def command(*args):
        calls.append(args)
        return json.dumps([{'State': {'Status': 'exited', 'Running': False, 'ExitCode': 0}}]).encode()
    monkeypatch.setattr(runner, 'command', command)
    assert not runner.stop_at_deadline('finished-container')
    assert calls == [('inspect', 'finished-container')]


def test_deadline_still_kills_live_execution(monkeypatch):
    runner = DockerRunner('sha256:' + 'a' * 64)
    calls = []
    def command(*args):
        calls.append(args)
        return json.dumps([{'State': {'Status': 'running', 'Running': True}}]).encode()
    monkeypatch.setattr(runner, 'command', command)
    assert runner.stop_at_deadline('live-container')
    assert calls == [('inspect', 'live-container'), ('kill', 'live-container')]
