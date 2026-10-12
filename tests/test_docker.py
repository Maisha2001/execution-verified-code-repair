import json
import os
from pathlib import Path

import pytest

from repairbench.sandbox import DockerRunner, Limits

pytestmark = pytest.mark.docker


@pytest.fixture
def runner():
    image = os.environ.get('REPAIR_IMAGE_FILE')
    if not image:
        pytest.skip('Set REPAIR_IMAGE_FILE to run actual container isolation checks')
    runtime = DockerRunner(Path(image).read_text().strip(), Limits(wall_seconds=4))
    runtime.preflight()
    return runtime


def run_source(runner, source):
    return runner.run({'test_probe.py': source.encode()}, 'test_probe.py')


def test_pass_fail_and_no_tests_are_distinct(runner):
    assert run_source(runner, 'def test_ok(): assert 2+2 == 4')['status'] == 'passed'
    assert run_source(runner, 'def test_bad(): assert False')['status'] == 'failed'
    assert run_source(runner, 'x=1')['status'] != 'passed'


def test_network_permissions_and_filesystem(runner):
    source = '''
import errno, os, resource, socket
from pathlib import Path
def test_policy():
    assert os.getuid() == 65532
    assert not Path('/var/run/docker.sock').exists()
    status = Path('/proc/self/status').read_text()
    assert 'NoNewPrivs:\t1' in status
    assert 'CapEff:\t0000000000000000' in status
    assert 'Seccomp:\t2' in status
    assert Path('/sys/fs/cgroup/pids.max').read_text().strip() == '32'
    assert Path('/sys/fs/cgroup/memory.max').read_text().strip() == str(256*1024*1024)
    assert resource.getrlimit(resource.RLIMIT_FSIZE)[0] == 8388608
    for target in ['/workspace/probe', '/opt/runner.py']:
        try:
            Path(target).write_text('x')
        except OSError as error:
            assert error.errno in (errno.EROFS, errno.EACCES)
        else:
            raise AssertionError('Writable protected filesystem')
    with socket.socket() as sock:
        sock.settimeout(0.3)
        try:
            sock.connect(('1.1.1.1', 80))
        except OSError:
            pass
        else:
            raise AssertionError('External network reachable')
'''
    result = run_source(runner, source)
    assert result['status'] == 'passed', result


def test_collection_hang_hits_wall_limit_and_is_removed(runner):
    result = run_source(runner, 'import time\nwhile True: time.sleep(0.1)')
    assert result['status'] == 'timeout', result
    assert result['container_removed']
    assert result['seconds'] < 20


def test_test_hang_is_failed(runner):
    result = run_source(runner, 'import time\ndef test_hang():\n    while True: time.sleep(0.1)')
    assert result['status'] == 'failed', result
    assert 'Timeout' in result['evidence']['output']


@pytest.mark.parametrize('source', ["import os\nwhile True: os.write(1, b'x'*4096)", "def test_flood():\n    while True: print('x'*4096, flush=True)"])
def test_output_flood_is_bounded(runner, source):
    result = run_source(runner, source)
    assert result['status'] == 'output_limit', result
    assert result['output_bytes'] <= runner.limits.output_bytes


def test_memory_flood_is_killed(runner):
    result = run_source(runner, 'chunks=[]\nwhile True: chunks.append(bytearray(8*1024*1024))')
    assert result['status'] == 'memory_limit', result


def test_forked_children_are_removed_with_container(runner):
    source = 'import subprocess, time\nsubprocess.Popen(["python", "-c", "import time; time.sleep(60)"])\nwhile True: time.sleep(0.1)'
    result = run_source(runner, source)
    assert result['status'] == 'timeout', result
    remaining = runner.command('ps', '--all', '--filter', 'label=repairbench.ephemeral=1', '--format', '{{.ID}}')
    assert remaining.strip() == b''
    assert runner.command('volume', 'ls', '--filter', 'label=repairbench.ephemeral=1', '--format', '{{.Name}}').strip() == b''


def test_file_and_pid_limits_are_enforced(runner):
    source = '''
import errno, subprocess
def test_limits():
    try:
        with open('/tmp/large', 'wb') as file:
            file.write(b'x' * (9*1024*1024))
    except OSError as error:
        assert error.errno == errno.EFBIG
    else:
        raise AssertionError('File limit not enforced')
    children = []
    try:
        for _ in range(64):
            try:
                children.append(subprocess.Popen(['python', '-c', 'import time; time.sleep(10)']))
            except OSError as error:
                assert error.errno == errno.EAGAIN
                break
        else:
            raise AssertionError('PID limit not enforced')
        assert 0 < len(children) < 32
    finally:
        for child in children: child.terminate()
        for child in children: child.wait()
'''
    result = run_source(runner, source)
    assert result['status'] == 'passed', result
