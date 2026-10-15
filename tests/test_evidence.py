import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.verify_evidence import verify


def fixture(root, data=b'{"result": "passed"}\n'):
    folder = root / 'results' / 'probe'
    folder.mkdir(parents=True)
    (folder / 'record.json').write_bytes(data)
    (folder / 'checksums.json').write_bytes(json.dumps({'record.json': hashlib.sha256(data).hexdigest()}).encode() + b'\n')
    return folder


def test_published_evidence():
    report = verify(Path(__file__).resolve().parents[1])
    assert len(report['files']) >= 3


@pytest.mark.parametrize('change, message', [
    ('tamper', 'checksum mismatch'), ('missing', 'missing evidence'),
    ('unlisted', 'unlisted JSON evidence'), ('crlf', 'must use LF'),
])
def test_rejects_corrupt_or_incomplete_evidence(tmp_path, change, message):
    folder = fixture(tmp_path, b'{"result": "passed"}\r\n' if change == 'crlf' else b'{}\n')
    if change == 'tamper':
        (folder / 'record.json').write_bytes(b'{"result": "failed"}\n')
    elif change == 'missing':
        (folder / 'record.json').unlink()
    elif change == 'unlisted':
        (folder / 'diagnostic.json').write_bytes(b'{}\n')
    with pytest.raises(ValueError, match=message):
        verify(tmp_path)


def test_git_normalization_and_staged_bytes(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args])

    git('init', '-q')
    (tmp_path / '.gitattributes').write_bytes(b'* text=auto eol=lf\n')
    folder = fixture(tmp_path, b'{"result": "passed"}\r\n')
    git('add', '.')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(tmp_path, staged=True)
    blob = git('show', ':results/probe/record.json')
    assert b'\r' not in blob
    (folder / 'record.json').write_bytes(blob)
    (folder / 'checksums.json').write_bytes(json.dumps({'record.json': hashlib.sha256(blob).hexdigest()}).encode() + b'\n')
    git('add', '.')
    assert verify(tmp_path, staged=True)['files'] == verify(tmp_path)['files']
    # The index audit must ignore a later, unstaged corruption of the checkout.
    (folder / 'record.json').write_bytes(b'{"result": "changed"}\n')
    verify(tmp_path, staged=True)
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify(tmp_path)
