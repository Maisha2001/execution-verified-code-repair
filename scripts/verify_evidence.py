"""Verify published evidence against exact checkout, index, or commit bytes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def verify(root: Path, *, staged: bool = False, revision: str | None = None) -> dict:
    if staged and revision is not None:
        raise ValueError('Choose staged or committed evidence, not both')

    def git(*args: str) -> bytes:
        return subprocess.check_output(['git', '-C', str(root), *args])

    if staged:
        paths = git('ls-files', '-z', '--', 'results').decode().split('\0')
    elif revision is not None:
        revision = git('rev-parse', '--verify', revision + '^{commit}').decode().strip()
        paths = git('ls-tree', '-rz', '--name-only', revision, '--', 'results').decode().split('\0')
    else:
        paths = [p.relative_to(root).as_posix() for p in (root / 'results').rglob('*') if p.is_file()]

    def read(path: str) -> bytes:
        if staged:
            return git('show', ':' + path)
        if revision is not None:
            return git('show', revision + ':' + path)
        return (root / path).read_bytes()

    manifests = sorted(p for p in paths if p.endswith('/checksums.json'))
    if not manifests:
        raise ValueError('No evidence checksum manifests found')
    checked = {}
    for path in manifests:
        raw = read(path)
        if b'\r' in raw:
            raise ValueError(f'{path}: checksum manifest must use LF')
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError(f'{path}: empty or invalid checksum manifest')
        parent = PurePosixPath(path).parent
        listed = set()
        for name, expected in manifest.items():
            if PurePosixPath(name).name != name or '\\' in name or name in {'.', '..', 'checksums.json'}:
                raise ValueError(f'{path}: invalid evidence filename {name!r}')
            if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
                raise ValueError(f'{path}: invalid SHA256 for {name}')
            target = str(parent / name)
            if target not in paths:
                raise ValueError(f'{target}: missing evidence')
            data = read(target)
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise ValueError(f'{target}: checksum mismatch: expected {expected}, got {actual}')
            if name.endswith('.json'):
                if b'\r' in data:
                    raise ValueError(f'{target}: JSON evidence must use LF')
                json.loads(data)
            checked[target] = actual
            listed.add(target)
        siblings = {p for p in paths if PurePosixPath(p).parent == parent and p.endswith('.json') and p != path}
        if not siblings.issubset(listed):
            raise ValueError(f'{path}: unlisted JSON evidence: {sorted(siblings - listed)}')
    return {'source': 'index' if staged else revision or 'worktree', 'manifests': manifests, 'files': checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--staged', action='store_true')
    source.add_argument('--revision')
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), staged=args.staged, revision=args.revision), indent=2))
