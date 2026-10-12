import hashlib
import io
import json
import re
import stat
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib.resources import files as resources
from pathlib import Path, PurePosixPath

REVISION = "4257f44b0ff1181dedaedee6a447e133219fcebf"
ARCHIVE_SHA256 = "0926648bc8bb4a027428785c5fbb8e36acae4555e4beec181934859d53298c62"
URL = f"https://codeload.github.com/jkoppel/QuixBugs/zip/{REVISION}"
MAX_ARCHIVE = 8_000_000
MAX_FILE = 1_000_000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and "\\" not in value and ":" not in value and all(part not in (".", "..", "") for part in value.split("/"))


def prepare(destination: Path, archive: Path | None = None) -> dict:
    if destination.exists():
        return verify(destination)
    if archive is None:
        with urllib.request.urlopen(URL, timeout=30) as response:
            raw = response.read(MAX_ARCHIVE + 1)
    else:
        if archive.stat().st_size > MAX_ARCHIVE:
            raise ValueError("Archive exceeds size limit")
        raw = archive.read_bytes()
    if len(raw) > MAX_ARCHIVE or digest(raw) != ARCHIVE_SHA256:
        raise ValueError("Archive does not match the pinned QuixBugs revision")
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        for item in zipped.infolist():
            relative = item.filename.partition("/")[2]
            if item.is_dir() or not relative:
                continue
            if not safe_path(relative) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe archive entry")
            if relative.split("/")[0] not in {"python_programs", "correct_python_programs", "python_testcases", "json_testcases"} and relative not in {"LICENSE", "legal_notes.txt", "conftest.py"}:
                continue
            if item.file_size > MAX_FILE:
                raise ValueError("Archive member exceeds size limit")
            files[relative] = zipped.read(item)
    tasks = sorted(PurePosixPath(name).stem.removeprefix("test_") for name in files if name.startswith("python_testcases/test_") and name.endswith(".py"))
    if len(tasks) != 40 or any(f"python_programs/{name}.py" not in files for name in tasks):
        raise ValueError("Unexpected benchmark task inventory")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {"dataset": "QuixBugs", "revision": REVISION, "archive_sha256": ARCHIVE_SHA256, "source": URL, "license": "MIT; see upstream legal_notes.txt", "tasks": tasks, "files": {name: digest(content) for name, content in sorted(files.items())}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return verify(destination)


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = json.loads(resources("repairbench").joinpath("quixbugs.json").read_text(encoding="utf-8"))
    if manifest != expected:
        raise ValueError("Dataset manifest differs from the packaged, pinned file inventory")
    if manifest["revision"] != REVISION or manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("Unexpected dataset identity")
    for name, expected in manifest["files"].items():
        path = root / name
        if not safe_path(name) or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or digest(path.read_bytes()) != expected:
            raise ValueError(f"Dataset file failed verification: {name}")
    return manifest


@dataclass(frozen=True)
class Task:
    name: str
    source: str
    files: dict[str, bytes]


def load_task(root: Path, name: str, *, reference: bool = False) -> Task:
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError("Invalid task identifier")
    manifest = verify(root)
    if name not in manifest["tasks"]:
        raise ValueError("Unknown task")
    selected = {"conftest.py", "python_testcases/load_testdata.py", "python_testcases/node.py", f"python_testcases/test_{name}.py", f"json_testcases/{name}.json", f"python_programs/{name}.py", "python_programs/node.py"}
    if reference:
        selected |= {f"correct_python_programs/{name}.py", "correct_python_programs/node.py"}
    files = {n: (root / n).read_bytes() for n in selected if n in manifest["files"]}
    source = files[f"python_programs/{name}.py"].decode("utf-8")
    return Task(name, source, files)
