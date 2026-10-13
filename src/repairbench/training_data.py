import ast
import io
import json
import time
import tokenize
import urllib.request
from collections import Counter

from .candidates import propose
from .data import Task, digest, verify
from .inference import messages
from .patching import apply_response

MBPP_REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
MBPP_URL = f"https://raw.githubusercontent.com/google-research/google-research/{MBPP_REVISION}/mbpp/sanitized-mbpp.json"
MBPP_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_mbpp(path):
    if not path.exists():
        with urllib.request.urlopen(MBPP_URL, timeout=30) as response:
            raw = response.read(1_000_001)
        if digest(raw) != MBPP_SHA256:
            raise ValueError("MBPP download checksum mismatch")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    if digest(raw) != MBPP_SHA256:
        raise ValueError("MBPP checksum mismatch")
    rows = json.loads(raw)
    if len(rows) != 427 or len({r['task_id'] for r in rows}) != 427:
        raise ValueError("Unexpected MBPP task inventory")
    return rows


def official_split(task_id):
    if 601 <= task_id <= 974:
        return "train"
    if 511 <= task_id <= 600:
        return "validation"
    if 11 <= task_id <= 510:
        return "test"
    if 1 <= task_id <= 10:
        return "prompt"
    raise ValueError("Unknown MBPP task ID")


def fingerprint(source):
    tree = ast.parse(source)
    names = {}
    for node in ast.walk(tree):
        for field in ("id", "arg", "name"):
            value = getattr(node, field, None)
            if isinstance(value, str):
                names.setdefault(value, f"identifier_{len(names)}")
                setattr(node, field, names[value])
    canonical = ast.unparse(tree)
    tokens = [t.string for t in tokenize.generate_tokens(io.StringIO(canonical).readline)
              if t.type not in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}]
    shingles = {tuple(tokens[i:i+7]) for i in range(max(1, len(tokens)-6))}
    return digest(ast.dump(tree, include_attributes=False).encode()), shingles


def similarity(left, right):
    if left[0] == right[0]:
        return 1.0
    return len(left[1] & right[1]) / max(1, len(left[1] | right[1]))


def split_controls(rows, benchmark_sources):
    features = {r['task_id']: fingerprint(r['code']) for r in rows}
    benchmark = [fingerprint(s) for s in benchmark_sources]
    priority = {"prompt": 0, "train": 1, "validation": 2, "test": 3}
    excluded = {}
    # Keep a clone only in its more protected split; never move held-out rows into training.
    ordered = sorted(rows, key=lambda r: (-priority[official_split(r['task_id'])], r['task_id']))
    retained = []
    for row in ordered:
        identity = row['task_id']
        feature = features[identity]
        if official_split(identity) == "prompt":
            excluded[str(identity)] = {"reason": "official_prompt_partition"}
            continue
        if any(similarity(feature, b) >= 0.8 for b in benchmark):
            excluded[str(identity)] = {"reason": "quixbugs_clone"}
            continue
        protected = [r['task_id'] for r in rows if priority[official_split(r['task_id'])] > priority[official_split(identity)]]
        collision = next((other for other in sorted(set(retained + protected)) if similarity(feature, features[other]) >= 0.8), None)
        if collision is not None:
            excluded[str(identity)] = {"reason": "duplicate_or_near_clone", "retained_id": collision}
        else:
            retained.append(identity)
    return {"policy": "official IDs; normalized AST exact or identifier-normalized 7-token Jaccard >=0.8; protected split wins",
            "excluded": excluded, "partitions": {split: sorted(i for i in retained if official_split(i) == split)
                                                  for split in ("train", "validation", "test")},
            "source_hashes": {str(r['task_id']): digest(r['code'].encode()) for r in rows}}


def task_payload(row, source):
    imports = "\n".join(row['test_imports'])
    tests = "import python_programs.mbpp_task as _candidate\n\n"
    for index, assertion in enumerate(row['test_list']):
        ast.parse(assertion)
        # This exec is emitted into the container payload, never run on the host.
        # The candidate namespace supports private names without collecting test_* helpers.
        tests += f"def test_case_{index}():\n    exec({(imports + chr(10) + assertion)!r}, vars(_candidate))\n\n"
    files = {"python_programs/mbpp_task.py": source.encode(), "test_task.py": tests.encode()}
    return Task("mbpp_task", source, files)


def compact_result(result):
    return {k: v for k, v in result.items() if k != "evidence"} | {
        "evidence": {k: v for k, v in (result.get('evidence') or {}).items() if k != "output"}}


def build_training_data(source, benchmark_root, output, runner):
    rows = load_mbpp(source)
    benchmark = verify(benchmark_root)
    sources = [(benchmark_root / f"correct_python_programs/{name}.py").read_text(encoding='utf-8') for name in benchmark['tasks']]
    controls = split_controls(rows, sources)
    configuration = {"mbpp_revision": MBPP_REVISION, "mbpp_sha256": MBPP_SHA256, "license": "CC-BY-4.0",
                     "benchmark_revision": benchmark['revision'], "controls": controls,
                     "runner": runner.preflight(), "candidate_budget": 6, "max_variants": 2, "test_wrapper_version": 2}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'configuration.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != configuration:
            raise ValueError("Data preparation configuration changed; use a new directory")
    else:
        write_json(config_path, configuration)
    if (output / 'manifest.json').exists():
        _, manifest = load_examples(output)
        return manifest
    (output / 'audits').mkdir(exist_ok=True)
    started = time.perf_counter()
    examples = {"train": [], "validation": []}
    audits = []
    for row in sorted(rows, key=lambda r: r['task_id']):
        identity = row['task_id']
        split = official_split(identity)
        if split not in examples or identity not in controls['partitions'][split]:
            continue
        checkpoint = output / 'audits' / f'{identity}.json'
        if checkpoint.exists():
            audit = json.loads(checkpoint.read_text())
        else:
            reference = task_payload(row, row['code'])
            outcome = runner.run(reference.files, 'test_task.py')
            audit = {"task_id": identity, "split": split, "reference": compact_result(outcome), "attempts": [], "examples": []}
            if outcome['status'] == 'passed' and outcome['evidence']['passed'] == len(row['test_list']) and not outcome['evidence']['skipped']:
                for candidate in propose(row['code'], 6):
                    try:
                        apply_response(reference, '```python\n' + candidate.source + '\n```')
                    except ValueError:
                        continue
                    broken = task_payload(row, candidate.source)
                    result = runner.run(broken.files, 'test_task.py')
                    audit['attempts'].append({"operation": candidate.operation, "source_sha256": candidate.sha256, "result": compact_result(result)})
                    evidence = result.get('evidence') or {}
                    if 'Timeout' in evidence.get('output', ''):
                        continue
                    # Only ordinary test failures qualify; timeouts, import errors and crashes do not train the model.
                    if result['status'] != 'failed' or evidence.get('failed', 0) < 1 or evidence.get('errors', 0) or evidence.get('skipped', 0):
                        continue
                    if evidence.get('passed', 0) + evidence['failed'] != len(row['test_list']):
                        continue
                    context = [{"path": "test_task.py", "start_line": 1, "text": broken.files['test_task.py'].decode()}]
                    audit['examples'].append({"id": f"mbpp-{identity}-{len(audit['attempts'])}", "task_id": identity, "split": split,
                        "messages": messages(broken, context), "target": '```python\n' + row['code'].rstrip() + '\n```',
                        "source_sha256": candidate.sha256, "target_sha256": digest(row['code'].encode()), "operation": candidate.operation})
                    if len(audit['examples']) == 2:
                        break
            write_json(checkpoint, audit)
        audits.append(audit)
        examples[split].extend(audit['examples'])
        print(json.dumps({"audited": len(audits), "task_id": identity, "examples": {k: len(v) for k, v in examples.items()}}), flush=True)
    for split, values in examples.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in values), encoding='utf-8')
    report = {"configuration_sha256": digest(config_path.read_bytes()), "files": {f'{s}.jsonl': digest((output / f'{s}.jsonl').read_bytes()) for s in examples},
              "examples": {s: len(v) for s, v in examples.items()}, "tasks_with_examples": {s: len({r['task_id'] for r in v}) for s, v in examples.items()},
              "audit_tasks": len(audits), "reference_status": dict(Counter(a['reference']['status'] for a in audits)),
              "reserved_test_ids": controls['partitions']['test'], "test_execution_or_generation": False,
              "current_invocation_seconds": round(time.perf_counter()-started, 6)}
    write_json(output / 'manifest.json', report)
    return report


def load_examples(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    config_path = directory / 'configuration.json'
    if digest(config_path.read_bytes()) != manifest['configuration_sha256']:
        raise ValueError("Training configuration checksum mismatch")
    config = json.loads(config_path.read_text())
    if config['mbpp_sha256'] != MBPP_SHA256 or config['mbpp_revision'] != MBPP_REVISION:
        raise ValueError("Unexpected training data identity")
    examples = {}
    for split in ('train', 'validation'):
        path = directory / f'{split}.jsonl'
        if digest(path.read_bytes()) != manifest['files'][path.name]:
            raise ValueError("Training data checksum mismatch")
        values = [json.loads(line) for line in path.read_text().splitlines()]
        if not values or any(r['split'] != split or official_split(r['task_id']) != split or r['task_id'] not in config['controls']['partitions'][split] for r in values):
            raise ValueError("Training split contamination or empty partition")
        examples[split] = values
    return examples, manifest
