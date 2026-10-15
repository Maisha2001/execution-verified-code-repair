import importlib.metadata
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .data import digest, verify
from .evaluate import write_json
from .feedback import repair_task
from .regressions import manifest as regression_manifest


def summarize(rows, budget):
    arms = {}
    for name in ('independent', 'feedback'):
        records = [r['arms'][name] for r in rows]
        eligible = [r for r in records if r['eligibility'] == 'eligible']
        attempts = [a for r in eligible for a in r['attempts']]
        generated = [a['generation'] for a in attempts if 'generation' in a]
        arms[name] = {
            'eligible': len(eligible), 'selected_visible_pass': sum(r['selected_candidate'] is not None for r in eligible),
            'accepted_with_regressions': sum(r['decision'] == 'accepted_with_regressions' for r in eligible),
            'accepted_visible_only': sum(r['decision'] == 'accepted_visible_only' for r in eligible),
            'regression_rejected': sum(r['decision'] == 'regression_rejected' for r in eligible),
            'decisions': dict(Counter(r['decision'] for r in records)),
            'attempts': len(attempts), 'executed': sum('run' in a for a in attempts),
            'rejections': dict(Counter(a['rejection'] for a in attempts if 'rejection' in a)),
            'generation_seconds': round(sum(g['seconds'] for g in generated), 6),
            'peak_gpu_allocated_bytes': max((g.get('peak_gpu_allocated_bytes') or 0 for g in generated), default=0),
        }
    common = [r for r in rows if all(a['eligibility'] == 'eligible' for a in r['arms'].values())]
    pairs = {'wins': [], 'losses': [], 'both': [], 'neither': []}
    for row in common:
        left = bool(row['arms']['feedback']['accepted_candidate'])
        right = bool(row['arms']['independent']['accepted_candidate'])
        pairs['both' if left and right else 'wins' if left else 'losses' if right else 'neither'].append(row['task'])
    return {'tasks': len(rows), 'candidate_budget': budget, 'arms': arms, 'feedback_versus_independent': pairs,
            'seconds': round(sum(r['seconds'] for r in rows), 6),
            'scope': 'Selected previously inspected public development tasks. Post-selection authored regression gate is never model feedback or a ranking signal. No independent generalization claim.'}


def evaluate_feedback(root, output, runner, generator, budget=4, tasks=None, *, resume=False,
                      max_seconds=3600, task_seconds=600):
    if (output.exists() and not resume) or (resume and not output.exists()):
        raise ValueError('Use a fresh output directory or explicitly resume')
    if not 1 <= budget <= 8 or max_seconds <= 0 or not 1 <= task_seconds <= 3600:
        raise ValueError('Invalid evaluation bounds')
    data = verify(root)
    selected = tasks or list(regression_manifest())
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(data['tasks']):
        raise ValueError('Invalid task selection')
    config = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_revision': data['revision'], 'archive_sha256': data['archive_sha256'],
        'tasks': selected, 'candidate_budget': budget, 'task_seconds': task_seconds,
        'code_model': generator.record, 'adapter': generator.adapter_record,
        'runtime': runner.preflight(), 'python': platform.python_version(),
        'packages': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'peft', 'langgraph', 'langchain-core', 'langsmith']},
        'generation': {'device': generator.device, 'max_new_tokens': generator.max_new_tokens, 'input_token_limit': 4096,
            'first_attempt': 'greedy', 'later_attempts': {'temperature': 0.7, 'top_p': 0.95},
            'seed': 'SHA256(task name) first32bits plus attempt offset modulo 2**31, common across arms',
            'context': 'No retrieved context; feedback arm receives only the preceding response and bounded visible-test feedback.'},
        'regression_sha256': regression_manifest(),
        'selection': 'Spend the fixed budget, then rank visible passing candidates by token edits, raw changed lines, and attempt number. Apply the regression gate only to the selected patch, without fallback or feedback.',
        'implementation_sha256': {p.name: digest(p.read_text(encoding='utf-8').encode()) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    if generator.device == 'cuda':
        gpu = generator.torch.cuda.get_device_properties(0)
        config['hardware'] = {'gpu': gpu.name, 'gpu_total_bytes': gpu.total_memory,
                              'cuda': generator.torch.version.cuda, 'cpu_threads': generator.torch.get_num_threads()}
    if resume:
        previous = json.loads((output/'config.json').read_text())
        if {k:v for k,v in previous.items() if k != 'started_at_utc'} != {k:v for k,v in config.items() if k != 'started_at_utc'}:
            raise ValueError('Evaluation configuration changed')
    else:
        write_json(output/'config.json', config)
    started = time.monotonic()
    rows = []
    for name in selected:
        path = output/'tasks'/f'{name}.json'
        checksum = output/'checksums'/f'{name}.sha256'
        if path.exists():
            if not checksum.exists() or checksum.read_text().strip() != digest(path.read_bytes()):
                raise ValueError('Task evidence changed')
            row = json.loads(path.read_text())
            if row['task'] != name:
                raise ValueError('Task identity changed')
        else:
            if time.monotonic() - started >= max_seconds:
                break
            start = time.monotonic()
            row = {'task': name, 'arms': {}}
            for arm in ('independent', 'feedback'):
                row['arms'][arm] = repair_task(root, name, runner, generator, budget,
                                               feedback=arm == 'feedback', max_seconds=task_seconds)
            row['seconds'] = round(time.monotonic() - start, 6)
            write_json(path, row)
            checksum.parent.mkdir(parents=True, exist_ok=True)
            checksum.write_text(digest(path.read_bytes()) + '\n')
        rows.append(row)
        report = summarize(rows, budget)
        write_json(output/'summary.json', {**report, 'completed': False, 'selected_tasks': len(selected)})
        print(json.dumps({'task': name, 'decisions': {k:v['decision'] for k,v in row['arms'].items()}}), flush=True)
    report = summarize(rows, budget)
    report.update(completed=len(rows) == len(selected), selected_tasks=len(selected),
                  current_invocation_seconds=round(time.monotonic() - started, 6))
    if report['completed']:
        report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output/'summary.json', report)
    return report
