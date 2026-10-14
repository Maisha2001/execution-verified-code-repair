import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from .candidates import patch, propose
from .data import digest, load_task, verify
from .evaluate import write_json
from .patching import apply_response

ARMS = ("deterministic", "unadapted", "retrieval")
ADAPTED_ARMS = ARMS + ("adapted", "adapted_retrieval")


def accepted(run, reference):
    return run["status"] == "passed" and all(run["evidence"].get(k) == reference["evidence"].get(k) for k in ("passed", "skipped"))


def compare_task(root, name, runner, generator, retriever, budget, arms=ARMS):
    task = load_task(root, name)
    test = f"python_testcases/test_{name}.py"
    reference = runner.run(load_task(root, name, reference=True).files, test, reference=True)
    buggy = runner.run(task.files, test)
    result = {"task": name, "source_sha256": digest(task.source.encode()), "reference": reference, "buggy": buggy, "arms": {}}
    if reference["status"] != "passed":
        result["eligibility"] = "reference_not_passing"
    elif buggy["status"] == "passed":
        result["eligibility"] = "bug_not_exposed"
    elif buggy["status"] not in {"failed", "timeout", "memory_limit", "output_limit"}:
        result["eligibility"] = "baseline_runtime_error"
    else:
        result["eligibility"] = "eligible"
    if result["eligibility"] != "eligible":
        return result
    started = time.perf_counter()
    context = retriever.retrieve(task)
    result["retrieval_seconds"] = round(time.perf_counter() - started, 6)
    result["context"] = context
    for arm in arms:
        attempts = []
        found = None
        seen = set()
        deterministic = list(propose(task.source, budget)) if arm == "deterministic" else []
        for i in range(budget):
            attempt = {"number": i + 1, "accepted": False}
            if arm == "deterministic":
                if i >= len(deterministic):
                    break
                candidate = deterministic[i]
                source = candidate.source
                payload = {**task.files, f"python_programs/{name}.py": source.encode()}
                attempt["operation"] = candidate.operation
            else:
                seed = (int(digest(name.encode())[:8], 16) + i) % (2**31)
                options = {'adapter': True} if arm in {'adapted', 'adapted_retrieval'} else {}
                response, stats = generator.generate(task, context if arm in {"retrieval", "adapted_retrieval"} else [], seed, sampled=i > 0, **options)
                attempt.update(response=response, generation=stats)
                try:
                    source, payload = apply_response(task, response)
                except ValueError as exc:
                    attempt["rejection"] = str(exc)
                    attempts.append(attempt)
                    continue
            sha = digest(source.encode())
            attempt["candidate_sha256"] = sha
            if sha in seen:
                attempt["rejection"] = "duplicate_candidate"
            else:
                seen.add(sha)
                run = runner.run(payload, test)
                attempt.update(run=run, accepted=accepted(run, reference))
                if attempt["accepted"]:
                    found = {"number": i + 1, "sha256": sha, "source": source, "patch": patch(task.source, source, f"python_programs/{name}.py")}
            attempts.append(attempt)
            if found:
                break
        result["arms"][arm] = {"attempts": attempts, "accepted_candidate": found}
    return result


def summary(results, budget, seconds, arm_names=ARMS):
    eligible = [r for r in results if r["eligibility"] == "eligible"]
    arms = {}
    for name in arm_names:
        rows = [r["arms"][name] for r in eligible]
        solved = [r["accepted_candidate"] for r in rows if r["accepted_candidate"]]
        attempts = [a for r in rows for a in r["attempts"]]
        generation = [a["generation"] for a in attempts if "generation" in a]
        arms[name] = {"resolved": len(solved), "eligible": len(eligible), "resolution_rate": len(solved) / len(eligible) if eligible else None, "resolved_first_k": {str(k): sum(s['number'] <= k for s in solved) for k in sorted({1, budget})}, "attempts": len(attempts), "executed_candidates": sum("run" in a for a in attempts), "rejected_before_execution": sum("rejection" in a for a in attempts), "generation_seconds": round(sum(g['seconds'] for g in generation), 6), "peak_gpu_allocated_bytes": max((g.get('peak_gpu_allocated_bytes') or 0 for g in generation), default=0), "patch_changed_lines": [sum(l.startswith(('+','-')) and not l.startswith(('+++','---')) for l in s['patch'].splitlines()) for s in solved]}
    paired = {}
    for candidate, baseline in [('retrieval', 'unadapted'), ('adapted', 'unadapted'),
                                ('adapted', 'retrieval'), ('adapted_retrieval', 'retrieval'),
                                ('adapted_retrieval', 'adapted')]:
        if candidate not in arm_names or baseline not in arm_names:
            continue
        wins, losses, both, neither = [], [], [], []
        for row in eligible:
            left = bool(row['arms'][candidate]['accepted_candidate'])
            right = bool(row['arms'][baseline]['accepted_candidate'])
            (both if left and right else wins if left else losses if right else neither).append(row['task'])
        paired[f'{candidate}_versus_{baseline}'] = {'wins': wins, 'losses': losses, 'both': both,
            'neither': neither, 'net_resolved': len(wins) - len(losses),
            'resolution_rate_difference': (len(wins) - len(losses)) / len(eligible) if eligible else None}
    return {"tasks": len(results), "eligible_tasks": len(eligible), "excluded": {r['task']: r['eligibility'] for r in results if r['eligibility'] != 'eligible'}, "candidate_budget": budget, "arms": arms, "paired": paired, "seconds": round(seconds, 6), "scope": "Visible-test diagnostic comparison on public QuixBugs; potential pretraining contamination. First-k success is not an unbiased pass@k estimate. No test-feedback loop; no independent generalization claim."}


def compare(root: Path, output: Path, runner, generator, retriever, budget=4, tasks=None, *, resume=False, max_seconds=None):
    if (output.exists() and not resume) or (resume and not output.exists()) or not 1 <= budget <= 8:
        raise ValueError("Use a fresh output directory, or explicitly resume an existing run; budget must be 1 to 8")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('Time budget must be positive')
    arm_names = ADAPTED_ARMS if getattr(generator, 'adapter_record', None) else ARMS
    manifest = verify(root)
    selected = tasks or manifest['tasks']
    if len(set(selected)) != len(selected) or not set(selected) <= set(manifest['tasks']):
        raise ValueError("Invalid task selection")
    runtime = runner.preflight()
    config = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_revision": manifest['revision'], "archive_sha256": manifest['archive_sha256'], "tasks": selected, "candidate_budget": budget, "code_model": generator.record, "embedding_model": retriever.record, "runtime": runtime, "python": platform.python_version(), "host_platform": platform.system(), "packages": {n: importlib.metadata.version(n) for n in ['torch','transformers','sentence-transformers','faiss-cpu','numpy']}, "generation": {"device": generator.device, "max_new_tokens": generator.max_new_tokens, "max_input_tokens": 4096, "first_attempt": "greedy", "later_attempts": {"temperature": 0.7, "top_p": 0.95}, "seed": "SHA256(task name) first32bits plus attempt offset, modulo 2**31; same across model arms"}, "retrieval": {"top_k": 3, "model_max_tokens": 256, "index": "normalized embeddings, exact FAISS inner product", "corpus": "Only current task's visible tests/fixtures and node/load helpers; no reference programs or other tasks"}, "policy": "All arms use the same original tests and independently audited references. Invalid and duplicate model outputs consume attempt budget. Stop each arm at first passing candidate. No reference source or feedback sent to model."}
    config['arms'] = list(arm_names)
    config['adapter'] = getattr(generator, 'adapter_record', None)
    if config['adapter']:
        config['packages']['peft'] = importlib.metadata.version('peft')
    config['implementation_sha256'] = {p.name: digest(p.read_text(encoding='utf-8').encode()) for p in sorted(Path(__file__).parent.glob('*.py'))}
    if generator.device == 'cuda':
        gpu = generator.torch.cuda.get_device_properties(0)
        config['hardware'] = {'gpu': gpu.name, 'gpu_total_bytes': gpu.total_memory, 'cuda': generator.torch.version.cuda, 'cpu_threads': generator.torch.get_num_threads()}
    if resume:
        previous = json.loads((output/'config.json').read_text())
        if {k:v for k,v in previous.items() if k != 'started_at_utc'} != {k:v for k,v in config.items() if k != 'started_at_utc'}:
            raise ValueError('Comparison configuration changed; cannot resume')
        config = previous
    else:
        write_json(output/'config.json', config)
    results = []
    start = time.perf_counter()
    for task in selected:
        task_path = output/'tasks'/f'{task}.json'
        if task_path.exists():
            result = json.loads(task_path.read_text())
            check = output/'checksums'/f'{task}.sha256'
            if not check.exists() or check.read_text().strip() != digest(task_path.read_bytes()):
                raise ValueError('Saved comparison task changed')
            if result['task'] != task or result['source_sha256'] != digest(load_task(root, task).source.encode()):
                raise ValueError('Saved comparison task identity changed')
        else:
            if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                break
            task_start = time.perf_counter()
            result = compare_task(root, task, runner, generator, retriever, budget, arm_names)
            result['comparison_seconds'] = round(time.perf_counter() - task_start, 6)
            write_json(task_path, result)
            check = output/'checksums'/f'{task}.sha256'
            check.parent.mkdir(parents=True, exist_ok=True)
            check.write_text(digest(task_path.read_bytes()) + '\n')
        results.append(result)
        report = summary(results, budget, sum(r['comparison_seconds'] for r in results), arm_names)
        report.update(completed=False, selected_tasks=len(selected))
        write_json(output/'summary.json', report)
        print(json.dumps({"task": task, "eligibility": result['eligibility'], "resolved": {n: bool(r['accepted_candidate']) for n, r in result['arms'].items()}}), flush=True)
    report = summary(results, budget, sum(r['comparison_seconds'] for r in results), arm_names)
    report.update(completed=len(results) == len(selected), selected_tasks=len(selected), current_invocation_seconds=round(time.perf_counter()-start, 6))
    if report['completed']:
        report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output/'summary.json', report)
    return report
