import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .candidates import patch, propose
from .data import digest, load_task, verify
from .sandbox import DockerRunner


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def task_result(root: Path, name: str, runner: DockerRunner, budget: int) -> dict:
    task = load_task(root, name)
    test_file = f"python_testcases/test_{name}.py"
    reference = load_task(root, name, reference=True)
    fixed = runner.run(reference.files, test_file, reference=True)
    buggy = runner.run(task.files, test_file)
    result = {"task": name, "source_sha256": digest(task.source.encode()), "reference": fixed, "buggy": buggy, "attempts": [], "accepted_candidate": None}
    if fixed["status"] != "passed":
        result["eligibility"] = "reference_not_passing"
        return result
    if buggy["status"] == "passed":
        result["eligibility"] = "bug_not_exposed"
        return result
    if buggy["status"] not in {"failed", "timeout", "memory_limit", "output_limit"}:
        result["eligibility"] = "baseline_runtime_error"
        return result
    result["eligibility"] = "eligible"
    reference_count = fixed["evidence"]["passed"]
    for number, candidate in enumerate(propose(task.source, budget), start=1):
        payload = dict(task.files)
        payload[f"python_programs/{name}.py"] = candidate.source.encode()
        run = runner.run(payload, test_file)
        accepted = run["status"] == "passed" and run["evidence"]["passed"] == reference_count and run["evidence"]["skipped"] == fixed["evidence"]["skipped"]
        attempt = {"number": number, "operation": candidate.operation, "line": candidate.line, "candidate_sha256": candidate.sha256, "run": run, "accepted": accepted}
        result["attempts"].append(attempt)
        if accepted:
            result["accepted_candidate"] = {"number": number, "source": candidate.source, "sha256": candidate.sha256, "patch": patch(task.source, candidate.source, f"python_programs/{name}.py")}
            break
    return result


def summarize(results: list[dict], elapsed: float, budget: int) -> dict:
    eligible = [r for r in results if r["eligibility"] == "eligible"]
    solved = [r for r in eligible if r["accepted_candidate"]]
    attempts = [a for r in eligible for a in r["attempts"]]
    latencies = [a["run"]["seconds"] for a in attempts]
    sizes = []
    for result in solved:
        lines = result["accepted_candidate"]["patch"].splitlines()
        sizes.append(sum(line.startswith(("+", "-")) and not line.startswith(("+++", "---")) for line in lines))
    return {"tasks": len(results), "eligible_tasks": len(eligible), "excluded": {r["task"]:r["eligibility"] for r in results if r["eligibility"] != "eligible"}, "resolved_visible_tests": len(solved), "resolved_fraction": len(solved)/len(eligible) if eligible else None, "no_change_resolved": 0 if eligible else None, "candidate_budget": budget, "candidates_tested": len(attempts), "resolved_by_attempt": {str(k):sum(r["accepted_candidate"]["number"] <= k for r in solved) for k in (1, 4, budget)}, "median_candidate_seconds": statistics.median(latencies) if latencies else None, "accepted_patch_changed_lines": sizes, "total_seconds": round(elapsed, 6), "scope": "Deterministic visible-test benchmark; not unbiased pass@k, unseen-code generalization or proof of correctness."}


def evaluate(root: Path, output: Path, runner: DockerRunner, budget: int, tasks: list[str] | None = None) -> dict:
    if output.exists():
        raise ValueError("Choose a new output directory; completed evidence is not overwritten")
    manifest = verify(root)
    selected = tasks or manifest["tasks"]
    if len(set(selected)) != len(selected) or not set(selected) <= set(manifest["tasks"]):
        raise ValueError("Invalid task selection")
    if not 1 <= budget <= 128:
        raise ValueError("Invalid candidate budget")
    environment = runner.preflight()
    output.mkdir(parents=True)
    started = time.perf_counter()
    config = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_revision":manifest["revision"], "archive_sha256":manifest["archive_sha256"], "tasks":selected, "candidate_budget":budget, "python":platform.python_version(), "host_platform":platform.system(), "runtime":environment, "policy":"Fixed line-ordered AST mutations; stop at first candidate passing all original tests. Reference code is used only for test auditing and never for candidate generation."}
    config["implementation_sha256"] = {p.name: digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(output/"config.json",config)
    results = []
    for name in selected:
        result = task_result(root,name,runner,budget)
        write_json(output/"tasks"/(name+".json"),result)
        results.append(result)
        write_json(output/"summary.json",summarize(results,time.perf_counter()-started,budget))
        print(json.dumps({"task":name,"eligibility":result["eligibility"],"attempts":len(result["attempts"]),"resolved":result["accepted_candidate"] is not None}),flush=True)
    summary = summarize(results,time.perf_counter()-started,budget)
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output/"summary.json",summary)
    return summary
