# Execution-Verified Python Code Repair

A reproducible Python repair benchmark that proposes small patches, runs them in
isolated containers, and records the test evidence behind each accepted patch.

The implementation compares bounded AST mutations with a pinned local code model,
with and without retrieved repository context, on 40 QuixBugs tasks. It records
reference-test audits and measured results. Fine-tuning and a browser workbench
are not yet included. Passing available tests is evidence, not proof of correctness.

## Run an example

Requirements: Python 3.11 and a running local Linux Docker engine with memory,
PID and seccomp controls. Docker Desktop's Linux engine works on Windows.
Image construction and dataset download need internet; candidate execution has
no network. This deterministic example needs no model weights or API keys.

```sh
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements/dev.txt -e .
python -m pip check
repair-bench data
python -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
docker build -f docker/Dockerfile --iidfile runs/image.id .
repair-bench doctor
repair-bench evaluate --tasks gcd --budget 12 --output runs/gcd
```

On PowerShell, activate with `.venv\Scripts\Activate.ps1`; on bash, use
`source .venv/bin/activate`. Each run requires a new output directory. Inspect
`runs/gcd/tasks/gcd.json` for the original failure, attempted changes, accepted
diff, test counts and resource evidence. The CLI is the current working demo.

## Local code model

[Model setup and reproduction](docs/INFERENCE.md) add a pinned Qwen code model,
MiniLM context embeddings and an exact FAISS index. Inference runs locally;
generated programs run only in the isolated Docker runtime.

```sh
# After installing the model dependencies and downloading pinned weights:
repair-bench compare --tasks gcd breadth_first_search --budget 4 --output runs/model-example
```

The report compares mutations, generation without context, and generation with
retrieved visible tests and helpers. It includes generated text, rejected outputs,
accepted patches, original test evidence, latency and memory measurements.
In the [full measured comparison](docs/MODEL_RESULTS.md), each model variant
resolved **10 of 39 eligible tasks (25.64%)**, versus **4 of 39 (10.26%)** for
mutations at the same four-attempt budget. Retrieval changed which tasks passed
but did not improve the overall rate. Peak model allocation was 4.31 GiB.

## Evaluation

```sh
repair-bench evaluate --budget 12 --output runs/baseline
```

The recorded run resolves **4 of 39 eligible tasks (10.26%)**, with one
reference-timeout exclusion, at a maximum of 12 candidates per task. That original
measurement is preserved independently of the model comparison.

The baseline changes one expression at a time: comparison boundaries, arithmetic
operators, two-argument call order, or zero/one constants. Candidates follow a
fixed source order, stop at the first passing patch, and never receive reference
solutions. Reference implementations are executed separately to check the tests.
Tasks with broken references or already-passing originals are reported separately.

See [measured results](docs/EVALUATION.md), [data provenance](docs/DATA.md) and
[execution boundaries](docs/SECURITY.md). Reports preserve per-task failures,
exclusions, accepted patches, runtime/image identity and implementation hashes.
This public benchmark cannot establish unseen-code generalization.

## Tests

```sh
python -m pytest -q
```

Actual isolation tests require the image built above. In bash:

```sh
REPAIR_IMAGE_FILE=runs/image.id python -m pytest -q
```

In PowerShell:

```powershell
$env:REPAIR_IMAGE_FILE = 'runs/image.id'
python -m pytest -q
```

Without that variable, Docker tests are explicitly skipped. CI builds the image,
runs the full test suite, and evaluates two actual benchmark tasks.
It also replays a published model-generated patch in the isolated runtime.
GPU generation is measured locally; CI does not download the model weights.

## Repository

- `src/repairbench/`: dataset verification, candidate generation, runner and evaluation.
- `docker/`: pinned runtime image and bounded test supervisor.
- `tests/`: unit tests and actual container isolation probes.
- `docs/`: data, security, interpretation and reproduction notes.
- `results/`: published evaluation evidence; downloaded data and local runs stay ignored.

Code is MIT licensed. QuixBugs remains under its upstream terms and attribution;
see [its license](docs/QUIXBUGS-LICENSE.txt).
