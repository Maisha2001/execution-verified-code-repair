# Bounded repair and regression checks

The local repair workflow spends a fixed candidate budget, uses visible-test
feedback for retries, ranks patches, and checks the selected patch against
separately authored development tests. It never runs generated Python on the
host. Each execution uses the existing isolated Docker runtime.

## Run the comparison

Complete the [model setup](INFERENCE.md) and [adapter training](TRAINING.md).
Install the additional local workflow dependencies:

```sh
python -m pip install -r requirements/workflow.txt
repair-bench repair --adapter runs/adapter-v2/checkpoint-000117 --budget 4 --output runs/feedback-v1 --max-seconds 3600
```

The default selection is eight public development tasks: binary search, flatten,
GCD, k-heapsort, knapsack, quicksort, shunting-yard and square root. This explicitly
selected set was previously inspected; it is not an unseen evaluation sample.
Use `--tasks gcd` for a shorter example. Omitting `--adapter` uses the pinned base
model. The original `compare` command and its measured five-arm results remain
available separately.

The command compares independent proposals with feedback-assisted proposals.
Both use the same model, original program, seeds, output limits and maximum
attempt count. The first attempt is greedy. Subsequent attempts use the existing
sampling settings. Both spend the complete budget, including after a visible
pass, to permit patch ranking. Invalid, unchanged, duplicate and oversized-prompt
rejections consume attempts. Duplicates are not executed again within an arm.
There is no retrieved context in either arm of this experiment.

## Control flow

[LangGraph](https://pypi.org/project/langgraph/1.2.12/) runs the local proposal and
execution nodes with conditional termination and a finite recursion limit. No
remote graph service or model API is used. LangSmith tracing is explicitly
disabled, including inside the graph invocation.

Only the preceding attempt's response and original visible-test observations
are given to the feedback arm. The response is capped at 6,000 characters and
test output at its last 1,600 characters; truncation is recorded. The model's
4,096-token input limit remains enforced without silent token truncation.
Program text and test output are untrusted data, not executable instructions
for the host or authority to change the workflow. Prompt text never includes
reference code, reference-run output or authored regression results.

Unique parsed candidates run against the unchanged original tests. A visible
pass must match the audited reference's passed and skipped test counts. After
all attempts, passing candidates rank by changed token count, changed line count,
then attempt number. Token differences ignore comments and line breaks but keep
indentation; this is a deterministic edit heuristic, not a proof of minimality.
Failed candidates are also ranked for inspection, but cannot be accepted.

The selected patch is fixed **before** the authored regression gate. Only that
patch is checked; failure does not trigger a fallback patch, new model call,
ranking change or retry. A separately isolated reference run first verifies
that the regression suite itself passes. A failing reference is reported as
an invalid regression audit rather than a model failure. Tasks without an
authored suite are explicitly labeled `accepted_visible_only`.

The eight MIT-licensed authored suites cover small-input oracles, empty inputs,
duplicates, valid displacement boundaries, nested non-list objects, precedence,
and positive-domain numerical tolerance. They are stored in
`src/repairbench/regressions.py` and hashed into the report. They were written
before this workflow's model runs, based on task contracts and known development
failures. They are not a comprehensive correctness suite or the final independent
evaluation. The reserved MBPP test partition is untouched.

## Bounds, recovery and evidence

`--task-seconds` defaults to 600 seconds per task and method, checked between
model calls and container executions. Container execution retains its own hard
limits. A currently running model call is bounded by output tokens; the task
timer is not an interruptible GPU deadline. The callable workflow also accepts
a cancellation predicate and stops at the next boundary without accepting a
partially evaluated patch. A cancelled or timed-out record preserves attempts.

`--max-seconds` stops before starting another task pair. Allow time for both
methods on one task; this outer timer is not a process deadline. Resume completed
task pairs with the same command and `--resume`. Configuration, implementation,
dataset, model, adapter, regression hashes and task evidence checksums must match.
An interrupted task pair without a saved record reruns from its beginning.
Explicit per-task time-limit outcomes are terminal evidence, not silently retried.

Reports retain original failures, generated responses, bounded feedback, each
rejection or isolated test result, rankings, the selected diff and the separate
regression decision. Earlier benchmark reports are not overwritten. Fixed-budget
selection differs from the earlier stop-at-first-pass comparison, so resolution
and runtime should not be compared as if the protocols were identical.

Passing visible and regression tests remains evidence, not proof of correctness.
Public pretraining contamination, selective development tasks and incomplete
regression coverage limit generalization claims.

## Measured comparison

The fixed adapter and four-attempt run completed all eight selected tasks in
498.69 seconds on an RTX 5070. Peak GPU allocation was 3.09 GiB. Both methods
used 32 generation attempts; independent proposals executed 19 unique parsed
candidates and feedback executed 17.

| Method | Passed original tests | Passed regression gate | Duplicate proposals | Invalid responses |
| --- | ---: | ---: | ---: | ---: |
| Independent proposals | 6/8 | 6/8 | 9 | 4 |
| Test-feedback proposals | 5/8 | 5/8 | 14 | 1 |

Feedback gained no task and lost `sqrt`. Both methods passed `flatten`, `gcd`,
`knapsack`, `quicksort` and `shunting_yard`; neither passed `find_in_sorted` or
`kheapsort`. This experiment does **not** show a repair-success benefit from
feedback. It does not estimate performance on unseen programs.

For `sqrt`, independent generation found passing patches on attempts two and
four; ranking selected the fourth, with two token edits. Feedback produced two
unique failing patches and two duplicate proposals. For `quicksort`, a one-token
edit failed one original test, so ranking selected a three-token edit that
passed. Smaller edits cannot outrank passing tests.

The authored gate rejected no selected model-generated patch in this sample.
A separate real-container integration test verifies that a deliberately
overfitted GCD patch passes the original tests but fails the authored gate.
That test demonstrates the rejection path, not the gate's general effectiveness.

The [summary](../results/feedback-v1/summary.json),
[paired outcomes](../results/feedback-v1/analysis.json),
[configuration and source hashes](../results/feedback-v1/config.json), and
[validation record](../results/feedback-v1/validation.json) accompany every
[task's raw evidence](../results/feedback-v1/tasks). All 79 local tests passed,
including isolation, cancellation during regression execution, duplicate-budget
handling and reference audits. A completed-run resume verified all eight task
records unchanged and recomputed the summary without generating new candidates.
The 253 reserved MBPP test tasks remain unused.

The runner's test requirements are separate from host graph dependencies.
A [clean-build validation](../results/feedback-runtime-v1/validation.json)
checks the rebuilt image, identical installed test packages and runner code,
all 79 tests, and replays all 11 accepted method/task patches against both test
stages. Original model outputs and evaluation records remain unchanged.
