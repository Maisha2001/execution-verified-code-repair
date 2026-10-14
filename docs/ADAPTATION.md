# Adapter repair comparison

The comparison evaluates one validation-selected LoRA checkpoint against the
unchanged base model at the same candidate budget. It separates the effect of
adaptation from adding retrieved context. All candidate programs run in the
existing isolated Docker runtime.

## Measured results

The October 13, 2026 run audited all 40 QuixBugs tasks. One reference,
`levenshtein`, exceeded the per-test time limit and is excluded for every arm.
All 39 eligible tasks were evaluated with a maximum of four attempts per arm.

| Method | First attempt | By four attempts | Resolution rate | Attempts | Executed | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Deterministic mutations | 1 | 4 | 10.26% | 118 | 118 | 0 |
| Unadapted Qwen | 3 | 10 | 25.64% | 138 | 44 | 94 |
| Unadapted + retrieval | 4 | 10 | 25.64% | 134 | 43 | 91 |
| Adapted Qwen | 15 | 21 | 53.85% | 102 | 67 | 35 |
| Adapted + retrieval | 9 | 17 | 43.59% | 119 | 70 | 49 |

Adaptation gained 13 tasks and lost two versus unadapted Qwen, a net gain of
11 tasks (28.21 percentage points). Adding retrieval to the adapted model gained
`find_in_sorted` but lost five tasks: `detect_cycle`,
`is_valid_parenthesization`, `quicksort`, `sieve` and `sqrt`. Its net loss of
four tasks shows that context was not uniformly beneficial. Every candidate and
acceptance/rejection decision in the original three arms matches the
[earlier comparison](MODEL_RESULTS.md); runtime measurements differ.

The full run took 4,530.03 seconds (75.5 minutes), excluding model loading,
on an RTX 5070. Peak allocated GPU memory was 4.32 GiB. Total generation time
was 1,058.63 seconds for unadapted, 1,267.60 for retrieval, 627.01 for adapted,
and 586.69 for adapted + retrieval. These totals reflect early stopping and
different response lengths, not equal numbers of generated tokens.

The preselected adapter was `checkpoint-000117`, SHA-256
`be346513058c363658dcaf324b2955ad843f98f53aa7f5ce6a4e7c4a265e44c2`.
No adapter or prompt was selected using these outcomes. The 253 reserved MBPP
test tasks remain unused. All 57 local tests passed, including actual Docker
isolation and tiny Qwen/LoRA generation checks. A real two-task GPU/container
smoke and an unchanged-evidence resume also passed.

Inspect the [summary](../results/adaptation-v1/summary.json),
[configuration and source hashes](../results/adaptation-v1/config.json),
[paired outcomes and failures](../results/adaptation-v1/analysis.json),
[validation evidence](../results/adaptation-v1/validation.json) and
[complete task records](../results/adaptation-v1/tasks).

### Failure analysis

The adapted model's two losses versus the base model were
[depth-first search](../results/adaptation-v1/tasks/depth_first_search.json) and
[k-heapsort](../results/adaptation-v1/tasks/kheapsort.json). Its first search
candidate checks visited nodes without recording visits, so cyclic input still
raises a recursion error. Its first heapsort candidate initializes a heap from
the first `k` items and then processes the whole input again, duplicating items.
Neither task was repaired within four adapted attempts.

For [quicksort](../results/adaptation-v1/tasks/quicksort.json), the adapted
model's accepted patch retains values equal to the pivot. With retrieved
context, its first candidate still discards duplicates and fails one of 13
visible tests; the arm never finds a passing patch within budget. Retrieval can
also help: the adapted retrieval arm repairs
[binary search](../results/adaptation-v1/tasks/find_in_sorted.json), which the
adapted-only arm misses.

Adaptation eliminates signature-change rejections in this run (75 unadapted,
50 retrieval, zero in either adapted arm). It does not eliminate invalid
outputs: adapted generation has 21 duplicate, eight unchanged, five formatting
and one parsing rejection; adapted + retrieval has 40 duplicate and nine
unchanged rejections. Passing patches still require independent regression
checks. Raw patch-size counts include removed comments and docstrings and must
not be interpreted as semantic edit minimality.

## Methods and controls

The five arms are deterministic mutations, unadapted Qwen, Qwen with retrieved
context, adapted Qwen, and adapted Qwen with retrieved context. All use the same
QuixBugs task list, audited references, original tests and acceptance policy.
Each arm receives at most four attempts and stops at its first passing candidate.
Invalid, unchanged and duplicate model outputs consume attempts. The first model
attempt is greedy; later attempts use the existing seeded sampling settings in
[model setup](INFERENCE.md). There is no test-feedback loop or prompt retuning.

The adapter is selected only by the MBPP validation target loss documented in
[training](TRAINING.md). Its checksummed checkpoint must identify the exact
verified base model. Only LoRA-only checkpoints are supported: base arms disable
the adapter during generation, while adapted arms enable it. A real tiny Qwen/LoRA
test verifies that enabling the adapter between base calls does not change base
outputs. Training, base-model and adapter hashes are recorded in each run.

Retrieval searches only the current task's visible tests, fixtures and graph
helpers. No reference program or other task is supplied as model context.
QuixBugs overlap was excluded when constructing MBPP training pairs. This makes
QuixBugs separate from the adapter's training data, but it is an already inspected
public development benchmark with possible foundation-model pretraining exposure.
The reserved MBPP test partition is not used by this comparison. Neither corpus
supports a claim of unseen pretraining data.

## Reproduce and resume

Complete [model setup](INFERENCE.md) and [adapter training](TRAINING.md) first.
Keep the complete local checkpoint directory, including its checksums and state.
Model weights, raw datasets and optimizer checkpoints are not bundled with the
repository. The full comparison runs locally, with no paid API or remote execution.

```sh
repair-bench compare --adapter runs/adapter-v2/checkpoint-000117 --budget 4 --output runs/adaptation-v1 --max-seconds 7200
```

Select the checkpoint identified by your own training run's `best_checkpoint`;
the command above names the recorded run's checkpoint. Do not select another
checkpoint after examining these repair outcomes. Omit `--adapter` to run the
original three-arm comparison.

Each completed task is saved with a checksum. A time limit stops before starting
another task; it is not a hard deadline inside model generation. Allow enough
time for one complete task. To continue with identical inputs and configuration:

```sh
repair-bench compare --adapter runs/adapter-v2/checkpoint-000117 --budget 4 --output runs/adaptation-v1 --resume --max-seconds 7200
```

Resume validates completed records and the dataset, model, adapter, implementation,
runtime and generation settings. Changed inputs or corrupted task evidence cause
an error. It does not silently overwrite results or regenerate completed tasks.
An interrupted current task may be rerun from its beginning with the same seeds;
completed tasks retain their measurements. `summary.json` marks incomplete runs.

## Reading the evidence

Reports retain every generated response, rejection, isolated test outcome and
accepted patch, as well as generation latency, GPU allocation and patch size.
Paired comparisons list tasks won, lost, solved by both methods or solved by
neither. Reference failures are excluded once for every arm, with reasons shown.
First-attempt and first-four-attempt success are descriptive resolution rates;
the greedy first attempt and stopping policy do not produce an unbiased pass@k
estimate. Total recorded time sums completed task times, including reference
audits, retrieval and candidate execution, but excludes model loading and pauses.

Passing visible tests is not proof of correctness. These measurements do not
cover independent regressions, arbitrary repository defects or hostile code.
Smaller language-model loss need not improve repair success, and adding context
can help some tasks while hurting others. Runtime depends on the device, output
length and early stopping; fewer attempts can make a method faster without making
its model intrinsically faster.
