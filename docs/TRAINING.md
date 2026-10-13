# Local adapter training

The training path adapts the pinned Qwen code model with LoRA on verified,
synthetically broken MBPP functions. Only query/value adapters are trainable.
Reference programs and mutations run in the existing isolated Docker runtime;
the host only parses and tokenizes source text.

## Data and split controls

[MBPP sanitized](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp)
contains 427 crowd-sourced, reviewed tasks from Austin et al., *Program Synthesis
with Large Language Models* (2021). The [dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp/blob/main/README.md)
specifies [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Changes here are synthetic single-expression bugs, test wrappers and chat-format
training pairs. Preserve that attribution when redistributing derived examples.

The source is pinned to revision `f82046ba5aabbbb427dbfd38a254d26bff08b533`.
Its 255,053-byte JSON file must match SHA-256
`ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9`.
The original ID partitions are retained: training 601Ã¢â‚¬â€œ974, validation 511Ã¢â‚¬â€œ600,
test 11Ã¢â‚¬â€œ510, and unused prompt tasks 1Ã¢â‚¬â€œ10. These ranges contain fewer rows in
the sanitized subset.

Before generating bugs, normalized AST equality and identifier-normalized
seven-token Jaccard similarity of at least 0.8 remove exact/near clones. A more
protected partition wins; no validation or test row is moved into training.
QuixBugs reference overlap is also excluded. Test sources are inspected only
for this static contamination screen. Their tests are not executed, predictions
are not generated, and they are not tokenized for training or validation.
This heuristic cannot detect all semantic duplicates or foundation-model
pretraining exposure.

For each retained training/validation task, all reference tests must pass
without skips. At most six deterministic mutations are examined; at most two
ordinary test-failing variants become repair pairs. Import failures, execution
limits, skipped cases and changed function signatures do not qualify.
All variants of a function remain in its original partition. The correct
module is the target; only the buggy module and visible tests form the input.

## Reproduce

Follow [model setup](INFERENCE.md) first, using a separate Python 3.11
environment and the appropriate PyTorch build. The measured training recipe
requires CUDA and does not automatically fall back to another device.

```sh
python -m pip install -r requirements/training.txt -e .
python -m pip check
repair-bench training-data --output data/repair-training-v2
repair-bench train --data data/repair-training-v2 --output runs/adapter-v2 --stop-after-steps 2
```

The first two optimizer steps provide a memory/latency check and a real
checkpoint. Inspect `runs/adapter-v2/summary.json` before continuing:

```sh
repair-bench train --data data/repair-training-v2 --output runs/adapter-v2 --resume runs/adapter-v2/checkpoint-000002
```

The default configuration is rank 8, alpha 16, query/value projections, zero
adapter dropout, three epochs, batch size one, accumulation four, AdamW at
2e-4 with five-step warmup and linear decay, gradient clipping at 1.0, and
gradient checkpointing. The frozen base uses bfloat16; adapters use float32.
There is no quantization. Sequences over 1,024 tokens are reported and excluded,
never silently truncated. Prompt tokens are masked; the loss supervises only
the corrected module and its chat terminator. Validation loss is weighted by
the number of target tokens; training averages per-example target losses.

## Recovery and evidence

Checkpoints include adapter weights, optimizer tensors, optimizer groups,
CPU/CUDA random states, example position, losses, dependency versions and input
hashes. Tensors use safetensors; metadata uses JSON. No pickle loader is used.
Directories appear atomically after all files and checksums are written.
Existing checkpoints cannot be overwritten. Resume verifies data, base model,
implementation, hardware identity, settings and parameter names before loading.
It continues the same deterministic example order at an optimizer boundary.
An existing run accepts only its latest complete checkpoint; it cannot be rewound.
GPU kernels may still differ across hardware or library versions; the exact
recovery assertion below is a CPU test, not a cross-device bitwise guarantee.

Checkpoints are saved every 20 steps, on normal completion and at requested
step/time stops. A crash between checkpoints resumes from the last complete
one; `.partial` directories are not valid checkpoints. `--max-seconds` bounds
an invocation and stops after a complete optimizer step. It is not a hard
kernel deadline. A checksum detects accidental corruption, not a hostile
replacement of both content and checksums; load artifacts only from trusted
local runs.

The best saved adapter is selected by validation loss; the initial base loss
is reported separately. A lower language-model loss does not establish better
repair success. Fixed-budget adapted-model comparisons are a separate
experiment. Public MBPP is not an unseen pretraining benchmark, and synthetic
single-expression bugs do not represent arbitrary real repository defects.

The CPU checkpoint test uses a tiny randomly initialized Qwen model with real
PEFT adapters, optimizer updates and dropout. It requires an interrupted run
to reproduce the next loss and every parameter exactly, while frozen weights
remain unchanged. CI runs this test without downloading pretrained weights.
GPU training measurements and split audits are recorded under
`results/training-v1/` after a verified run.

## Measured run

The RTX 5070 run used 155 training pairs from 90 functions and 58 validation
pairs from 32 different functions. All 152 reference audits passed; 30 functions
had no qualifying mutation within the fixed budget. No sequence exceeded the
1,024-token limit. The 253 retained official test tasks remain unused.

| Measurement | Observed value |
| --- | ---: |
| Optimizer updates / training epochs | 117 / 3 |
| Trainable adapter parameters | 1,089,536 (0.0705%) |
| Initial validation target-token loss | 0.120968 |
| Selected adapter validation loss | 0.047608 |
| Training invocations, including loading and validation | 129.49 s |
| Peak GPU allocation | 4.27 GiB |
| Reference and mutation audit | 839.51 s |
| Local tests | 52 passed, 50.068 s |

Training stopped at update 2 and resumed in a separate process. The selected
checkpoint is `checkpoint-000117`. Reloading that adapter separately
reproduced its validation loss exactly on this machine. The CPU recovery test
also compares every parameter and the next loss against uninterrupted training.

[Training history](../results/training-v1/summary.json),
[split and source hashes](../results/training-v1/data-configuration.json),
[all reference/mutation audits](../results/training-v1/data-audits.json) and
[verification](../results/training-v1/validation.json) retain the measured evidence.
Base weights remain pinned to the model setup; adapter weights and full optimizer
checkpoints are produced locally by the commands above. GPU allocation excludes
driver/display memory. The runtime is the sum of the two measured invocations;
it excludes the pause between them and the separate final reload.

The loss improvement is a training diagnostic, not a measured repair-rate gain.
These short public programs may be familiar from pretraining, and much of each
correct target can be copied from its buggy input. Fixed-budget generation and
regression evaluation are needed before claiming improved repair behavior.
