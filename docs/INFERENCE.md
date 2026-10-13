# Local model inference and repository context

The model comparison runs three independent repair methods against the same
original tests: expression mutations, an unadapted code model, and the same
unadapted model with retrieved context. No training or feedback from earlier
candidate test runs is used. The command-line reports are the current demo.

## Setup

First complete the dataset and Docker setup in the [README](../README.md).
In that project's isolated Python 3.11 environment:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/inference.txt
python -m pip check
repair-bench download-models
repair-bench compare --tasks gcd breadth_first_search --budget 4 --output runs/model-example
```

For an explicit CPU run, install PyTorch from its CPU wheel index and add
`--device cpu`. CPU model inference uses float32 and has not been timed in the
published GPU comparison. CUDA uses bfloat16 and SDPA without quantization.
The measured GPU is an NVIDIA RTX 5070 with approximately 12 GiB of memory.
This inference check does not establish memory feasibility for later training.

Downloads are public model files, not inference API calls. The combined local
weights and tokenizer/configuration files occupy approximately 3.2 GB. Both
models are pinned to immutable revisions; only the listed files are downloaded.
The code model's Apache license and both model cards stay alongside the weights.
No pickle weights or repository Python code are loaded. Inference is offline,
uses safetensors, disables remote code and Hub telemetry, and verifies the
download's recorded file hashes before loading. The local hash record is a
reproducibility check, not protection against a malicious local administrator.

| Purpose | Model and original documentation | Revision |
| --- | --- | --- |
| Python generation | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), Apache-2.0 | `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Context embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

## Retrieval and patch boundaries

Each task gets its own small FAISS index of normalized embeddings. The corpus
contains only that task's visible tests, JSON fixtures and node/test-data helpers.
It excludes reference implementations, other benchmark tasks and arbitrary files
on the host. No reference source enters either model. Tests can mention the name
of the reference package, but its code is not retrieved or provided.

Chunks contain up to 20 lines with four lines of overlap, further bounded to
1,200 characters; the embedding model has a 256-token input limit. Long lines
are split into separate segments. The top three exact inner-product neighbors
are recorded with their path, starting line, text, score and content hash.
MiniLM is a general text encoder; relevance and truncation remain limitations.

The prompt contains the failing module and, in the retrieval arm, those chunks.
The generator must return one complete fenced Python module. The patch parser
checks syntax, retains the original top-level function signatures, and replaces
only the selected program in a copy of the execution payload. It never imports
or executes candidate code on the host. Tests and helpers remain unchanged.
Malformed, unchanged or duplicate outputs consume the attempt budget. Syntax
checking is not a safety guarantee; every accepted candidate still runs in the
[isolated runtime](SECURITY.md).

## Reproduce and interpret

```sh
repair-bench compare --budget 4 --output runs/model-comparison
```

Each method gets at most four attempts per task and stops at its first passing
patch. The first model attempt is greedy; later attempts sample at temperature
0.7 and top-p 0.95. Seeds are derived from task names and attempt positions and
are shared across the two model arms. The pinned model defaults also apply:
top-k 20 for sampling and a repetition penalty of 1.1 for all attempts.
Outputs are limited to 768 new tokens;
prompts above 4,096 tokens are rejected rather than silently truncated. Hardware
and library differences can change results even with the same seeds.

Reference and original programs are audited once per task, then shared across
the three methods. Acceptance requires a passing isolated run and the same
passed/skipped test counts as the reference. Reports include raw generated
responses, rejections, test evidence, accepted diffs, token counts, generation
time, GPU allocation peaks, model revisions and dependency versions. Evidence
is checkpointed after each task; an unfinished run lacks `completed_at_utc`.
Use a new output directory for each run; automatic resume is not implemented.

This is a public, potentially memorized benchmark. Scores against its visible
tests do not establish correctness, independent generalization, security against
hostile submissions, or the benefit of fine-tuning. A later independent suite is
necessary for such claims. First-k success here is not an unbiased statistical
pass@k estimate. Learned inference is optional; deterministic evaluation and the
lightweight CI test suite do not download model weights.
