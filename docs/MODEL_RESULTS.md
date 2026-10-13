# Measured model and context comparison

Completed on 2026-10-13T22:07:26Z. The complete configuration, model-file hashes,
responses, rejections, test evidence and accepted patches are in
[`results/model-v1`](../results/model-v1). The original
[deterministic measurement](EVALUATION.md) remains unchanged.

All 40 pinned QuixBugs tasks were audited. The same 39 tasks were eligible in
each method; the `levenshtein` reference exceeded the original per-test limit.
Each method received a maximum of four attempts per task and stopped at its
first passing patch. Invalid and duplicate model responses consumed attempts.

| Measure | Mutations | Unadapted model | Model with retrieved context |
| --- | --- | --- | --- |
| Resolved visible tests | 4 / 39 | 10 / 39 | 10 / 39 |
| Resolution rate | 10.26% | 25.64% | 25.64% |
| Resolved on first attempt | 1 | 3 | 4 |
| Attempts used | 118 | 138 | 134 |
| Candidates executed | 118 | 44 | 43 |
| Rejected before execution | 0 | 94 | 91 |
| Median patch changed lines, additions plus removals | 2 | 20 | 3.5 |
| Generation time | Not applicable | 1,042.010 s | 1,248.984 s |
| Peak PyTorch GPU allocation | Not applicable | 2.94 GiB | 4.31 GiB |

The two model methods tie in total repairs. Retrieval adds successes on
`bucketsort`, `flatten`, `gcd`, `max_sublist_sum` and `shunting_yard`, but loses
`detect_cycle`, `is_valid_parenthesization`, `kheapsort`, `shortest_paths` and
`sieve`. This run does not show an aggregate advantage from adding context.
The smaller median patch size with context is an observation on different sets
of accepted tasks, not a controlled claim about patch quality.

## Failures and interpretation

The strict signature check rejected 75 model-only and 50 context-assisted
responses. It compares argument names, defaults and annotations; adding type
annotations also violates this fixed interface policy. Unchanged programs account
for another 14 and 17 rejections. The context-assisted method additionally had
13 syntax failures and nine invalid code-block responses. These rejected outputs
remain in the attempt counts and are never treated as successful repairs.

Among executed candidates, the model-only method had 32 test failures and two
output-limit failures; the context-assisted method had 33 test failures. All
accepted patches matched the reference's passed/skipped test counts. Test files,
fixtures and helpers were unchanged. The saved model repair for `bitcount` is
also replayed by the Docker integration test without requiring model downloads.

Preliminary two-attempt smoke tests used `gcd` and `breadth_first_search` before
this full four-attempt run. This is a development benchmark, not an untouched holdout. QuixBugs
may also have appeared in pretraining data. A small model and strict response
format leave many tasks unresolved; passing visible tests is not proof of a
general repair. There was no task-specific prompt tuning, adaptation or iterative
test feedback in the full run. These first-k counts are not unbiased pass@k
estimates, and one seeded run does not establish statistical significance.

## Runtime and reproduction

The comparison took **2,963.580 seconds (49.39 minutes)** after model loading,
including reference audits, retrieval, generation, isolated tests and cleanup.
Host: Windows, Python 3.11.15, RTX 5070, PyTorch 2.11.0+cu128 and CUDA 12.8.
Models used bfloat16 on the GPU and CPU embeddings with four PyTorch threads.
The maximum 4.31 GiB allocation includes the resident code model and inference
tensors; it excludes other applications' GPU allocations. This verifies local
inference fit, not future training capacity or production throughput.

Candidate execution retained the baseline's pinned Docker image, one-CPU quota,
256 MiB memory cap, 32-process/thread cap, network isolation, read-only filesystem,
eight-second wall limit and one-second pytest test limit. No generated program
was executed on the host. Model loading and dependency installation are outside
the reported comparison duration.

Follow [the inference setup](INFERENCE.md), then run:

```sh
repair-bench compare --budget 4 --output runs/model-reproduction
```

Revisions, file hashes, dependency versions, prompts, chunk provenance and fixed
budgets are recorded. Device/library differences may change stochastic outputs
and resource outcomes. The downloaded weights remain ignored rather than being
redistributed here. QuixBugs excerpts and patches retain its
[upstream license](QUIXBUGS-LICENSE.txt) and [provenance notes](QUIXBUGS-NOTES.txt).
