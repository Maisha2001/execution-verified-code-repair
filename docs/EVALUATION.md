# Deterministic baseline evaluation

Measured on 2026-10-12T02:09:38.043982+00:00. The complete evidence is in
[`results/baseline-v1`](../results/baseline-v1), including configuration,
per-task runs, failures, accepted source/diffs and validation results.

| Measure | Observed |
| --- | --- |
| Pinned tasks audited | 40 |
| Eligible tasks (reference passes, original fails) | 39 |
| No-change baseline on eligible tasks | 0 / 39 |
| Tasks resolved against visible tests | 4 / 39 (10.26%) |
| Resolved across the entire 40-task cohort | 4 / 40 (10.00%) |
| Maximum proposed candidates per task | 12 |
| Candidates actually tested | 228 |
| Resolved within first 1 / 4 / 12 candidates | 1 / 4 / 4 |
| Median candidate latency | 1.769 s |
| Total audit and search time | 769.810 s |

Candidate latency includes workspace staging, Docker startup, test execution and
result inspection, but excludes final resource removal. Total time includes
cleanup. Execution is sequential, with a one-CPU quota, 256 MiB memory, a
32-process/thread ceiling and an eight-second wall limit. Each pytest test has
a one-second limit. Host: Windows, Python 3.11.15,
Docker 29.3.0. The runtime image ID, base image
digest, engine resources and implementation hashes are retained for auditing.
These timings are observations on this host, not throughput guarantees.

## Accepted patches

| Task | Candidate number | Passed tests | Skipped tests |
| --- | --- | --- | --- |
| find_first_in_sorted | 2 | 7 | 0 |
| gcd | 3 | 6 | 0 |
| quicksort | 3 | 13 | 0 |
| rpn_eval | 1 | 6 | 0 |

Changed-line counts (added plus removed) are
2, 2, 2, 7.
Every accepted patch changes only the candidate program. Tests, fixtures and
reference code are unchanged. Reference solutions are never supplied to the
candidate generator. Acceptance requires a successful container and matching
reference passed/skipped counts, with no reported test errors or failures.

## Exclusions and unsuccessful attempts

- `levenshtein`: Reference test exceeded the one-second per-test limit.

Candidate outcomes: failed=199, output_limit=21, passed=4, timeout=4.
An output-limit outcome means the test process exceeded its bounded diagnostic
stream; it is never accepted. Full failure logs are bounded deliberately.
The mutation rules propose no changes for: `breadth_first_search`, `depth_first_search`, `detect_cycle`, `reverse_linked_list`, `topological_ordering`.
These tasks stay in the eligible denominator and count as unresolved. Other
tasks may have fewer than 12 distinct syntactically valid mutations. The budget
is a maximum, not a promise that every task has 12 candidates.

## Interpretation and reproduction

This measures a fixed, deterministic search over visible benchmark tests. The
first-k counts are not statistical pass@k estimates. There is no training,
tuning on a held-out split, stochastic sampling or learned model in this
baseline. The benchmark is public and potentially present in model training
corpora; its score must not be used to claim unseen-code generalization.
Default upstream skipped tests remain skipped. Available tests can miss wrong
behavior, even when a small patch passes all of them.

Follow the [README](../README.md) to build the pinned image and download verified
data, then run `repair-bench evaluate --budget 12 --output runs/reproduction`.
Use an unused output directory. Candidate order and patches are deterministic;
resource outcomes and timings can vary with hardware and engine load. Reports
are saved after every task. An interrupted report lacks `completed_at_utc` and
must not be treated as a completed evaluation.

All 30 unit, policy and real isolation tests passed in
35.946 seconds, with no skips. The runtime tests also cover
Docker attach clients that outlive a completed container: this must not be
misreported as a program timeout. The first diagnostic measurement exposed that
issue; the published measurement was rerun after the fix, with unchanged
candidate rules and budget. No diagnostic scores are substituted into this run.

QuixBugs excerpts in failure logs and accepted patches retain the upstream
[MIT attribution](QUIXBUGS-LICENSE.txt) and [provenance notes](QUIXBUGS-NOTES.txt).
