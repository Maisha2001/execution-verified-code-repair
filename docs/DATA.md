# Data provenance

Source: [QuixBugs](https://github.com/jkoppel/QuixBugs), revision
`4257f44b0ff1181dedaedee6a447e133219fcebf`.
The downloaded ZIP is 623,489 bytes, SHA-256
`0926648bc8bb4a027428785c5fbb8e36acae4555e4beec181934859d53298c62`.

The project contains 40 small Python programs with defects, accompanying tests,
and reference implementations. Upstream marks its code MIT, copyright
2017–2019 James Koppel. Its [legal notes](https://github.com/jkoppel/QuixBugs/blob/4257f44b0ff1181dedaedee6a447e133219fcebf/legal_notes.txt)
also discuss the provenance of the original textbook exercises. Retain both
the license and legal notes when reusing the downloaded benchmark. The original
license is included in `docs/QUIXBUGS-LICENSE.txt` and in the downloaded data.

`repair-bench data` verifies the whole archive before extracting selected files.
The package contains an immutable expected inventory of 176 file hashes. Every
task load compares the local manifest and file contents against that inventory;
rewriting the local manifest cannot make modified benchmark files valid.
Extraction rejects unsafe paths, links and oversized files. Downloaded programs
are never imported by the host application.

Only the chosen buggy module, required node helper, its tests and test data are
copied into a candidate's workspace. Reference code is added only to a separate
reference-audit container. The deterministic generator takes the buggy source
and a numeric candidate budget, with no access to test output or reference code.

The baseline uses the original default test selection, including upstream skips
for slow examples. Skipped counts are preserved. A passing candidate must match
the passing reference's passed/skipped counts. This is a public diagnostic suite,
not an independent held-out set. There is no train/validation split or model
training in this version. Future learned-model claims require independent tasks
and explicit contamination controls; no such claim is made here.

An existing dataset directory is verified rather than overwritten. If preparation
is interrupted, use a fresh destination with `repair-bench data --output PATH`.
An already downloaded, byte-identical archive can be supplied with `--archive`.
