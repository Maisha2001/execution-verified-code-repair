# Execution boundaries

This is a local research runner for curated benchmark tasks and constrained
patches. It is not a public service for arbitrary hostile uploads. Containers
share a kernel; Docker isolation does not eliminate kernel vulnerabilities, and
test code could deliberately tamper with Python's test machinery. Test counts
and passing assertions are not cryptographic proof of semantic correctness.

The host loads programs only as data. Each execution uses a fresh Linux container
and a fresh Docker-managed workspace volume. A never-started preparation
container receives bounded file bytes; the execution container mounts that
volume read-only. No host directory, Docker socket, credential or reference
solution is mounted into candidate execution. The volume and containers are
removed after each attempt, including failed and timed-out attempts.

| Control | Default |
| --- | --- |
| Network | `none` |
| User | UID/GID 65532, all Linux capabilities dropped |
| Privilege changes | `no-new-privileges`, Docker default seccomp |
| Root filesystem / workspace | Read-only |
| Memory / swap | 256 MiB memory, no extra swap |
| Processes and threads | 32 |
| CPU | One CPU quota, 5 CPU-seconds per process |
| Wall time | 8 seconds from container attachment; 1 second per pytest test |
| Output | 32 KiB test output, 128 KiB host protocol buffer |
| Temporary storage | 16 MiB tmpfs, noexec/nosuid/nodev |
| Individual file size | 8 MiB |
| Open descriptors | 128 per process |
| Input | At most 128 files, 2 MB total, 1 MB per file |

Docker setup/inspection operations have separate bounded timeouts, so reported
per-attempt latency can exceed the execution wall limit. That latency excludes
final resource removal, which is counted in overall evaluation time. Output overflow, memory
kill, wall timeout, malformed runner protocol and test failures are separate
outcomes. A setup failure raises an error; there is no direct-host execution
fallback. Image IDs must be immutable and the expected runtime label must exist.
Use only the image built from this repository; the label is not a signature.

The image installs only the pinned test dependencies in `requirements/runtime.txt`.
Host development dependencies include that file and the local workflow packages;
model and graph packages are not installed in the candidate execution image.

The actual Docker test suite checks ordinary success/failure, empty test suites,
read-only paths, non-root/capability/seccomp settings, denied external networking,
collection/test hangs, memory/output floods, file/PID limits, and descendant
cleanup. Run these checks on each deployment engine; a configuration flag alone
does not establish enforcement.

If the host process or Docker itself is forcibly terminated, normal cleanup may
not complete. Inspect containers and volumes carrying `repairbench.ephemeral=1`
before removing anything; do not remove an active run's resources. Run one
benchmark at a time while using the cleanup integration test.
