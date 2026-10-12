import io
import json
import re
import subprocess
import tarfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass

from .data import safe_path


class RuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 8.0
    memory_mb: int = 256
    pids: int = 32
    output_bytes: int = 131072

    def __post_init__(self):
        if not 1 <= self.wall_seconds <= 60 or not 128 <= self.memory_mb <= 1024 or not 8 <= self.pids <= 64 or not 4096 <= self.output_bytes <= 1048576:
            raise ValueError("Unsupported sandbox limits")


def archive_files(files: dict[str, bytes]) -> bytes:
    if not files or len(files) > 128 or sum(map(len, files.values())) > 2_000_000:
        raise ValueError("Invalid workspace size")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        directories = set()
        for name, content in sorted(files.items()):
            if not safe_path(name) or len(content) > 1_000_000:
                raise ValueError("Unsafe workspace entry")
            pieces = name.split("/")
            for i in range(1, len(pieces)):
                directory = "/".join(pieces[:i])
                if directory not in directories:
                    item = tarfile.TarInfo(directory)
                    item.type = tarfile.DIRTYPE
                    item.mode = 0o555
                    archive.addfile(item)
                    directories.add(directory)
            item = tarfile.TarInfo(name)
            item.size = len(content)
            item.mode = 0o444
            archive.addfile(item, io.BytesIO(content))
    return data.getvalue()


class DockerRunner:
    def __init__(self, image: str, limits: Limits | None = None):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("Use an immutable local image ID from docker build --iidfile")
        self.image = image
        self.limits = limits or Limits()

    def command(self, *args: str, input: bytes | None = None, timeout: float = 20) -> bytes:
        try:
            result = subprocess.run(["docker", *args], input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeUnavailable("Local Docker runtime is unavailable") from exc
        if result.returncode:
            raise RuntimeUnavailable(result.stderr.decode("utf-8", errors="replace")[-2000:])
        return result.stdout

    def preflight(self) -> dict:
        info = json.loads(self.command("info", "--format", "{{json .}}", timeout=10))
        if info.get("OSType") != "linux" or not info.get("MemoryLimit") or not info.get("PidsLimit"):
            raise RuntimeUnavailable("A local Linux Docker engine with memory and PID controls is required")
        if not any("seccomp" in option for option in info.get("SecurityOptions", [])):
            raise RuntimeUnavailable("Docker's default seccomp profile must be active")
        image = json.loads(self.command("image", "inspect", self.image))[0]
        if image["Id"] != self.image or image["Config"].get("Labels", {}).get("repairbench.runtime") != "1":
            raise RuntimeUnavailable("Build the repository's runner image first")
        return {"server_version": info["ServerVersion"], "operating_system": info["OperatingSystem"], "cpus": info["NCPU"], "memory_bytes": info["MemTotal"], "image_id": self.image, "limits": asdict(self.limits)}

    def create_args(self, name: str, test_file: str, reference: bool, volume: str | None = None) -> list[str]:
        if not safe_path(test_file) or not test_file.endswith(".py"):
            raise ValueError("Invalid test path")
        args = ["create", "--name", name, "--label", "repairbench.ephemeral=1", "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", str(self.limits.pids), "--memory", f"{self.limits.memory_mb}m", "--memory-swap", f"{self.limits.memory_mb}m", "--cpus", "1", "--ulimit", "cpu=5:5", "--ulimit", "fsize=8388608:8388608", "--ulimit", "nofile=128:128", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=16777216,mode=1777", "--user", "65532:65532", "--workdir", "/workspace", "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1", self.image, test_file]
        if reference:
            args.append("--correct")
        if volume:
            args[1:1] = ["--mount", f"type=volume,src={volume},dst=/workspace,readonly,volume-nocopy"]
        return args

    def stop_at_deadline(self, name: str) -> bool:
        state = json.loads(self.command("inspect", name))[0]["State"]
        # The Docker attach client can outlive an already completed container.
        if state["Status"] == "exited" and not state["Running"]:
            return False
        try:
            self.command("kill", name)
        except RuntimeUnavailable:
            if json.loads(self.command("inspect", name))[0]["State"]["Running"]:
                raise
        return True

    def run(self, files: dict[str, bytes], test_file: str, *, reference: bool = False) -> dict:
        payload = archive_files(files)
        name = "repairbench-" + uuid.uuid4().hex
        volume = name + "-workspace"
        preparation = name + "-prepare"
        started = time.perf_counter()
        created = False
        volume_created = False
        preparation_created = False
        try:
            self.command("volume", "create", "--label", "repairbench.ephemeral=1", volume)
            volume_created = True
            self.command("create", "--name", preparation, "--label", "repairbench.ephemeral=1", "--network", "none", "--mount", f"type=volume,src={volume},dst=/workspace", "--entrypoint", "/bin/true", self.image)
            preparation_created = True
            self.command("cp", "-", preparation + ":/workspace", input=payload)
            self.command("rm", preparation)
            preparation_created = False
            self.command(*self.create_args(name, test_file, reference, volume))
            created = True
            config = json.loads(self.command("inspect", name))[0]
            host = config["HostConfig"]
            if host["NetworkMode"] != "none" or not host["ReadonlyRootfs"] or host["Memory"] != self.limits.memory_mb * 1024 * 1024 or host["PidsLimit"] != self.limits.pids or host.get("Binds") or host.get("Privileged") or config["Config"]["User"] != "65532:65532":
                raise RuntimeUnavailable("Container isolation policy was not applied")
            mounts = config["Mounts"]
            if len(mounts) != 1 or mounts[0]["Type"] != "volume" or mounts[0]["Name"] != volume or mounts[0]["Destination"] != "/workspace" or mounts[0]["RW"]:
                raise RuntimeUnavailable("Read-only workspace policy was not applied")
            proc = subprocess.Popen(["docker", "start", "--attach", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            chunks = bytearray()
            overflow = threading.Event()

            def read_output():
                while chunk := proc.stdout.read(4096):
                    remaining = self.limits.output_bytes - len(chunks)
                    chunks.extend(chunk[:max(remaining, 0)])
                    if len(chunk) > remaining:
                        overflow.set()

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            deadline = time.monotonic() + self.limits.wall_seconds
            status = None
            while proc.poll() is None:
                if overflow.is_set() or time.monotonic() >= deadline:
                    was_running = self.stop_at_deadline(name)
                    status = "output_limit" if overflow.is_set() else "timeout" if was_running else None
                    break
                time.sleep(0.02)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            reader.join(timeout=3)
            if reader.is_alive():
                raise RuntimeUnavailable("Docker output stream did not close")
            state = json.loads(self.command("inspect", name))[0]["State"]
            if state.get("OOMKilled"):
                status = "memory_limit"
            elif overflow.is_set():
                status = "output_limit"
            parsed = None
            try:
                parsed = json.loads(bytes(chunks))
            except (ValueError, UnicodeDecodeError):
                pass
            if status is None:
                if not isinstance(parsed, dict) or parsed.get("protocol") != 1:
                    status = "runner_error"
                elif parsed.get("output_limit"):
                    status = "output_limit"
                elif parsed.get("exit_code") == 0 and state["ExitCode"] == 0 and parsed.get("passed", 0) > 0 and parsed.get("failed") == 0 and parsed.get("errors") == 0:
                    status = "passed"
                else:
                    status = "failed"
            return {"status": status, "seconds": round(time.perf_counter() - started, 6), "container_exit_code": state["ExitCode"], "evidence": parsed, "output_bytes": len(chunks), "container_removed": True}
        finally:
            try:
                if created:
                    self.command("rm", "--force", name)
            finally:
                try:
                    if preparation_created:
                        self.command("rm", "--force", preparation)
                finally:
                    if volume_created:
                        self.command("volume", "rm", volume)
