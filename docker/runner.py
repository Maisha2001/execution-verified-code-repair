import json
import os
import resource
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "-p", "pytest_timeout", "--timeout=1", "--timeout-method=signal", "--junitxml=/tmp/results.xml", *sys.argv[1:]]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    output = bytearray()
    limit_hit = threading.Event()

    def read():
        while part := process.stdout.read(4096):
            left = 32768 - len(output)
            output.extend(part[:max(0, left)])
            if len(part) > left:
                limit_hit.set()
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    code = process.wait()
    thread.join(timeout=1)
    summary = {"protocol": 1, "exit_code": code, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "seconds": round(time.perf_counter()-started, 6), "max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "output_limit": limit_hit.is_set(), "output": bytes(output).decode("utf-8", errors="replace")}
    path = Path("/tmp/results.xml")
    if path.is_file() and path.stat().st_size <= 1_000_000:
        try:
            xml = ET.parse(path)
            for case in xml.iter("testcase"):
                if case.find("failure") is not None: summary["failed"] += 1
                elif case.find("error") is not None: summary["errors"] += 1
                elif case.find("skipped") is not None: summary["skipped"] += 1
                else: summary["passed"] += 1
        except ET.ParseError:
            summary["errors"] += 1
    print(json.dumps(summary), flush=True)
    return 0 if code == 0 and not limit_hit.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
