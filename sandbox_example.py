#!/usr/bin/env python3
"""Runs untrusted Python through the real sandbox (harness.run_code) and checks that it is sealed.

No AI account is needed. Run ./setup_sandbox.sh --minimal first. Exits with 0 only if every check passes.
"""
import os
import shutil
import sys
import tempfile

import harness

CHECKS = '''
import json, os, socket
def blocked(action):
    try:
        action()
        return False
    except OSError:
        return True
def connect():
    s = socket.socket()
    s.settimeout(3)
    s.connect(("1.1.1.1", 53))
print(json.dumps({
    "no_network": blocked(connect),
    "only_loopback": os.listdir("/sys/class/net") == ["lo"],
    "usr_read_only": blocked(lambda: open("/usr/sandbox_test", "w")),
    "data_readable": open("/work/data.txt").read() == "input data",
    "data_read_only": blocked(lambda: open("/work/data.txt", "a")),
    "host_secret_hidden": not os.path.exists(SECRET),
    "host_home_only_has_python": os.listdir(HOST_HOME) == [".sandbox-python"],
    "not_root": os.getuid() != 0,
    "tmp_is_fresh": not os.path.exists("/tmp/left_behind"),
}))
open("/tmp/left_behind", "w").write("gone after this run")
open("/work/kept/note.txt", "w").write("kept between runs")
'''

os.makedirs(f"{harness.HOME}/.sandbox-runs", exist_ok=True)
work = tempfile.mkdtemp(prefix="run-", dir=f"{harness.HOME}/.sandbox-runs")
host_only = tempfile.mkdtemp(prefix="host-only-", dir=harness.HOME)
try:
    secret = f"{host_only}/secret.txt"
    with open(secret, "w") as f:
        f.write("the container must not see this")
    with open(f"{work}/data.txt", "w") as f:
        f.write("input data")
    os.makedirs(f"{work}/kept")
    state = {"id": "example", "runs": 0, "work": work, "mem": "2g", "timeout": 60, "mounted_files": ["data.txt"],
             "persist": ["kept"], "prelude": f"SECRET = {secret!r}\nHOST_HOME = {harness.HOME!r}\n"}

    import json
    results = {}
    for run in (1, 2):  # a second run shows that /tmp starts clean and only the mounted folder persists
        state["runs"] = run - 1
        output, seconds, _, failed = harness.run_code(state, CHECKS, run)
        print(f"run {run}: {seconds} s")
        if failed:
            print(output)
            sys.exit(1)
        results.update({k: results.get(k, True) and v for k, v in json.loads(output.splitlines()[-1]).items()})
    results["kept_folder_persists"] = open(f"{work}/kept/note.txt").read() == "kept between runs"

    timeout_state = dict(state, timeout=3, id="example-timeout")
    output, _, _, failed = harness.run_code(timeout_state, "import time; time.sleep(60)", 1)
    results["runaway_code_is_killed"] = failed and "timed out" in output

    for check, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL'}  {check}")
    sys.exit(0 if all(results.values()) else 1)
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(host_only, ignore_errors=True)
