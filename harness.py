#!/usr/bin/env python3
"""Blog-style harness for Claude and Astra, built like the one behind "Sparks of Weight-Space Programming"
(Experiments 3, 7 and 8; see lm_understand-logs/README.md and lm_understand-logs/harness/).

The model gets NO tools of its own. Every turn is one call of its CLI with every tool switched off and the system
instructions "Answer the user's request." To compute, the model replies with one ```python block; we run that block
in a sandbox and send the printed output back as the next message ("OUTPUT of run k of at most N:"). A reply without
a code block ends the current part (the last part's reply is the final answer).

  chat  Each turn runs the CLI in a throwaway Docker container, which removes itself afterwards:
          Claude: claude -p --tools "" --system-prompt "Answer the user's request."
          Astra:  codex exec / codex exec resume, with the blog's config (every tool feature off, web search off,
                  "Answer the user's request." as the instructions, no auto-compaction)
        The conversation is kept on disk in the run's work folder (~/.sandbox-runs/run-XXXXXX), so the next turn resumes
        it. The container sees none of your files; the model can only talk.
  code  Each run is a fresh throwaway container: no network, your GPU, 600 s, limited RAM, output cut at 20,000
        characters, a folder holding weights.bin and any data files (read-only) plus any folders the job keeps between
        runs (experiment 4's edits). Before the model's code, the job's prelude runs.

A job (--job job.json) can have several parts ("phases"), each with its own message, run budget, kind and answer check:
  {"agent": "claude"|"codex", "effort": "max", "weights": "/abs/weights.bin" | null,
   "files": {"galaxies.npy": "/abs/..."}, "persist": ["edits"], "prelude": "python code",
   "timeout": 600, "mem": "12g",
   "phases": [{"name": "study", "message": "...", "max_runs": 60, "kind": "chat" | "single_code",
               "validate": null | {"type": "moments", "ids": [...], "max_words": 80} | {"type": "json"} | {"type": "account"}}],
   "evaluate": null | {"budget": 10, "cmd": [... "{edits}" ...]}}
  kind "single_code": the reply must be code calling submit(...); the part ends as soon as one run submits without an
  error (experiment-2 readers). EVALUATE: a reply of just that word runs evaluate.cmd on the host (experiment 4).

Exit codes: 0 finished, 1 failed (still failing after retries), 65 over the output cap even at the fallback effort
(not worth retrying), 73 another process is running this run,
75 usage limit (state.json limit_until says when it resets), 130 stopped (Ctrl-C or kill; state saved).

usage:
  python3 harness.py claude prompts/harness_bare.txt claude-bare              (one-part run, as before)
  python3 harness.py --job job.json --run-dir DIR
  python3 harness.py --continue DIR                                             picks up a stopped run where it was
options for one-part runs: --effort max, --runs 100, --timeout 600, --weights W, --run-dir PATH, --mem 24g
testing: HARNESS_FAKE_LIMIT_AFTER=<turn> [HARNESS_FAKE_LIMIT_SECONDS=60] [HARNESS_FAKE_LIMIT_JOB=<part of the job
name>] pretends a usage limit once before that turn
"""

import argparse
import datetime
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
USER = os.environ.get("USER", "user")
UID, GID = os.getuid(), os.getgid()
PY = f"{HOME}/.sandbox-python"
IMAGE = "sandbox-host:1"


def find_codex_dir():
    """The folder of Codex's native binary: $CODEX_DIR, or found from the codex on PATH (Linux x64)."""
    if os.environ.get("CODEX_DIR"):
        return os.environ["CODEX_DIR"]
    found = shutil.which("codex")
    if not found:
        return ""
    package = os.path.dirname(os.path.dirname(os.path.realpath(found)))
    hits = glob.glob(f"{package}/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl")
    return hits[0] if hits else ""


CODEX_DIR = find_codex_dir()
MODELS = {"claude": "claude-fable-5-1", "codex": "gpt-6-astra"}
INSTRUCTIONS = "Answer the user's request."
MAX_OUT, MEM, THREADS = 20_000, "24g", "8"
GPU_DEVICES = ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"]
RETRY_WAITS = [60, 180, 600]  # seconds between attempts when a turn fails for another reason; then we stop
EXIT_FAILED, EXIT_OUTPUT_CAP, EXIT_LOCKED, EXIT_LIMIT, EXIT_STOPPED = 1, 65, 73, 75, 130
CLAUDE_MAX_OUTPUT = "128000"  # Claude Code's default of 64,000 is too low at max effort
OUTPUT_CAP = re.compile(r"exceeded the \d+ output token maximum", re.I)
FALLBACK_EFFORT = "xhigh"  # a max-effort turn that hits the output cap is retried once at this effort, and recorded
SUBMIT_MARK = "<<<SUBMISSIONS>>>"
LIMIT_WORDS = re.compile(r"usage limit|rate.?limit|limit reached|hit your limit|too many requests|\b429\b|quota|"
                         r"out of (?:credits|extra usage)", re.I)

# The blog's harness/codex_config.toml, with Astra as the model and our instructions file.
CODEX_CONFIG = f'''# Default = "no tools": every built-in capability that gives the model a tool is switched off.
model = "gpt-6-astra"
model_reasoning_effort = "max"
approval_policy = "never"
sandbox_mode = "read-only"
web_search = "disabled"
model_instructions_file = "{HOME}/.codex/instructions.md"

[agents]
enabled = false

[features]
code_mode = false
collaboration_modes = false
shell_tool = false
unified_exec = false
unified_exec_tty = false
shell_snapshot = false
view_image = false
image_generation = false
multi_agent = false
apps = false
plugins = false
remote_plugin = false
browser_use = false
browser_use_external = false
browser_use_full_cdp_access = false
in_app_browser = false
computer_use = false
code_mode_host = false
skill_search = false
skill_mcp_dependency_install = false
sleep_tool = false
tool_suggest = false
goals = false
hooks = false
worktrees = false
workspace_dependencies = false
'''

# Defined before every reader's code (experiment 2): collects what the reader submits.
SUBMIT_PRELUDE = f'''
import atexit as _atexit, json as _json
_SUBMITTED = {{}}
def submit(moment, patch=None, deps=None):
    """moment: e.g. "M01"; patch: 16 x 16 x 3 array (rows, columns, red/green/blue); deps: list of (row, col, weight)."""
    entry = {{"patch": None, "deps": []}}
    if patch is not None:
        p = np.asarray(patch, dtype=float)
        if p.shape != (16, 16, 3):
            raise ValueError(f"{{moment}}: the patch must have shape (16, 16, 3), not {{p.shape}}")
        if not np.isfinite(p).all():
            raise ValueError(f"{{moment}}: the patch has non-finite numbers")
        entry["patch"] = np.round(p, 5).tolist()
    for item in deps or []:
        r, c, w = item
        entry["deps"].append([int(r), int(c), float(w)])
    _SUBMITTED[str(moment)] = entry
@_atexit.register
def _dump():
    print("\\n{SUBMIT_MARK}" + _json.dumps(_SUBMITTED))
'''

running = set()  # names of our containers that are running now, killed on Ctrl-C


def now():
    return datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def log(rd, text):
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with open(f"{rd}/loop.log", "a") as f:
        f.write(line + "\n")


def show(tag, text, lines=25):
    text = (text or "").rstrip()
    if not text:
        return
    rows = text.splitlines()
    more = f"\n    ... [{len(rows) - lines} more lines in the logs]" if len(rows) > lines else ""
    print(f"  {tag}\n" + "\n".join("    " + r for r in rows[:lines]) + more, flush=True)


def save(rd, st):
    with open(f"{rd}/state.json.tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(f"{rd}/state.json.tmp", f"{rd}/state.json")


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


# ------------------------------------------------------------------ usage limits
def parse_reset(text, default_wait=1800):
    """When a usage limit ends, from an error message: '|<epoch>', 'try again in 2 hours 3 minutes', 'resets at 11pm'."""
    t = time.time()
    m = re.search(r"\|(\d{10})\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:try again|resets?|available) in\s+((?:\d+\s*(?:days?|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b"
                  r"[\s,and]*)+)", text, re.I)
    if m:
        unit = {"d": 86400, "h": 3600, "m": 60, "s": 1}
        secs = sum(int(n) * unit[u[0].lower()] for n, u in re.findall(r"(\d+)\s*([a-zA-Z]+)", m.group(1)))
        if secs:
            return int(t + secs)
    m = re.search(r"(?:try again|resets?|available)[^0-9]{0,25}(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)", text, re.I)
    if m:
        h, mi, ap = int(m.group(1)) % 12, int(m.group(2) or 0), m.group(3).lower()
        h += 12 if ap.startswith("p") else 0
        d = datetime.datetime.now().replace(hour=h, minute=mi, second=0, microsecond=0)
        if d.timestamp() <= t:
            d += datetime.timedelta(days=1)
        return int(d.timestamp())
    return int(t + default_wait)


# ------------------------------------------------------------------ chat: one CLI call per turn, in a container
def common(name):
    return ["docker", "run", "--rm", "-i", "--init", "--name", name, "--user", f"{UID}:{GID}", "--hostname", "sandbox",
            "-v", "/usr:/usr:ro", "-v", "/etc:/etc:ro",
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev,mode=1777,size=8g",
            "--tmpfs", f"{HOME}:rw,exec,nosuid,nodev,uid={UID},gid={GID},mode=0700",
            "--tmpfs", f"/work:rw,nosuid,nodev,uid={UID},gid={GID},mode=0700", "-w", "/work",
            "-e", f"HOME={HOME}", "-e", f"USER={USER}", "-e", f"LOGNAME={USER}", "-e", "LANG=C.UTF-8"]


def claude_cmd(st, name):
    claude_bin = os.path.realpath(shutil.which("claude"))
    a = common(name) + ["-v", f"{claude_bin}:/opt/claude/bin/claude:ro", "-e", "DISABLE_AUTOUPDATER=1",
                        "-e", f"CLAUDE_CODE_MAX_OUTPUT_TOKENS={CLAUDE_MAX_OUTPUT}",
                        "-v", f"{HOME}/.claude:{HOME}/.claude"]
    for d in sorted(os.listdir(f"{HOME}/.claude")):  # hide every subfolder of your ~/.claude ...
        p = f"{HOME}/.claude/{d}"
        if os.path.isdir(p) and not os.path.islink(p) and d != "projects":
            a += ["--tmpfs", f"{p}:rw,exec,nosuid,nodev,uid={UID},gid={GID},mode=0700"]
    a += ["-v", f"{st['work']}/claude-projects:{HOME}/.claude/projects"]  # ... projects = this run's conversation only
    if os.path.exists(f"{HOME}/.claude/history.jsonl"):
        a += ["-v", f"{st['work']}/history.jsonl:{HOME}/.claude/history.jsonl"]
    a += ["-e", "PATH=/opt/claude/bin:/usr/bin:/bin", IMAGE,
          "claude", "-p", "--model", st["model"], "--effort", st["effort"], "--tools", "",
          "--system-prompt", INSTRUCTIONS, "--strict-mcp-config", "--disable-slash-commands",
          "--output-format", "stream-json", "--verbose"]
    if st.get("session"):
        a += ["--resume", st["session"]]
    return a


def codex_cmd(st, name):
    a = common(name) + ["-v", f"{CODEX_DIR}:/opt/codex:ro",
                        "-v", f"{st['work']}/codex-home:{HOME}/.codex",
                        "-v", f"{HOME}/.codex/auth.json:{HOME}/.codex/auth.json",
                        "-e", f"CODEX_HOME={HOME}/.codex", "-e", "PATH=/opt/codex/bin:/usr/bin:/bin", IMAGE,
                        "codex", "exec"]
    if st.get("thread"):
        a += ["resume"]
    a += ["--strict-config", "--skip-git-repo-check", "--json", "-m", st["model"],
          "-c", f'model_reasoning_effort="{st["effort"]}"', "-c", "model_context_window=872000",
          "-c", "model_auto_compact_token_limit=1000000",
          "-c", f'model_instructions_file="{HOME}/.codex/instructions.md"']
    if st.get("thread"):
        a += [st["thread"]]
    return a + ["-"]


def one_call(st, message, tdir, attempt):
    """Run one CLI call. Returns (reply text or None, record); record["limited_until"] is set on a usage limit."""
    name = f"bh-{st['id']}-t{st['turn'] + 1:03d}-{attempt}"
    cmd = claude_cmd(st, name) if st["agent"] == "claude" else codex_cmd(st, name)
    rec = {"agent": st["agent"], "model": st["model"], "effort": st["effort"], "attempt": attempt,
           "cmd": cmd[cmd.index(IMAGE) + 1:], "tool_items": [], "errors": []}
    texts, result, last_reset = [], None, None
    t0 = time.time()
    running.add(name)
    with open(f"{tdir}/events.jsonl", "a") as ev_f, open(f"{tdir}/stderr.log", "a") as err_f:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_f, text=True)
        p.stdin.write(message)
        p.stdin.close()
        for line in p.stdout:
            ev_f.write(line)
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type", "")
            if st["agent"] == "claude":
                if t == "system" and ev.get("subtype") == "init":
                    rec["session"] = ev.get("session_id")
                    rec["tools_offered"] = ev.get("tools")
                    rec["mcp_servers"] = ev.get("mcp_servers")
                elif t == "rate_limit_event":
                    info = ev.get("rate_limit_info") or {}
                    last_reset = info.get("resetsAt") or last_reset
                    if info.get("status") not in (None, "allowed", "allowed_warning"):
                        rec["limited_until"] = int(info.get("resetsAt") or time.time() + 1800)
                elif t == "assistant":
                    for c in ev.get("message", {}).get("content", []):
                        if c.get("type") == "thinking" and c.get("thinking"):
                            show("[thinking]", c["thinking"], 6)
                        elif c.get("type") == "tool_use":
                            rec["tool_items"].append(c.get("name"))
                elif t == "result":
                    result = ev
                    rec["session"] = ev.get("session_id") or rec.get("session")
            else:
                item = ev.get("item") or {}
                if t == "thread.started":
                    rec["thread"] = ev.get("thread_id")
                elif t == "item.completed":
                    it = item.get("type")
                    if it == "agent_message":
                        texts.append(item.get("text", ""))
                    elif it == "reasoning":
                        show("[thinking]", item.get("text", ""), 6)
                    elif it == "error":
                        rec["errors"].append(item.get("message", ""))
                    else:
                        rec["tool_items"].append(it)
                elif t == "turn.completed":
                    rec["usage"] = ev.get("usage")
                elif t in ("turn.failed", "error"):
                    rec["errors"].append(json.dumps(ev)[:2000])
        p.wait()
    running.discard(name)
    rec["rc"], rec["wall_s"] = p.returncode, round(time.time() - t0, 1)
    if st["agent"] == "claude" and result is not None:
        rec["usage"], rec["cost_usd_api_equivalent"] = result.get("usage"), result.get("total_cost_usd")
        if result.get("subtype") == "success" and not result.get("is_error"):
            texts = [result.get("result", "")]
        else:
            rec["errors"].append(json.dumps({k: result.get(k) for k in ("subtype", "is_error", "result")})[:2000])
    reply = "\n\n".join(x for x in texts if x).strip()
    ok = p.returncode == 0 and bool(reply)
    if not ok and "limited_until" not in rec:
        with open(f"{tdir}/stderr.log") as f:
            stderr_tail = f.read()[-4000:]
        problem = " ".join(rec["errors"]) + " " + stderr_tail
        if LIMIT_WORDS.search(problem):
            rec["limited_until"] = parse_reset(problem) if not last_reset else int(last_reset)
    rec["output_cap"] = not ok and bool(OUTPUT_CAP.search(" ".join(rec["errors"])))
    rec["tool_free"] = not rec["tool_items"]
    return (reply if ok else None), rec


# ------------------------------------------------------------------ code: one fresh sandbox per run
def run_code(st, code, k):
    """Returns (output shown to the model, seconds, submissions or None, failed?)."""
    name = f"bh-{st['id']}-r{st['runs'] + 1:03d}"
    mem = st.get("mem", MEM)
    cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--user", f"{UID}:{GID}",
           "--hostname", "sandbox", "--memory", mem, "--memory-swap", mem, "--pids-limit", "4096",
           "-v", "/usr:/usr:ro", "-v", "/etc:/etc:ro", "-v", f"{PY}:{PY}:ro",
           "--tmpfs", "/tmp:rw,exec,nosuid,nodev,mode=1777,size=8g",
           "--tmpfs", f"/work:rw,exec,nosuid,nodev,uid={UID},gid={GID},mode=0700", "-w", "/work",
           "-e", "HOME=/tmp", "-e", f"PATH={PY}/env/bin:/usr/bin:/bin", "-e", "LANG=C.UTF-8",
           "-e", f"OMP_NUM_THREADS={THREADS}", "-e", f"MKL_NUM_THREADS={THREADS}"]
    for f in st.get("mounted_files", []):  # weights.bin and data files, linked into the neutral work folder
        cmd += ["-v", f"{st['work']}/{f}:/work/{f}:ro"]
    for d in st.get("persist", []):
        cmd += ["-v", f"{st['work']}/{d}:/work/{d}"]
    cmd += [x for d in GPU_DEVICES if os.path.exists(d) for x in ("--device", d)]
    cmd += [IMAGE, f"{PY}/env/bin/python", "-I", "-"]
    t0 = time.time()
    running.add(name)
    failed, captured = False, None
    try:
        p = subprocess.run(cmd, input=st["prelude"] + "\n" + code, capture_output=True, text=True,
                           timeout=st["timeout"])
        stdout = p.stdout
        if SUBMIT_MARK in stdout:
            stdout, _, blob = stdout.rpartition(SUBMIT_MARK)
            try:
                captured = json.loads(blob.strip().splitlines()[0])
            except (json.JSONDecodeError, IndexError):
                captured = None
        out = stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        failed = p.returncode != 0
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        out, failed = f"[timed out after {st['timeout']} s]", True
    finally:
        running.discard(name)
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    return out.strip() or "(no output)", round(time.time() - t0, 1), captured, failed


# ------------------------------------------------------------------ answer checks
def check(spec, reply):
    """Returns (problems, parsed)."""
    if not spec:
        return [], reply
    if spec["type"] == "json":
        m = re.search(r"\{.*\}", reply, re.S)
        try:
            obj = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict):
            return ["Your reply must be only a JSON object, as asked."], None
        return [], obj
    if spec["type"] == "account":
        probs = [f"Your reply must contain a line starting with {k}" for k in ("ACCOUNT:", "CLAIMS:") if k not in reply]
        return probs, reply
    if spec["type"] == "moments":
        texts = {}
        for line in reply.splitlines():
            m = re.match(r"^[^A-Za-z0-9]*(M\d{2})[*_\s]*[:\-–—][*_\s]*(.+)$", line.strip())
            if m and m.group(1) in spec["ids"]:
                texts[m.group(1)] = m.group(2).strip().strip("*").strip()
        probs = [f"{i} is missing" for i in spec["ids"] if i not in texts]
        for i, t in texts.items():
            words = len(t.split())
            if words > spec.get("max_words", 80):
                probs.append(f"{i} has {words} words (at most {spec.get('max_words', 80)})")
            if re.search(r"\d", t):
                probs.append(f"{i} contains digits")
        return probs, texts
    return [], reply


# ------------------------------------------------------------------ the conversation loop
def phase_of(st):
    return st["phases"][st["phase"]]


def finish_phase(rd, st, reply, parsed, problems, forced=False):
    i, ph = st["phase"], phase_of(st)
    stem = f"{rd}/phase_{i + 1:02d}_{ph['name']}"
    write(stem + ".md", reply if isinstance(reply, str) else json.dumps(reply))
    with open(stem + ".json", "w") as f:
        json.dump({"name": ph["name"], "parsed": parsed, "problems": problems, "forced": forced,
                   "runs": st["phase_runs"]}, f, indent=1)
    log(rd, f"part {i + 1} ({ph['name']}) done after {st['phase_runs']} runs" + (f"; problems: {problems}" if problems else ""))
    st.update(phase_runs=0, fix=0, over=0)
    if i + 1 < len(st["phases"]):
        st["phase"] = i + 1
        st["next"] = phase_of(st)["message"]
    else:
        st["done"], st["next"] = True, None
        write(f"{rd}/final_answer.md", reply if isinstance(reply, str) else json.dumps(reply))
        log(rd, f"finished after {st['runs']} runs and {st['turn']} turns, tool-free {st['tool_free']}")


def evaluate(rd, st):
    ev = st["evaluate"]
    cmd = [x.replace("{edits}", f"{st['work']}/edits") for x in ev["cmd"]]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=ev.get("timeout", 3600))
        line = (p.stdout.strip().splitlines() or ["(no result)"])[-1]
        if p.returncode != 0:
            line = "the evaluation failed: " + (p.stderr.strip().splitlines() or ["?"])[-1]
    except subprocess.TimeoutExpired:
        line = "the evaluation timed out"
    with open(f"{rd}/evaluations.jsonl", "a") as f:
        f.write(json.dumps({"turn": st["turn"], "n": st["evals"] + 1, "s": round(time.time() - t0, 1), "result": line}) + "\n")
    return line


def loop(rd, st):
    who = "Claude" if st["agent"] == "claude" else "Astra"
    fake_after = int(os.environ.get("HARNESS_FAKE_LIMIT_AFTER", "0") or 0)
    if os.environ.get("HARNESS_FAKE_LIMIT_JOB") and os.environ["HARNESS_FAKE_LIMIT_JOB"] not in st.get("name", ""):
        fake_after = 0
    while not st.get("done"):
        t = st["turn"] + 1
        if fake_after and t == fake_after and not st.get("faked_limit"):
            st["faked_limit"], st["limit_until"] = True, int(time.time() + int(os.environ.get("HARNESS_FAKE_LIMIT_SECONDS", "60")))
            save(rd, st)
            log(rd, f"turn {t}: (test) pretending a usage limit until {time.strftime('%H:%M:%S', time.localtime(st['limit_until']))}")
            sys.exit(EXIT_LIMIT)
        tdir = f"{rd}/turns/t{t:03d}"
        os.makedirs(tdir, exist_ok=True)
        write(f"{tdir}/prompt.txt", st["next"])
        resumed = st.get("session") or st.get("thread")
        log(rd, f"turn {t}: {'resume' if resumed else 'new conversation'} ({len(st['next']):,} chars sent)")
        reply = None
        for attempt, wait in enumerate([0] + RETRY_WAITS):
            if wait:
                log(rd, f"turn {t}: the call failed ({'; '.join(rec['errors'])[:300] or 'rc ' + str(rec['rc'])}); "
                        f"trying again in {wait} s")
                time.sleep(wait)
            reply, rec = one_call(st, st["next"], tdir, attempt)
            with open(f"{tdir}/run.json", "w") as f:
                json.dump(rec, f, indent=1)
            if reply is None and rec.get("output_cap"):  # retrying the same settings cannot help
                if st["effort"] != "max":
                    save(rd, st)
                    log(rd, f"turn {t}: over the output cap at effort {st['effort']}; giving up on this job")
                    sys.exit(EXIT_OUTPUT_CAP)
                st["effort_fallback"] = {"from": st["effort"], "to": FALLBACK_EFFORT, "turn": t}
                st["effort"] = FALLBACK_EFFORT
                save(rd, st)
                log(rd, f"turn {t}: over the output cap at max effort; retrying once at {FALLBACK_EFFORT} (recorded)")
                reply, rec = one_call(st, st["next"], tdir, "fallback")
                with open(f"{tdir}/run.json", "w") as f:
                    json.dump(rec, f, indent=1)
                if reply is None and rec.get("output_cap"):
                    save(rd, st)
                    log(rd, f"turn {t}: over the output cap at {FALLBACK_EFFORT} too; giving up on this job")
                    sys.exit(EXIT_OUTPUT_CAP)
            if reply is not None or rec.get("limited_until"):
                break
        if reply is None:
            if rec.get("limited_until"):
                st["limit_until"] = rec["limited_until"]
                save(rd, st)
                log(rd, f"turn {t}: usage limit; it resets at "
                        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st['limit_until']))}. Stopped here; continue then.")
                sys.exit(EXIT_LIMIT)
            save(rd, st)
            log(rd, f"turn {t}: still failing, stopped. Continue later with: python3 harness.py --continue {rd}")
            sys.exit(EXIT_FAILED)
        # only a turn that went through moves the conversation on (a failed or stopped first turn starts afresh)
        st["session"] = rec.get("session") or st.get("session")
        st["thread"] = rec.get("thread") or st.get("thread")
        st["turn"], st["limit_until"] = t, None
        st["tool_free"] = st.get("tool_free", True) and rec["tool_free"]
        write(f"{tdir}/answer.md", reply)
        ph = phase_of(st)
        blocks = re.findall(r"```python[^\n]*\n(.*?)```", reply, re.S)
        log(rd, f"turn {t}: {who} replied in {rec['wall_s']} s ({len(reply):,} chars, {len(blocks)} code "
                f"block{'s' if len(blocks) != 1 else ''}), part {st['phase'] + 1} of {len(st['phases'])} ({ph['name']}), "
                f"tool-free {rec['tool_free']}")

        if st.get("evaluate") and re.fullmatch(r"\W*EVALUATE\W*", reply.strip(), re.I):
            budget = st["evaluate"]["budget"]
            if st["evals"] < budget:
                line = evaluate(rd, st)
                st["evals"] += 1
                show(f"[evaluation {st['evals']} of {budget}]", line)
                st["next"] = f"EVALUATION {st['evals']} of at most {budget} (validation galaxies): {line}"
            else:
                st["next"] = "You have no evaluations left."
            save(rd, st)
            continue

        if blocks:
            if st["phase_runs"] >= ph["max_runs"]:
                st["over"] = st.get("over", 0) + 1
                if st["over"] > 2:
                    finish_phase(rd, st, reply, None, ["kept sending code with no runs left"], forced=True)
                else:
                    st["next"] = ("You have no runs left for this part. Reply WITHOUT a code block, as asked."
                                  if ph.get("kind") != "single_code" else
                                  "No more runs are possible for this part.")
                save(rd, st)
                continue
            k = st["phase_runs"] + 1
            show(f"[code, run {k} of at most {ph['max_runs']}]", blocks[0])
            out, secs, captured, failed = run_code(st, blocks[0], k)
            show(f"[output, {len(out):,} chars, {secs} s]", out)
            st["phase_runs"], st["runs"] = k, st["runs"] + 1
            with open(f"{rd}/tool_calls.jsonl", "a") as f:
                f.write(json.dumps({"turn": t, "part": st["phase"] + 1, "run": k, "s": secs, "code": blocks[0],
                                    "out": out, "submitted": captured}) + "\n")
            if ph.get("kind") == "single_code":
                if captured and not failed:
                    finish_phase(rd, st, reply, captured, [])
                else:
                    st["fix"] = st.get("fix", 0) + 1
                    if st["fix"] > 2:
                        finish_phase(rd, st, reply, captured or {}, ["code kept failing"], forced=True)
                    else:
                        st["next"] = (f"Your code {'failed' if failed else 'submitted nothing'}. Output:\n```\n{out}\n```\n"
                                      "Reply with one corrected code block that calls submit() for every moment.")
                save(rd, st)
                continue
            note = f"(Your reply had {len(blocks)} code blocks; only the first one was run.)\n" if len(blocks) > 1 else ""
            st["next"] = f"{note}OUTPUT of run {k} of at most {ph['max_runs']}:\n```\n{out}\n```"
            save(rd, st)
            continue

        if ph.get("kind") == "single_code":
            st["fix"] = st.get("fix", 0) + 1
            if st["fix"] > 2:
                finish_phase(rd, st, reply, {}, ["never sent code"], forced=True)
            else:
                st["next"] = "Reply with exactly one ```python code block that calls submit() for every moment."
            save(rd, st)
            continue
        problems, parsed = check(ph.get("validate"), reply)
        if problems and st.get("fix", 0) < 2:
            st["fix"] = st.get("fix", 0) + 1
            st["next"] = ("Please send the whole answer again, fixing these problems: " + "; ".join(problems) + ".")
            log(rd, f"turn {t}: answer has problems, asked again: {problems}")
            save(rd, st)
            continue
        finish_phase(rd, st, reply, parsed, problems)
        save(rd, st)
        if st.get("done"):
            print("\n" + reply + "\n", flush=True)


# ------------------------------------------------------------------ setting up a run
def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def setup(job, rd):
    rd = os.path.abspath(rd)
    os.makedirs(f"{rd}/turns")
    os.makedirs(f"{HOME}/.sandbox-runs", exist_ok=True)
    work = tempfile.mkdtemp(prefix="run-", dir=f"{HOME}/.sandbox-runs")  # neutral path: the code can see it
    os.symlink(work, f"{rd}/work")
    mounted = []
    if job.get("weights"):  # a hard link costs no space; the code only ever sees it read-only
        link_or_copy(job["weights"], f"{work}/weights.bin")
        layout = json.load(open(os.path.join(os.path.dirname(job["weights"]), "layout.json")))
        assert sha256(f"{work}/weights.bin") == layout["sha256"], f"{job['weights']} does not match its layout.json"
        mounted.append("weights.bin")
    for name, src in (job.get("files") or {}).items():
        link_or_copy(src, f"{work}/{name}")
        mounted.append(name)
    for d in job.get("persist") or []:
        os.makedirs(f"{work}/{d}")
    agent = job["agent"]
    if agent == "claude":
        os.makedirs(f"{work}/claude-projects")
        open(f"{work}/history.jsonl", "w").close()
    else:
        os.makedirs(f"{work}/codex-home")
        write(f"{work}/codex-home/config.toml", CODEX_CONFIG)
        write(f"{work}/codex-home/instructions.md", INSTRUCTIONS + "\n")
        open(f"{work}/codex-home/auth.json", "w").close()  # mount point; your real auth.json is mounted on it
    with open(f"{rd}/job.json", "w") as f:
        json.dump(job, f, indent=1)
    st = {"agent": agent, "model": job.get("model") or MODELS[agent], "effort": job.get("effort", "max"),
          "name": job.get("name", os.path.basename(rd)), "id": os.path.basename(work).removeprefix("run-").lower(),
          "work": work, "mounted_files": mounted, "persist": job.get("persist") or [], "prelude": job.get("prelude", ""),
          "timeout": job.get("timeout", 600), "mem": job.get("mem", MEM), "phases": job["phases"],
          "evaluate": job.get("evaluate"), "phase": 0, "phase_runs": 0, "fix": 0, "over": 0, "turn": 0, "runs": 0,
          "evals": 0, "session": None, "thread": None, "next": job["phases"][0]["message"], "done": False,
          "tool_free": True, "limit_until": None, "started": now()}
    save(rd, st)
    return rd, st


def one_part_job(args):
    prompt = open(args.prompt).read()
    m = re.search(r"```python\n(.*?)```", prompt, re.S)
    return {"agent": args.agent, "effort": args.effort, "name": args.name, "weights": os.path.abspath(args.weights),
            "prompt_file": os.path.abspath(args.prompt), "timeout": args.timeout, "mem": args.mem,
            "prelude": "import numpy as np\n" + (m.group(1) + "\nW = load_weights()\n" if m else ""),
            "phases": [{"name": "main", "message": prompt, "max_runs": args.runs, "kind": "chat"}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", nargs="?", choices=["claude", "codex"])
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--effort", default="max")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--weights", default=f"{HERE}/weights.bin")
    ap.add_argument("--run-dir")
    ap.add_argument("--mem", default=MEM)
    ap.add_argument("--job")
    ap.add_argument("--continue", dest="cont", metavar="RUN_DIR")
    args = ap.parse_args()
    if args.cont:
        rd = os.path.abspath(args.cont)
        st = json.load(open(f"{rd}/state.json"))
        if st.get("done"):
            print(f"{rd} is already finished", flush=True)
            sys.exit(0)
    elif args.job:
        if not args.run_dir:
            ap.error("--job needs --run-dir")
        rd, st = setup(json.load(open(args.job)), args.run_dir)
    elif args.agent and args.prompt and args.name:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        rd, st = setup(one_part_job(args), args.run_dir or f"{HERE}/blog_runs/{args.name}-{stamp}")
    else:
        ap.error("give: agent prompt name, or --job JOB --run-dir DIR, or --continue RUN_DIR")
    lock = open(f"{rd}/.lock", "w")  # one process per run folder: a second --continue on a live run refuses
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{rd} is already being run by another process", flush=True)
        sys.exit(EXIT_LOCKED)
    print(f"run folder: {rd}\nmodel: {st['model']}, effort {st['effort']}, {len(st['phases'])} part(s), "
          f"code runs of up to {st['timeout']} s", flush=True)

    def on_term(signum, frame):  # treat a plain kill like Ctrl-C: stop the containers, save the state
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    try:
        loop(rd, st)
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not cut the clean-up short
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for name in list(running):
            subprocess.run(["docker", "kill", name], capture_output=True)
        save(rd, st)
        print(f"\nstopped. Continue with: python3 harness.py --continue {rd}", flush=True)
        sys.exit(EXIT_STOPPED)


if __name__ == "__main__":
    main()
