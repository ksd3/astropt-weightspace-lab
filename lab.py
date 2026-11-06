#!/usr/bin/env python3
"""The lab: runs the experiments on a list of AstroPT checkpoints, unattended, for as long as it takes, then winds
itself down.

  python3 lab.py start CONFIG.toml [--survive-reboot]   start in the background (survives closing the terminal;
                                                        with --survive-reboot also restarts after a reboot)
  python3 lab.py status [CONFIG.toml]                   print the status file, lab_runs/<name>/STATUS.md
  python3 lab.py wake [CONFIG.toml]                     stop waiting (usage limits, retries) and try again now
  python3 lab.py stop [CONFIG.toml]                     stop cleanly: every job saves its place; start again to resume
  python3 lab.py run CONFIG.toml                        run in the foreground (start runs this under a supervisor)

How it runs
  - Every checkpoint is prepared once (ckpt_prepare.py: numbers-only weights, checks, measured facts), at most
    resources.prepare_ahead checkpoints ahead of the agents, and only if the disk has room.
  - Jobs run in lanes: one per account (claude, codex) and one for the GPU (preparing, scoring). Each account only
    ever has one conversation going, and only one GPU job runs at a time. Checkpoints are worked through in order.
  - A usage limit pauses that account's lane until the limit resets (the reset time comes from the CLI's own message);
    the other lane carries on. `wake` retries at once (for example after you reset a limit).
  - Any other failure is retried after 5 minutes, 30 minutes and 2 hours, then marked failed (STATUS.md says where its
    log is) and the jobs that depend on it are skipped.
  - If this process dies, the supervisor restarts it; each job resumes where it stopped. --survive-reboot adds a
    crontab @reboot line (removed again when the lab finishes).
  - When every job is done (and, in watch mode, no new checkpoint has appeared for watch_minutes), it deletes the
    converted weights (if resources.delete_weights_when_done), writes the final results and exits.
Everything is in lab_runs/<name>/: STATUS.md (the one file to watch), events.log, state.json, checkpoints/<id>/ (the
prepared checkpoints), jobs/<job>/ (each job's harness run: every message, reply, code run and answer), results.json.
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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ASTROPT_PY = os.environ.get("ASTROPT_PY", sys.executable)  # a Python with torch + astropt installed
RUNS = f"{HERE}/lab_runs"
LANES = ["claude", "codex", "gpu"]
TYPE_ORDER = {"prep": 0, "exp1": 1, "exp2e": 2, "exp4": 3, "exp2r": 4, "judge": 5, "exp4s": 6, "done": 7}
BACKOFF = [300, 1800, 7200]
CRON_MARK = "# astropt-lab:"
DEFAULTS = {
    "name": "lab", "checkpoints": [], "max_per_folder": 0, "agents": ["claude", "codex"], "effort": "max", "blind": True,
    "watch_minutes": 0,
    "exp1": {"run": True, "only": [], "prompts": ["bare", "full"], "runs": 100},
    "exp2": {"run": True, "only": [], "study_runs": 60, "batch_runs": 5, "moments": 40, "batch": 10,
             "readers": ["claude", "codex"], "reader_effort": "high", "conditions": ["desc", "account", "none", "shuffled"]},
    "exp4": {"run": True, "only": [], "levels": ["L0", "L1", "L2"], "runs": 50, "evaluations": 10},
    "judge": {"run": True, "judges": ["claude", "codex"], "effort": "high"},
    "resources": {"mem": "12g", "prepare_ahead": 1, "delete_weights_when_done": True, "min_free_disk_gb": 20,
                  "keep_downloads": False},
}


# ------------------------------------------------------------------ config, paths, small helpers
def merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path):
    import tomllib
    with open(path, "rb") as f:
        cfg = merge(DEFAULTS, tomllib.load(f))
    cfg["config_path"] = os.path.abspath(path)
    bad = [a for a in cfg["agents"] + cfg["exp2"]["readers"] + cfg["judge"]["judges"] if a not in ("claude", "codex")]
    if bad or not re.fullmatch(r"[A-Za-z0-9_.-]+", cfg["name"]) or not cfg["checkpoints"]:
        raise SystemExit(f"bad config {path}: agents must be claude/codex, name letters/digits, checkpoints non-empty")
    return cfg


def root_of(cfg):
    return f"{RUNS}/{cfg['name']}"


def find_root(arg):
    if arg and arg.endswith(".toml"):
        return root_of(load_config(arg))
    if arg:
        return f"{RUNS}/{arg}"
    roots = sorted(glob.glob(f"{RUNS}/*/state.json"), key=os.path.getmtime)
    if not roots:
        raise SystemExit("no lab run found; give the config file")
    return os.path.dirname(roots[-1])


def stamp(t=None):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t or time.time()))


def hm(t):
    return time.strftime("%H:%M", time.localtime(t))


def dur(s):
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def atomic_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def normalize(entry):
    """A pasted https://huggingface.co/[datasets/]owner/repo[/tree/main/folder] link -> hf://[datasets/]owner/repo/folder."""
    m = re.match(r"https?://huggingface\.co/(datasets/)?([^/]+/[^/]+?)(?:/(?:tree|blob|resolve)/[^/]+)?(/.*)?/?$",
                 entry.strip())
    return f"hf://{m.group(1) or ''}{m.group(2)}{(m.group(3) or '').rstrip('/')}" if m else entry.strip()


def hf_files(entry):
    """hf://[datasets/]owner/repo[/path] -> that file, or every .pt file under the folder / matching the glob."""
    import fnmatch
    prefix, kind = ("hf://datasets/", "dataset") if entry.startswith("hf://datasets/") else ("hf://", "model")
    parts = entry[len(prefix):].strip("/").split("/")
    repo, path = "/".join(parts[:2]), "/".join(parts[2:])
    if path.endswith(".pt") and not any(ch in path for ch in "*?["):
        return [entry]
    from huggingface_hub import HfApi
    try:
        names = HfApi().list_repo_files(repo, repo_type=kind)
    except Exception:  # a stale saved token can break even public repos: try without it
        names = HfApi(token=False).list_repo_files(repo, repo_type=kind)
    files = [f for f in names if f.endswith(".pt")]
    if any(ch in path for ch in "*?["):
        files = [f for f in files if fnmatch.fnmatch(f, path)]
    elif path:
        files = [f for f in files if f.startswith(path.rstrip("/") + "/")]
    return [f"{prefix}{repo}/{f}" for f in sorted(files, key=natural)]


def thin(sources, n):
    """At most n checkpoints from each folder, evenly spread through it (always the first and the last): from a folder
    of log-spaced training checkpoints this keeps a log-spaced subset."""
    if not n:
        return sources
    folders = {}
    for s in sources:
        folders.setdefault(s.rsplit("/", 1)[0], []).append(s)
    keep = set()
    for files in folders.values():
        idx = sorted({round(i * (len(files) - 1) / max(1, n - 1)) for i in range(n)}) if len(files) > n else range(len(files))
        keep.update(files[i] for i in idx)
    return [s for s in sources if s in keep]


def discover(entries, per_folder=0):
    """Checkpoint sources in order: files, folders (every .pt inside, in natural order), globs, Hugging Face repos
    (hf://[datasets/]owner/repo[/folder | /glob | /file.pt], or a pasted https://huggingface.co/... link), at most
    per_folder from each folder. Local files still being written (changed in the last 2 minutes) wait for the next scan."""
    out, missing = [], []
    for e in map(normalize, entries):
        if e.startswith("hf://"):
            try:
                found = hf_files(e)
            except Exception as err:  # offline, private, or a typo: report it and try again at the next scan
                found = []
                missing.append(f"{e} ({type(err).__name__})")
            out += found
            if not found and not any(m.startswith(e) for m in missing):
                missing.append(e)
            continue
        p = os.path.expanduser(e)
        if any(ch in p for ch in "*?["):
            files = sorted(glob.glob(p, recursive=True), key=natural)
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "**", "*.pt"), recursive=True), key=natural)
        elif os.path.isfile(p):
            files = [p]
        else:
            missing.append(e)
            continue
        out += [os.path.abspath(f) for f in files if os.path.isfile(f) and time.time() - os.path.getmtime(f) > 120]
    return thin(out, per_folder), missing


def ck_id(source):
    tail = "-".join(source.rstrip("/").split("/")[-2:])
    slug = re.sub(r"[^A-Za-z0-9]+", "-", tail.removesuffix(".pt")).strip("-")[-40:]
    return f"{slug}-{hashlib.sha1(source.encode()).hexdigest()[:6]}"


# ------------------------------------------------------------------ the plan: every job, its lane and what it needs
def wanted(cfg, exp, source):
    """An experiment's `only` list (parts of checkpoint paths) limits it to those checkpoints; empty = all of them."""
    only = cfg[exp].get("only") or []
    return cfg[exp]["run"] and (not only or any(s in source for s in only))


def plan(state, cfg):
    tasks = state["tasks"]

    def add(tid, typ, ck, lane, deps, **params):
        if tid not in tasks:
            tasks[tid] = {"id": tid, "type": typ, "ck": ck, "lane": lane, "deps": deps, "params": params,
                          "status": "pending", "attempts": 0, "not_before": 0, "note": "", "result": None}
    for c in state["checkpoints"]:
        ck = c["id"]
        prep = f"prep|{ck}"
        add(prep, "prep", ck, "gpu", [], source=c["source"])
        mine = [prep]
        if wanted(cfg, "exp1", c["source"]):
            for a in cfg["agents"]:
                for p in cfg["exp1"]["prompts"]:
                    t = f"exp1|{ck}|{a}|{p}"
                    add(t, "exp1", ck, a, [prep], agent=a, prompt=p)
                    mine.append(t)
                    if cfg["judge"]["run"]:
                        for j in cfg["judge"]["judges"]:
                            add(f"judge|{ck}|{a}|{p}|{j}", "judge", ck, j, [t], agent=j, of_agent=a, prompt=p)
                            mine.append(f"judge|{ck}|{a}|{p}|{j}")
        if wanted(cfg, "exp2", c["source"]):
            for a in cfg["agents"]:
                add(f"exp2e|{ck}|{a}", "exp2e", ck, a, [prep], agent=a)
                mine.append(f"exp2e|{ck}|{a}")
            for r in cfg["exp2"]["readers"]:
                for cond in cfg["exp2"]["conditions"]:
                    for a in (["-"] if cond == "none" else cfg["agents"]):
                        t = f"exp2r|{ck}|{r}|{cond}|{a}"
                        add(t, "exp2r", ck, r, [prep] if a == "-" else [f"exp2e|{ck}|{a}"], agent=r, condition=cond,
                            explainer=a)
                        mine.append(t)
        if wanted(cfg, "exp4", c["source"]):
            for a in cfg["agents"]:
                for lv in cfg["exp4"]["levels"]:
                    add(f"exp4|{ck}|{a}|{lv}", "exp4", ck, a, [prep], agent=a, level=lv)
                    add(f"exp4s|{ck}|{a}|{lv}", "exp4s", ck, "gpu", [f"exp4|{ck}|{a}|{lv}"], agent=a, level=lv)
                    mine += [f"exp4|{ck}|{a}|{lv}", f"exp4s|{ck}|{a}|{lv}"]
        add(f"done|{ck}", "done", ck, "cpu", [t for t in mine if t != prep] or [prep])


# ------------------------------------------------------------------ the lab
class Lab:
    def __init__(self, cfg):
        self.cfg, self.root = cfg, root_of(cfg)
        for d in ("checkpoints", "jobs", "results"):
            os.makedirs(f"{self.root}/{d}", exist_ok=True)
        self.children = {}  # task id -> Popen
        self.stopping = False
        self.last_status = 0.0
        self.dirty = True
        path = f"{self.root}/state.json"
        if os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)
        else:
            self.state = {"name": cfg["name"], "started": time.time(), "checkpoints": [], "tasks": {}, "events": [],
                          "lanes": {l: {"paused_until": 0, "why": ""} for l in LANES}, "last_new": time.time(),
                          "missing": []}
        self.state["config_path"] = cfg["config_path"]
        for t in self.state["tasks"].values():  # a job that was running when the lab last stopped resumes; if its
            if t["status"] == "running":       # old process is still alive (the lab crashed), it is adopted
                t["status"] = "pending"
                t["elsewhere"] = self.held_elsewhere(t)
                if t["elsewhere"]:
                    t["note"] = "still running from before the lab restarted; waiting for it"

    # -- jobs still running in a process this lab did not start (it crashed and restarted)
    def lock_path(self, t):
        if t["type"] == "prep":
            return f"{self.ckdir(t['ck'])}/.lock"
        if t["type"] == "exp4s":
            return f"{self.jobdir(t)}/.lock"
        return f"{self.jobdir(t)}/run/.lock"

    def held_elsewhere(self, t):
        p = self.lock_path(t)
        if not os.path.exists(p):
            return False
        with open(p, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                return False
            except BlockingIOError:
                return True

    def lane_busy(self, lane):
        if any(self.state["tasks"][i]["lane"] == lane for i in self.children):
            return True
        for t in self.state["tasks"].values():
            if t["lane"] == lane and t.get("elsewhere"):
                if self.held_elsewhere(t):
                    return True
                t["elsewhere"], t["not_before"], t["note"] = False, 0, ""
                self.dirty = True
        return False

    def already_done(self, t):
        """Work a dead process finished: a prepared checkpoint or a score file, from this job's own source."""
        if t["type"] == "prep":
            p = f"{self.ckdir(t['ck'])}/meta.json"
            if os.path.exists(p):
                with open(p) as f:
                    return json.load(f).get("source") == t["params"]["source"]
        if t["type"] == "exp4s":
            return os.path.exists(f"{self.jobdir(t)}/score.json")
        return False

    # -- bookkeeping
    def save(self):
        atomic_json(f"{self.root}/state.json", self.state)

    def event(self, text):
        line = f"{stamp()}  {text}"
        self.state["events"] = (self.state["events"] + [line])[-40:]
        with open(f"{self.root}/events.log", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
        self.dirty = True

    def ckdir(self, ck):
        return f"{self.root}/checkpoints/{ck}"

    def ck_label(self, ck):
        p = f"{self.ckdir(ck)}/meta.json"
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["label"]
        return ck

    def jobdir(self, t):
        return f"{self.root}/jobs/{t['id'].replace('|', '__')}"

    def scan(self):
        found, missing = discover(self.cfg["checkpoints"], self.cfg["max_per_folder"])
        known = {c["source"] for c in self.state["checkpoints"]}
        new = [s for s in found if s not in known]
        for s in new:
            self.state["checkpoints"].append({"id": ck_id(s), "source": s, "added": time.time()})
            self.event(f"new checkpoint: {s}")
        if new:
            self.state["last_new"] = time.time()
        self.state["missing"] = missing
        plan(self.state, self.cfg)
        self.save()

    # -- choosing what runs next
    def ck_index(self, ck):
        return next(i for i, c in enumerate(self.state["checkpoints"]) if c["id"] == ck)

    def deps_state(self, t):
        sts = [self.state["tasks"][d]["status"] for d in t["deps"]]
        if any(s in ("failed", "skipped") for s in sts):
            return "broken"
        return "ok" if all(s == "done" for s in sts) else "wait"

    def prep_allowed(self, t):
        res = self.cfg["resources"]
        active = [c for c in self.state["checkpoints"]
                  if self.state["tasks"].get(f"prep|{c['id']}", {}).get("status") in ("done", "running")
                  and self.state["tasks"].get(f"done|{c['id']}", {}).get("status") not in ("done", "skipped")]
        if len(active) > res["prepare_ahead"]:
            return False
        free = shutil.disk_usage(self.root).free / 1e9
        src = t["params"]["source"]
        need = res["min_free_disk_gb"] + (os.path.getsize(src) / 1e9 / 2.5 if os.path.isfile(src) else 4)
        if free < need:
            if "disk" not in t["note"]:
                self.event(f"waiting for disk space before preparing {t['ck']} ({free:.0f} GB free, need {need:.0f})")
            t["note"] = f"waiting for disk space ({free:.0f} GB free, need {need:.0f})"
            return False
        return True

    def pick(self, lane):
        if self.state["lanes"].get(lane, {}).get("paused_until", 0) > time.time():
            return None
        cands = []
        for t in self.state["tasks"].values():
            if t["lane"] != lane or t["status"] != "pending" or t["not_before"] > time.time():
                continue
            ds = self.deps_state(t)
            if ds == "broken":
                t["status"], t["note"] = "skipped", "a job it needs failed"
                self.dirty = True
                continue
            if ds == "wait":
                continue
            cands.append(t)
        for t in sorted(cands, key=lambda t: (self.ck_index(t["ck"]), TYPE_ORDER[t["type"]], t["id"])):
            if t["type"] == "prep" and not self.prep_allowed(t):
                continue
            return t
        return None

    # -- starting jobs
    def build_job(self, t):
        import lab_jobs as lj
        ck, p, cfg = self.ckdir(t["ck"]), t["params"], self.cfg
        if t["type"] == "exp1":
            return lj.exp1_job(ck, p["agent"], p["prompt"], cfg)
        if t["type"] == "exp2e":
            return lj.exp2_explainer_job(ck, p["agent"], cfg)
        if t["type"] == "exp2r":
            exp = None if p["explainer"] == "-" else \
                self.jobdir(self.state["tasks"][f"exp2e|{t['ck']}|{p['explainer']}"]) + "/run"
            return lj.exp2_reader_job(ck, p["agent"], cfg, p["condition"], exp)
        if t["type"] == "exp4":
            return lj.exp4_job(ck, p["agent"], p["level"], cfg)
        if t["type"] == "judge":
            src = self.jobdir(self.state["tasks"][f"exp1|{t['ck']}|{p['of_agent']}|{p['prompt']}"]) + "/run"
            with open(f"{src}/final_answer.md") as f:
                answer = f.read()
            variant = "bare" if p["prompt"] == "bare" else ("blind" if cfg["blind"] else "full")
            return lj.judge_job(ck, p["agent"], answer, variant, cfg)
        raise ValueError(t["type"])

    def start(self, t):
        jd = self.jobdir(t)
        os.makedirs(jd, exist_ok=True)
        if self.already_done(t):
            t["status"] = "running"
            self.finish(t, 0)
            return
        if self.held_elsewhere(t):
            t.update(elsewhere=True, note="still running in another process; waiting for it")
            self.dirty = True
            return
        log = open(f"{jd}/console.log", "a")
        if t["type"] == "prep":
            cmd = [ASTROPT_PY, f"{HERE}/ckpt_prepare.py", t["params"]["source"], self.ckdir(t["ck"])] + \
                ([] if self.cfg["resources"]["keep_downloads"] else ["--drop-download"])
        elif t["type"] == "exp4s":
            src = self.jobdir(self.state["tasks"][f"exp4|{t['ck']}|{t['params']['agent']}|{t['params']['level']}"])
            cmd = [ASTROPT_PY, f"{HERE}/lab_jobs.py", "score-exp4", self.ckdir(t["ck"]), f"{src}/run/work/edits",
                   f"{jd}/score.json"]
        else:
            run = f"{jd}/run"
            if os.path.exists(f"{run}/state.json"):
                cmd = ["python3", f"{HERE}/harness.py", "--continue", run]
            else:
                if os.path.lexists(run):  # a run that died before it started: start it again
                    if os.path.islink(f"{run}/work"):
                        shutil.rmtree(os.path.realpath(f"{run}/work"), ignore_errors=True)
                    shutil.rmtree(run)
                job = self.build_job(t)
                atomic_json(f"{jd}/job.json", job)
                cmd = ["python3", f"{HERE}/harness.py", "--job", f"{jd}/job.json", "--run-dir", run]
        t.update(status="running", started=time.time(), attempts_started=t.get("attempts_started", 0) + 1)
        t["note"] = ""
        self.children[t["id"]] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                  cwd=HERE)
        self.event(f"start  {self.describe(t)}")
        self.save()

    def describe(self, t):
        p = t["params"]
        lab = self.ck_label(t["ck"])
        return {"prep": f"prepare {lab}",
                "exp1": f"exp1 {p.get('agent')} {p.get('prompt')} on {lab}",
                "judge": f"judge {p.get('agent')} grading {p.get('of_agent')} {p.get('prompt')} on {lab}",
                "exp2e": f"exp2 explainer {p.get('agent')} on {lab}",
                "exp2r": f"exp2 reader {p.get('agent')}, {p.get('condition')}"
                         + (f" from {p.get('explainer')}" if p.get("explainer") not in (None, "-") else "") + f", on {lab}",
                "exp4": f"exp4 {p.get('agent')} {p.get('level')} on {lab}",
                "exp4s": f"exp4 scoring {p.get('agent')} {p.get('level')} on {lab}",
                "done": f"tidy up {lab}"}[t["type"]]

    # -- finished jobs
    def finish(self, t, rc):
        jd = self.jobdir(t)
        run_state = {}
        if os.path.exists(f"{jd}/run/state.json"):
            with open(f"{jd}/run/state.json") as f:
                run_state = json.load(f)
        if rc == 0:
            try:
                t["result"] = self.result_of(t, run_state)
            except Exception as e:  # a result we cannot read counts as a failure of the job
                rc, t["note"] = 1, f"result unreadable: {type(e).__name__}: {e}"
        if rc == 0:
            t.update(status="done", finished=time.time())
            self.event(f"done   {self.describe(t)}" + (f" ({self.short(t)})" if self.short(t) else ""))
        elif t["type"] == "prep" and rc == 3:
            reason = open(f"{self.ckdir(t['ck'])}/unsupported.txt").read().strip() \
                if os.path.exists(f"{self.ckdir(t['ck'])}/unsupported.txt") else "unsupported"
            t.update(status="failed", note=f"unsupported checkpoint: {reason}")
            self.event(f"skip   {t['ck']}: {reason}")
        elif rc == 75:
            until = int(run_state.get("limit_until") or time.time() + 1800) + 60
            lane = self.state["lanes"].setdefault(t["lane"], {})
            lane.update(paused_until=max(lane.get("paused_until", 0), until), why="usage limit")
            t.update(status="pending", not_before=until, note=f"usage limit, resets {hm(until - 60)}")
            self.event(f"limit  {t['lane']} hit its usage limit; its jobs wait until {hm(until)} ({self.describe(t)})")
        elif rc == 65:
            t.update(status="failed", note="over the output cap even at the fallback effort; not retried")
            self.event(f"FAILED {self.describe(t)}: over the output cap even at the fallback effort; moving on")
        elif rc == 73:
            t.update(status="pending", elsewhere=True, note="still running in another process; waiting for it")
        elif self.stopping and rc in (130, -2, -15, 143):
            t.update(status="pending", note="stopped; resumes on restart")
        else:
            t["attempts"] += 1
            if t["attempts"] > len(BACKOFF):
                t.update(status="failed", note=f"failed {t['attempts']} times; see {os.path.relpath(jd, HERE)}/console.log")
                self.event(f"FAILED {self.describe(t)} (exit {rc}); see {os.path.relpath(jd, HERE)}/console.log")
            else:
                wait = BACKOFF[t["attempts"] - 1]
                t.update(status="pending", not_before=time.time() + wait,
                         note=f"failed (exit {rc}), retry {t['attempts']} at {hm(time.time() + wait)}")
                self.event(f"retry  {self.describe(t)} failed (exit {rc}); trying again at {hm(time.time() + wait)}")
        self.save()

    def short(self, t):
        r = t.get("result") or {}
        if t["type"] == "exp2r" and "fve_patch" in r:
            return (f"patch similarity {r.get('cos_patch', float('nan')):.3f} (FVE {r['fve_patch']:.3f}), "
                    f"dependence overlap {r.get('overlap_deps', float('nan')):.3f} (FVE {r['fve_deps']:.3f})")
        if t["type"] == "judge" and "score" in r:
            return f"score {r['score']:.2f}, confabulations {r['confabulations']}"
        if t["type"] == "exp4s" and "edit" in r:
            return f"physics rise {r['edit']['rise']:+.4f} (random edit {r['random_edit']['rise']:+.4f}), guard " \
                   f"{'ok' if r['edit']['guard_ok'] else 'FAILED'}"
        if t["type"] in ("exp1", "exp2e", "exp4") and "runs" in r:
            return f"{r['runs']} code runs, {r['turns']} turns" + ("" if r.get("tool_free", True) else ", NOT tool-free") \
                + (f", DROPPED TO {r['effort_fallback']['to']} EFFORT" if r.get("effort_fallback") else "")
        return ""

    def result_of(self, t, run_state):
        import lab_jobs as lj
        jd, ck = self.jobdir(t), self.ckdir(t["ck"])
        if t["type"] == "prep":
            with open(f"{ck}/meta.json") as f:
                meta = json.load(f)
            return {"label": meta["label"], "state": meta["training_state"], "physics_base": meta["physics_base"]}
        if t["type"] == "exp4s":
            with open(f"{jd}/score.json") as f:
                return json.load(f)
        res = {"runs": run_state.get("runs"), "turns": run_state.get("turn"), "tool_free": run_state.get("tool_free"),
               "evals": run_state.get("evals"), "effort": run_state.get("effort"),
               "effort_fallback": run_state.get("effort_fallback")}
        if t["type"] == "exp2r":
            res.update(lj.score_reader(ck, f"{jd}/run", self.cfg))
        if t["type"] == "judge":
            with open(f"{jd}/job.json") as f:
                items = json.load(f)["items"]
            res.update(lj.score_judge(f"{jd}/run", items))
        return res

    def tidy(self, t):
        """All of a checkpoint's jobs are finished: free the disk its converted weights take (all hard links)."""
        if self.cfg["resources"]["delete_weights_when_done"]:
            paths = [f"{self.ckdir(t['ck'])}/weights.bin"]
            for o in self.state["tasks"].values():
                if o["ck"] == t["ck"]:
                    paths.append(f"{self.jobdir(o)}/run/work/weights.bin")
            for p in paths:
                if os.path.lexists(p):
                    os.remove(os.path.realpath(p) if os.path.islink(p) else p)
        t.update(status="done", finished=time.time())
        self.event(f"done   {self.describe(t)}")

    # -- the status file
    def write_status(self, final=None):
        st, now = self.state, time.time()
        tasks = list(st["tasks"].values())
        count = {s: sum(1 for t in tasks if t["status"] == s) for s in ("done", "running", "pending", "failed", "skipped")}
        waiting = sum(1 for t in tasks if t["status"] == "pending" and t["not_before"] > now)
        head = final or ("STOPPING" if self.stopping else
                         ("WAITING (usage limits)" if all(st["lanes"].get(l, {}).get("paused_until", 0) > now
                                                          for l in self.cfg["agents"]) and not self.children else "RUNNING"))
        L = [f"# AstroPT lab: {st['name']} — {head}", "",
             f"updated {stamp(now)} · started {stamp(st['started'])} ({dur(now - st['started'])} ago) · "
             f"config {st['config_path']}", "",
             f"**Progress:** {count['done']} of {len(tasks)} jobs done, {count['running']} running, "
             f"{count['pending']} to do ({waiting} waiting), {count['failed']} failed, {count['skipped']} skipped.", "",
             "## Right now"]
        for lane in LANES:
            run = [self.state["tasks"][i] for i in self.children if self.state["tasks"][i]["lane"] == lane]
            pu = st["lanes"].get(lane, {}).get("paused_until", 0)
            if run:
                t = run[0]
                extra = ""
                rs = f"{self.jobdir(t)}/run/state.json"
                if os.path.exists(rs):
                    try:
                        with open(rs) as f:
                            r = json.load(f)
                        extra = (f" — turn {r['turn']}, part {r['phase'] + 1} of {len(r['phases'])}, {r['runs']} code runs"
                                 + (f", {r['evals']} evaluations" if r.get("evaluate") else ""))
                    except (json.JSONDecodeError, KeyError):
                        pass
                L.append(f"- **{lane}**: {self.describe(t)}{extra} — {dur(now - t['started'])}")
            elif pu > now:
                L.append(f"- **{lane}**: paused by its usage limit until {hm(pu)} ({dur(pu - now)}); "
                         f"`python3 lab.py wake` to retry now")
            else:
                L.append(f"- **{lane}**: idle")
        if st.get("missing"):
            L.append(f"- not found (yet): {', '.join(st['missing'])}")
        L += ["", "## Checkpoints", "", "| # | checkpoint | how trained | jobs done | state |", "|---|---|---|---|---|"]
        for i, c in enumerate(st["checkpoints"]):
            mine = [t for t in tasks if t["ck"] == c["id"]]
            prep = st["tasks"].get(f"prep|{c['id']}", {})
            trained = ((prep.get("result") or {}).get("state") or {}).get("state", "")
            state = ("failed: " + prep["note"]) if prep.get("status") == "failed" else \
                ("finished" if st["tasks"].get(f"done|{c['id']}", {}).get("status") == "done" else
                 ("in progress" if any(t["status"] in ("done", "running") for t in mine) else "waiting"))
            L.append(f"| {i + 1} | {self.ck_label(c['id'])} | {trained} | "
                     f"{sum(t['status'] == 'done' for t in mine)}/{len(mine)} | {state} |")
        L += ["", "## Results so far"] + self.results_md()
        bad = [t for t in tasks if t["status"] == "failed"] + \
              [t for t in tasks if t["status"] == "pending" and t["note"].startswith("failed")]
        if bad:
            L += ["", "## Needs attention"] + [f"- {self.describe(t)}: {t['note']}" for t in bad[:20]]
        low = [t for t in tasks if (t.get("result") or {}).get("effort_fallback")]
        if low:
            L += ["", "## Ran below max effort (hit the output cap)"] + [f"- {self.describe(t)}" for t in low]
        L += ["", "## Recent events", ""] + [f"    {e}" for e in st["events"][-15:]]
        with open(f"{self.root}/STATUS.md.tmp", "w") as f:
            f.write("\n".join(L) + "\n")
        os.replace(f"{self.root}/STATUS.md.tmp", f"{self.root}/STATUS.md")
        self.last_status, self.dirty = now, False

    def results_md(self):
        tasks = self.state["tasks"].values()
        L = []
        rows = [t for t in tasks if t["type"] == "judge" and t["status"] == "done"]
        if rows:
            L += ["", "**Experiment 1** (judge score 0-1, averaged over judges; confabulations)", "",
                  "| checkpoint | agent | prompt | score | heads | training state | confabulations |", "|---|---|---|---|---|---|---|"]
            groups = {}
            for t in rows:
                p = t["params"]
                groups.setdefault((t["ck"], p["of_agent"], p["prompt"]), []).append(t["result"])
            for (ck, a, pr), rs in groups.items():
                avg = lambda k: sum(r[k] for r in rs if r.get(k) is not None) / max(1, sum(r.get(k) is not None for r in rs))
                heads = f"{avg('heads'):.2f}" if any(r.get("heads") is not None for r in rs) else "-"
                L.append(f"| {self.ck_label(ck)} | {a} | {pr} | {avg('score'):.2f} | {heads} | "
                         f"{avg('training_state'):.2f} | {avg('confabulations'):.1f} |")
        rows = [t for t in tasks if t["type"] == "exp2r" and t["status"] == "done"]
        if rows:
            L += ["", "**Experiment 2** (how well a reader rebuilds each moment from the words: patch similarity, cosine, "
                  "0 = no information, 1 = exact; dependence overlap, 0 to 1; and the fraction of variance explained, as "
                  "natural language autoencoders score it, 0 = no better than the average moment)", "",
                  "| checkpoint | reader | explainer | condition | patch similarity | dependence overlap | patch FVE | "
                  "dependence FVE |", "|---|---|---|---|---|---|---|---|"]
            for t in sorted(rows, key=lambda t: (self.ck_index(t["ck"]), t["params"]["explainer"], t["params"]["agent"],
                                                 t["params"]["condition"])):
                p, r = t["params"], t["result"]
                L.append(f"| {self.ck_label(t['ck'])} | {p['agent']} | {p['explainer']} | {p['condition']} | "
                         f"{r.get('cos_patch', float('nan')):.3f} | {r.get('overlap_deps', float('nan')):.3f} | "
                         f"{r['fve_patch']:.3f} | {r['fve_deps']:.3f} |")
        rows = [t for t in tasks if t["type"] == "exp4s" and t["status"] == "done"]
        if rows:
            L += ["", "**Experiment 4** (rise of the physics score on the test galaxies; it counts only if the loss rose "
                  "under 2%; 'before' = the checkpoint as it is, 'random start' = the same network with random weights)",
                  "", "| checkpoint | agent | level | before | random start | rise | random edit | loss change | counts |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for t in sorted(rows, key=lambda t: (self.ck_index(t["ck"]), t["params"]["agent"], t["params"]["level"])):
                b, e, rnd = t["result"]["base"], t["result"]["edit"], t["result"]["random_edit"]
                ri = b.get("random_init_physics")
                L.append(f"| {self.ck_label(t['ck'])} | {t['params']['agent']} | {t['params']['level']} | "
                         f"{b['physics']:.3f} | {'-' if ri is None else f'{ri:.3f}'} | {e['rise']:+.4f} | {rnd['rise']:+.4f} | "
                         f"{e['loss_change']:+.2%} | {'yes' if e['guard_ok'] else 'no'} |")
        return L or ["", "(none yet)"]

    # -- the main loop
    def run(self):
        self.scan()
        self.event(f"lab running ({sum(t['status'] != 'done' for t in self.state['tasks'].values())} jobs to do)")
        last_scan, stop_sent, forwarded = time.time(), 0.0, False

        def on_signal(signum, frame):
            self.stopping = True
        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        while True:
            now = time.time()
            if os.path.exists(f"{self.root}/WAKE"):
                os.remove(f"{self.root}/WAKE")
                for lane in self.state["lanes"].values():
                    lane["paused_until"] = 0
                for t in self.state["tasks"].values():
                    if t["status"] == "pending":
                        t["not_before"] = 0
                self.event("woken up: retrying everything that was waiting")
            for tid, p in list(self.children.items()):
                rc = p.poll()
                if rc is not None:
                    del self.children[tid]
                    self.finish(self.state["tasks"][tid], rc)
            if self.stopping:
                if not self.children:
                    break
                if not stop_sent:  # a Ctrl-C or `lab.py stop` reaches the jobs too; pass it on only if they missed it
                    stop_sent, forwarded = now, False
                    self.event("stopping: waiting for the running jobs to save their place")
                elif now - stop_sent > 15 and not forwarded:
                    forwarded = True
                    for p in self.children.values():
                        p.send_signal(signal.SIGINT)
                elif now - stop_sent > 150:
                    for p in self.children.values():
                        p.kill()
            else:
                if self.cfg["watch_minutes"] and now - last_scan > 600:
                    last_scan = now
                    self.scan()
                for lane in LANES:
                    if not self.lane_busy(lane):
                        t = self.pick(lane)
                        if t:
                            self.start(t)
                for t in list(self.state["tasks"].values()):  # tidy-up jobs run right here, once nothing else needs
                    if t["type"] == "done" and t["status"] == "pending" and all(  # this checkpoint's weights
                            self.state["tasks"][d]["status"] in ("done", "failed", "skipped") for d in t["deps"]):
                        self.tidy(t)
                        self.save()
                if not self.children and all(t["status"] in ("done", "failed", "skipped")
                                             for t in self.state["tasks"].values()):
                    idle = now - self.state.get("last_new", now)
                    if not self.cfg["watch_minutes"] or idle > self.cfg["watch_minutes"] * 60:
                        return self.wind_down()
            if self.dirty or now - self.last_status > 30:
                self.save()
                self.write_status()
            time.sleep(2)
        self.save()
        self.write_status(final="STOPPED (start it again to resume)")
        self.event("stopped")
        return 130

    def wind_down(self):
        import lab_jobs as lj
        tasks = self.state["tasks"].values()
        for t in tasks:  # rescore every reader with the current scoring code, so all results are comparable
            if t["type"] == "exp2r" and t["status"] == "done":
                try:
                    t["result"].update(lj.score_reader(self.ckdir(t["ck"]), f"{self.jobdir(t)}/run", self.cfg))
                except Exception as e:
                    t["note"] = f"rescoring failed: {type(e).__name__}: {e}"
        results = [{"job": t["id"], "type": t["type"], "checkpoint": self.ck_label(t["ck"]), "params": t["params"],
                    "status": t["status"], "result": t["result"], "note": t["note"]} for t in tasks]
        atomic_json(f"{self.root}/results.json", results)
        self.state["finished"] = time.time()
        self.event(f"FINISHED: {sum(t['status'] == 'done' for t in tasks)} jobs done, "
                   f"{sum(t['status'] == 'failed' for t in tasks)} failed; results in results.json")
        self.save()
        self.write_status(final="FINISHED")
        remove_cron(self.cfg["name"])
        return 0


# ------------------------------------------------------------------ start / stop / supervise
def pid_alive(path):
    try:
        with open(path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def remove_cron(name):
    try:
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return
    keep = [l for l in cur.splitlines() if CRON_MARK + name not in l]
    if len(keep) != len(cur.splitlines()):
        subprocess.run(["crontab", "-"], input="\n".join(keep) + ("\n" if keep else ""), text=True)


def add_cron(name, cfg_path):
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if CRON_MARK + name in cur:
        return
    line = f"@reboot sleep 60 && cd {HERE} && /usr/bin/python3 lab.py start {cfg_path} {CRON_MARK}{name}"
    subprocess.run(["crontab", "-"], input=(cur.rstrip("\n") + "\n" if cur.strip() else "") + line + "\n", text=True)


def cmd_start(cfg_path, survive_reboot):
    cfg = load_config(cfg_path)
    root = root_of(cfg)
    os.makedirs(root, exist_ok=True)
    if pid_alive(f"{root}/supervisor.pid"):
        print(f"already running; status: {root}/STATUS.md")
        return
    if os.path.exists(f"{root}/STOP"):
        os.remove(f"{root}/STOP")
    with open(f"{root}/supervisor.log", "a") as log:
        p = subprocess.Popen([ASTROPT_PY, os.path.abspath(__file__), "supervise", cfg["config_path"]], cwd=HERE,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(f"{root}/supervisor.pid", "w") as f:
        f.write(str(p.pid))
    if survive_reboot:
        add_cron(cfg["name"], cfg["config_path"])
    print(f"started (pid {p.pid}). Watch: {root}/STATUS.md   (python3 lab.py status | wake | stop)")


def cmd_supervise(cfg_path):
    cfg = load_config(cfg_path)
    root = root_of(cfg)
    flag = {"stop": False}

    def on_signal(signum, frame):  # a handler (not SIG_IGN) so the child still gets Ctrl-C normally
        flag["stop"] = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    backoff = 30
    while True:
        t0 = time.time()
        rc = subprocess.run([ASTROPT_PY, os.path.abspath(__file__), "run", cfg_path], cwd=HERE).returncode
        stamp_now = stamp()
        if rc == 0:
            print(f"{stamp_now} supervisor: the lab finished", flush=True)
            break
        if flag["stop"] or os.path.exists(f"{root}/STOP") or rc in (2, 4):
            print(f"{stamp_now} supervisor: stopped (exit {rc})", flush=True)
            break
        backoff = 30 if time.time() - t0 > 3600 else min(backoff * 2, 900)
        print(f"{stamp_now} supervisor: the lab exited with {rc}; restarting in {backoff} s", flush=True)
        for _ in range(backoff):
            if flag["stop"] or os.path.exists(f"{root}/STOP"):
                break
            time.sleep(1)
        if flag["stop"] or os.path.exists(f"{root}/STOP"):
            break
    for f in ("supervisor.pid", "STOP"):
        if os.path.exists(f"{root}/{f}"):
            os.remove(f"{root}/{f}")


def cmd_run(cfg_path):
    if os.path.realpath(sys.executable) != os.path.realpath(ASTROPT_PY):  # job building, scoring and Hugging Face
        os.execv(ASTROPT_PY, [ASTROPT_PY, os.path.abspath(__file__), "run", cfg_path])  # listing need its packages
    cfg = load_config(cfg_path)
    root = root_of(cfg)
    os.makedirs(root, exist_ok=True)
    lock = open(f"{root}/.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{root} is already running in another process", flush=True)
        return 4
    return Lab(cfg).run()


def cmd_stop(arg):
    root = find_root(arg)
    open(f"{root}/STOP", "w").close()
    pid = pid_alive(f"{root}/supervisor.pid")
    if not pid:
        print("not running")
        return
    os.killpg(pid, signal.SIGINT)  # the lab and every job get it: they save their place and exit
    print("stopping (every job saves its place) ...", flush=True)
    for _ in range(180):
        if not pid_alive(f"{root}/supervisor.pid"):
            print("stopped")
            return
        time.sleep(1)
    print(f"still stopping; check {root}/STATUS.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["start", "run", "supervise", "status", "wake", "stop"])
    ap.add_argument("config", nargs="?")
    ap.add_argument("--survive-reboot", action="store_true")
    a = ap.parse_args()
    if a.command in ("start", "run", "supervise") and not a.config:
        ap.error(f"{a.command} needs the config file")
    if a.command == "start":
        cmd_start(a.config, a.survive_reboot)
    elif a.command == "run":
        sys.exit(cmd_run(a.config))
    elif a.command == "supervise":
        cmd_supervise(a.config)
    elif a.command == "status":
        root = find_root(a.config)
        print(open(f"{root}/STATUS.md").read() if os.path.exists(f"{root}/STATUS.md") else "no status yet")
    elif a.command == "wake":
        root = find_root(a.config)
        open(f"{root}/WAKE", "w").close()
        print("woken: waiting jobs retry within a few seconds")
    elif a.command == "stop":
        cmd_stop(a.config)


if __name__ == "__main__":
    main()
