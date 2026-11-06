# A sandbox for letting AI agents run code on model weights

This repo runs coding agents with **all of their own tools switched off**. An agent can only reply with text. When it replies with a Python block, the block runs in a throwaway Docker container with no network, and the printed output goes back as the next message.

It was built to test whether agents can understand and edit a neural network from its raw weights. The sandbox itself works for any task.

## Try the sandbox

You need Linux, Docker and [uv](https://docs.astral.sh/uv/). No AI account or GPU is needed.

```bash
git clone https://github.com/ksd3/astropt-weightspace-lab
cd astropt-weightspace-lab
./setup_sandbox.sh --minimal
python3 sandbox_example.py
```

`sandbox_example.py` runs code through the real sandbox and tests the walls:

```
PASS  no_network
PASS  only_loopback
PASS  usr_read_only
PASS  data_readable
PASS  data_read_only
PASS  host_secret_hidden
PASS  host_home_only_has_python
PASS  not_root
PASS  tmp_is_fresh
PASS  kept_folder_persists
PASS  runaway_code_is_killed
```

## How it works

```
 agent CLI, tools off            code runner
 (fresh container each turn)     (fresh container each run)
        |                               ^
        | reply with a ```python block  |
        +---------> harness.py ---------+
        ^                |
        +-- "OUTPUT of run k of at most N" <-- printed output
```

| | Agent container | Code container |
|---|---|---|
| Network | Yes, to reach its API | **None** |
| Tools | All off | Runs one Python block |
| Your files | Only its login file | Only the files you mount, read-only |
| `/usr`, `/etc` | Read-only from the host | Read-only from the host |
| Limits | | Time, RAM, process count; output cut at 20,000 characters |
| Afterwards | Removed; the conversation stays in the run folder | Removed; only folders you choose to keep remain |

The code is in `harness.py`: `common()` and `claude_cmd()` / `codex_cmd()` build the agent container, and `run_code()` builds the code container. `sandbox.sh` is a simpler version for one agent call.

## Limits of this sandbox

This is meant to keep experiments clean, not to contain a determined attacker.

- Containers share your kernel, and `/etc` from your machine is readable, including `/etc/passwd`.
- The clean Python is mounted at its usual path, so the code container can see your home folder's path, but nothing else in it.
- GPU devices are passed through when present.
- The agent container holds your CLI login credentials and has network access, which is why its tools are off.
- Only Linux x64 has been tested.

## Run a real agent

Install and log in to the [Claude Code](https://claude.com/claude-code) or [Codex](https://github.com/openai/codex) CLI. Then write a job file:

```bash
cat > job.json <<'JSON'
{"agent": "claude", "effort": "low", "weights": null, "files": {}, "persist": [],
 "prelude": "", "timeout": 60, "mem": "2g",
 "phases": [{"name": "demo", "kind": "chat", "max_runs": 2,
             "message": "Reply with one python block that prints 6*7. After you see the output, state the answer."}]}
JSON
python3 harness.py --job job.json --run-dir runs/demo
```

Use `"agent": "codex"` for Codex. The model names are in `MODELS` at the top of `harness.py`. The transcript is in `runs/demo/turns/`, and the code runs are in `runs/demo/tool_calls.jsonl`.

## The weight-space experiments

`lab.py` runs four experiments on a list of [astroPT](https://github.com/smith42/astroPT) checkpoints, unattended. It handles crashes and usage limits and writes everything to one status file. See [LAB.md](LAB.md).

```bash
./setup_sandbox.sh                       # full packages, including PyTorch with CUDA: about 7 GB

git clone https://github.com/smith42/astroPT
git -C astroPT checkout fa361ae
uv venv --python 3.13.5 astroPT/.venv
uv pip install --python astroPT/.venv/bin/python -r requirements-astropt.txt
uv pip install --python astroPT/.venv/bin/python --no-deps -e astroPT
export ASTROPT_PY="$PWD/astroPT/.venv/bin/python"

"$ASTROPT_PY" galaxy_data.py             # downloads the test galaxies once
cp lab_config.example.toml my_config.toml   # then list your checkpoints in it
python3 lab.py start my_config.toml
```

The experiments use a GPU and a lot of AI usage, and they don't include weights or results. Package versions are pinned, but agent CLI versions and model availability will change over time, so exact outputs won't repeat.

## Remove everything

```bash
docker rmi sandbox-host:1
rm -rf ~/.sandbox-python ~/.sandbox-runs
```
