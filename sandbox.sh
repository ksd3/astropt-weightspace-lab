#!/usr/bin/env bash
# Run one agent CLI (claude or codex) in a throwaway Docker container that can see only:
#   /work     a fresh copy of weights.bin, if there is one; what the agent writes there is kept in runs/<name>-<time>/
#   python    a clean Python 3.13 on PATH (numpy, scipy, matplotlib, pillow, scikit-learn, pandas, PyTorch + GPU)
#   its login claude: your ~/.claude with every subfolder (projects, sessions, memory, skills, plugins, ...) and
#             history.jsonl swapped for empty ones; codex: an empty Codex home holding only your auth.json
#             (token refreshes still reach your real login files)
#   /usr and /etc, read-only
# Nothing else from your home folder: no astroPT repo, no Hugging Face cache, no prompts folder, no past sessions.
# Own process list (it cannot see the other runs), own /tmp, internet only for the agent's own API.
# The run folder really lives in ~/.sandbox-runs/run-XXXXXX, because a container can read the host path of its
# mounts (/proc/self/mountinfo) and this folder's path names the experiment; runs/<name>-<time> links to it.
# The prompt comes in on stdin. The agent's raw event stream is saved as runs/<name>-<time>.jsonl and printed
# readably as it arrives (pretty.py); stderr is saved as runs/<name>-<time>.stderr.
#
# usage: ./sandbox.sh <run-name> claude|codex <agent arguments...> < prompt.txt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HOME/.sandbox-python"
CLAUDE_BIN="$(readlink -f "$(command -v claude || true)")"
CODEX_DIR="${CODEX_DIR:-}"
if [ -z "$CODEX_DIR" ] && command -v codex >/dev/null; then
  CODEX_DIR="$(dirname "$(dirname "$(readlink -f "$(command -v codex)")")")/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl"
fi
IMAGE=sandbox-host:1

if [ $# -lt 2 ]; then sed -n '2,16p' "$0"; exit 2; fi
name="$1"; shift
agent="$1"
mkdir -p "$HERE/runs" "$HOME/.sandbox-runs"
work="$(mktemp -d "$HOME/.sandbox-runs/run-XXXXXX")"
run="$HERE/runs/$name-$(date +%Y%m%d-%H%M%S)"
ln -s "$work" "$run"
if [ -e "$HERE/weights.bin" ]; then cp "$HERE/weights.bin" "$work/"; fi

uid="$(id -u)"; gid="$(id -g)"
a=(--rm -i --init --user "$uid:$gid" --hostname sandbox --shm-size 8g
   -v /usr:/usr:ro -v /etc:/etc:ro
   --tmpfs "/tmp:rw,exec,nosuid,nodev,mode=1777,size=16g" --tmpfs "/var/tmp:rw,nosuid,nodev,mode=1777"
   --tmpfs "$HOME:rw,exec,nosuid,nodev,uid=$uid,gid=$gid,mode=0700"
   -v "$work:/work" -w /work
   -v "$PY:$PY:ro"
   -e HOME="$HOME" -e USER="$USER" -e LOGNAME="$USER" -e SHELL=/bin/bash -e LANG=C.UTF-8)
for d in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
  if [ -e "$d" ]; then a+=(--device "$d"); fi
done
path="$PY/env/bin:/usr/local/bin:/usr/bin:/bin"

case "$agent" in
  claude)
    a+=(-v "$CLAUDE_BIN:/opt/claude/bin/claude:ro" -e DISABLE_AUTOUPDATER=1 -v "$HOME/.claude:$HOME/.claude")
    for d in "$HOME"/.claude/*/ "$HOME"/.claude/.[!.]*/; do
      if [ -d "$d" ]; then a+=(--tmpfs "${d%/}:rw,exec,nosuid,nodev,uid=$uid,gid=$gid,mode=0700"); fi
    done
    if [ -e "$HOME/.claude/history.jsonl" ]; then
      : > "$work.history.jsonl"
      a+=(-v "$work.history.jsonl:$HOME/.claude/history.jsonl")
    fi
    path="/opt/claude/bin:$path" ;;
  codex)
    a+=(-v "$CODEX_DIR:/opt/codex:ro"
        --tmpfs "$HOME/.codex:rw,exec,nosuid,nodev,uid=$uid,gid=$gid,mode=0700"
        -v "$HOME/.codex/auth.json:$HOME/.codex/auth.json" -e CODEX_HOME="$HOME/.codex")
    path="/opt/codex/bin:/opt/codex/codex-path:$path" ;;
  *) echo "the agent must be claude or codex" >&2; exit 2 ;;
esac
a+=(-e PATH="$path")

echo "run folder: $run" >&2
docker run "${a[@]}" "$IMAGE" "$@" 2> >(tee "$run.stderr" >&2) | tee "$run.jsonl" | python3 -u "$HERE/pretty.py"
