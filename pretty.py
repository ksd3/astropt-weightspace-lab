"""Print a Claude Code (stream-json) or Codex (exec --json) event stream readably, as it arrives.

  ./sandbox.sh ... | python3 -u pretty.py        (sandbox.sh does this for you)
  python3 pretty.py < runs/<name>-<time>.jsonl     (replay a saved run)
"""

import json
import sys

CUT = 700  # characters of tool output to show (the .jsonl log keeps everything)


def short(s, n=CUT):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[:n] + f" ... [{len(s) - n} more chars]"


def say(tag, text):
    text = str(text).rstrip()
    if text:
        print(f"{tag} " + text.replace("\n", "\n" + " " * (len(tag) + 1)), flush=True)


def tool_text(content):
    if isinstance(content, list):
        return "\n".join(c.get("text", f"[{c.get('type')}]") for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else json.dumps(content)


def claude(ev):
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        say("[start]", f"model {ev.get('model')}, cwd {ev.get('cwd')}, permission mode {ev.get('permissionMode')}\n"
                       f"tools: {', '.join(ev.get('tools', []))}\nMCP servers: {ev.get('mcp_servers')}")
    elif t == "assistant":
        for c in ev.get("message", {}).get("content", []):
            if c.get("type") == "text":
                say("[claude]", c["text"])
            elif c.get("type") == "thinking":
                say("[thinking]", c.get("thinking", ""))
            elif c.get("type") == "tool_use":
                inp = c.get("input", {})
                arg = inp.get("command") or inp.get("file_path") or inp.get("pattern") or inp.get("prompt") or inp
                say(f"[{c.get('name')}]", short(arg, 3000))
    elif t == "user":
        content = ev.get("message", {}).get("content", [])
        for c in content if isinstance(content, list) else []:
            if c.get("type") == "tool_result":
                say("  -> error:" if c.get("is_error") else "  ->", short(tool_text(c.get("content", ""))))
    elif t == "result":
        say("[done]", f"{ev.get('subtype')}: {ev.get('num_turns')} turns, {ev.get('duration_ms', 0) / 60000:.1f} min, "
                      f"cost ${ev.get('total_cost_usd', 0):.2f}\n{ev.get('result', '')}")
    elif t == "system":
        say("[system]", f"{ev.get('subtype')} {short({k: v for k, v in ev.items() if k not in ('type', 'subtype', 'session_id', 'uuid')}, 300)}")
    elif t != "stream_event":
        say(f"[{t}]", short(ev, 500))


def codex(ev):
    t = ev.get("type", "")
    item = ev.get("item") or {}
    it = item.get("type")
    if t == "thread.started":
        say("[start]", f"thread {ev.get('thread_id')}")
    elif t == "item.started" and it == "command_execution":
        say("[shell]", short(item.get("command", ""), 3000))
    elif t == "item.completed":
        if it == "agent_message":
            say("[codex]", item.get("text", ""))
        elif it == "reasoning":
            say("[thinking]", item.get("text", ""))
        elif it == "command_execution":
            say(f"  -> exit {item.get('exit_code')}:", short(item.get("aggregated_output", "")))
        elif it == "file_change":
            say("[files]", ", ".join(f"{c.get('kind')} {c.get('path')}" for c in item.get("changes", [])))
        else:
            say(f"[{it}]", short({k: v for k, v in item.items() if k not in ("id", "type")}, 1500))
    elif t == "turn.completed":
        say("[done]", f"usage {ev.get('usage')}")
    elif t in ("turn.failed", "error"):
        say("[error]", short(ev, 2000))
    elif t not in ("turn.started", "item.started", "item.updated"):
        say(f"[{t}]", short(ev, 300))


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        print(line, flush=True)
        continue
    try:
        (codex if "." in str(ev.get("type", "")) or "item" in ev else claude)(ev)
    except Exception as e:  # never let the viewer kill the run
        print(f"[viewer could not show an event: {e}] {short(line, 300)}", flush=True)
