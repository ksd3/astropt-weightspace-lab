"""The jobs of every experiment for one prepared checkpoint, and the scoring of their results (used by lab.py).

Experiments
  exp1   read the weights, no running the network (blog experiment 3 style); prompts bare / full / blind
  exp2   explain how the network builds a galaxy (running allowed), then describe 40 moments in words; readers
         rebuild each moment's predicted patch and dependence map from the words alone; scored by the fraction of
         variance explained (FVE), as natural language autoencoders score their explanations
  exp4   edit the weights to raise the physics score, levels L0 (one run), L1 (runs, no evaluations), L2 (runs and
         evaluations); scored on the test galaxies against a random edit of the same size
  judge  grades exp1 answers against the facts measured on the checkpoint

  python lab_jobs.py score-exp4 CKPT_DIR EDITS_DIR OUT.json      (GPU; lab.py runs it as a separate process)
"""

import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galaxy_data as gd  # noqa: E402
import harness  # noqa: E402  (stdlib only: its constants and the reader's submit() prelude)
import prompts_lib as pl  # noqa: E402
from spiral import ORDER, POSITION, RING, patch_to_vector  # noqa: E402

ASTROPT_PY = os.environ.get("ASTROPT_PY", sys.executable)  # a Python with torch + astropt installed
DATA = f"{HERE}/data/galaxies"

EDIT_PRELUDE = '''
import os as _os
for _a in W.values():
    _a.setflags(write=False)
_SHAPES = dict(SHAPES)
def save_edit(name, array):
    """Store a new value for one tensor (same shape as W[name]); kept between runs."""
    if name not in _SHAPES:
        raise KeyError(f"unknown tensor {name}; the names are those in SHAPES")
    a = np.asarray(array, dtype=np.float32)
    if a.shape != tuple(_SHAPES[name]):
        raise ValueError(f"{name} must have shape {tuple(_SHAPES[name])}, not {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name}: the new values must be finite")
    w, e = W[name].astype(np.float64).ravel(), a.astype(np.float64).ravel()
    ww = w @ w
    if ww > 0:
        c = (w @ e) / ww
        if abs(c - 1) > 1e-6 and np.linalg.norm(e - c * w) <= 1e-6 * max(np.linalg.norm(e), 1e-30):
            raise ValueError(f"{name}: multiplying a whole tensor by one number is not allowed")
    np.save(f"edits/{name}.npy", a)
def load_edits():
    """{name: array} of your current edits."""
    return {f[:-4]: np.load(f"edits/{f}") for f in sorted(_os.listdir("edits")) if f.endswith(".npy")}
def drop_edit(name):
    p = f"edits/{name}.npy"
    if _os.path.exists(p):
        _os.remove(p)
'''


def meta_of(ck):
    with open(f"{ck}/meta.json") as f:
        return json.load(f)


def base_job(ck, agent, cfg, name, effort=None):
    return {"agent": agent, "effort": effort or cfg["effort"], "name": name, "weights": f"{ck}/weights.bin",
            "timeout": 600, "mem": cfg["resources"]["mem"], "files": {}, "persist": []}


# ------------------------------------------------------------------ experiment 1
def exp1_job(ck, agent, prompt, cfg):
    meta = meta_of(ck)
    variant = "bare" if prompt == "bare" else ("blind" if cfg["blind"] else "full")
    text = pl.exp1(meta, variant, runs=cfg["exp1"]["runs"])
    job = base_job(ck, agent, cfg, f"exp1-{variant}")
    job["prelude"] = "import numpy as np\n" + ("" if variant == "bare" else pl.code(meta) + "\nW = load_weights()\n")
    job["phases"] = [{"name": f"exp1-{variant}", "message": text, "max_runs": cfg["exp1"]["runs"], "kind": "chat"}]
    job["variant"] = variant
    return job


# ------------------------------------------------------------------ experiment 2
def moments(cfg):
    return gd.moments(with_seq=False)[:cfg["exp2"]["moments"]]


def batches(cfg):
    ms, b = moments(cfg), cfg["exp2"]["batch"]
    return [ms[i:i + b] for i in range(0, len(ms), b)]


def exp2_explainer_job(ck, agent, cfg):
    meta, e = meta_of(ck), cfg["exp2"]
    bs = batches(cfg)
    job = base_job(ck, agent, cfg, "exp2-explain")
    job["files"] = {"galaxies.npy": f"{DATA}/study_images.npy", "moments.npz": f"{DATA}/moments.npz"}
    job["prelude"] = ("import numpy as np\n" + pl.code(meta) + "\nW = load_weights()\n" + pl.GENERATE +
                      "\nGALAXIES = np.load('galaxies.npy')\n")
    job["phases"] = [{"name": "study", "kind": "chat", "max_runs": e["study_runs"], "validate": {"type": "account"},
                      "message": pl.exp2_study(meta, cfg["blind"], e["study_runs"], sum(map(len, bs)), e["batch"])}]
    for i, chunk in enumerate(bs):
        job["phases"].append({"name": f"moments{i + 1}", "kind": "chat", "max_runs": e["batch_runs"],
                              "message": pl.exp2_batch(chunk, i + 1, len(bs), e["batch_runs"]),
                              "validate": {"type": "moments", "ids": [m["id"] for m in chunk], "max_words": 80}})
    return job


def explainer_output(run_dir):
    """(account text, {moment id: description}) from a finished explainer run, cleaned to the rules: no digits,
    at most 80 words (descriptions) / 400 words (account)."""
    def clean(t, n):
        return " ".join(re.sub(r"\d", "", t).split()[:n])
    texts, account = {}, ""
    for f in sorted(os.listdir(run_dir)):
        if not (f.startswith("phase_") and f.endswith(".json")):
            continue
        with open(f"{run_dir}/{f}") as fh:
            ph = json.load(fh)
        if ph["name"] == "study":
            with open(f"{run_dir}/{f[:-5]}.md") as fh:
                reply = fh.read()
            m = re.search(r"ACCOUNT:\s*(.*?)(?:\n\s*CLAIMS:|\Z)", reply, re.S)
            account = clean(m.group(1) if m else reply, 400)
        elif ph["name"].startswith("moments") and isinstance(ph.get("parsed"), dict):
            texts.update({k: clean(v, 80) for k, v in ph["parsed"].items()})
    return account, texts


def shuffle_texts(texts, ids):
    """Each moment gets the description of another moment (a fixed derangement: the one 7 places on)."""
    k = 7 if len(ids) > 7 else max(1, len(ids) // 2)
    return {ids[i]: texts.get(ids[(i + k) % len(ids)], "") for i in range(len(ids))}


def exp2_reader_job(ck, reader, cfg, condition, explainer_run=None):
    job = base_job(ck, reader, cfg, f"exp2-read-{condition}", effort=cfg["exp2"]["reader_effort"])
    job["weights"] = None  # readers get no network and no data, only words
    job["prelude"] = "import numpy as np\nORDER = " + repr([tuple(o) for o in ORDER]) + "\n" + harness.SUBMIT_PRELUDE
    account, texts = explainer_output(explainer_run) if explainer_run else ("", {})
    ids = [m["id"] for m in moments(cfg)]
    if condition == "shuffled":
        texts = shuffle_texts(texts, ids)
    job["phases"] = [{"name": f"read{i + 1}", "kind": "single_code", "max_runs": 3,
                      "message": pl.reader(condition, chunk, texts, account)} for i, chunk in enumerate(batches(cfg))]
    job["timeout"] = 60
    return job


def fve(R, T):
    """Fraction of variance explained: 1 - sum |R - T|^2 / sum |mean(T) - T|^2 (0 = no better than the average)."""
    R, T = np.asarray(R, dtype=np.float64), np.asarray(T, dtype=np.float64)
    den = ((T - T.mean(0, keepdims=True)) ** 2).sum()
    return float(1 - ((R - T) ** 2).sum() / den) if den > 0 else 0.0


def unit(v):
    v = np.asarray(v, dtype=np.float64) - np.mean(v)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def reader_submissions(run_dir):
    subs = {}
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("phase_") and f.endswith(".json"):
            with open(f"{run_dir}/{f}") as fh:
                parsed = json.load(fh).get("parsed")
            if isinstance(parsed, dict):
                subs.update(parsed)
    return subs


def score_reader(ck, run_dir, cfg):
    """FVE of one reader run against the checkpoint's measured moments: the predicted patch and the dependence map."""
    truth = np.load(f"{ck}/truth/moments.npz")
    tid = [str(i) for i in truth["ids"]]
    ms = moments(cfg)
    subs = reader_submissions(run_dir)
    R_p, T_p, R_d, T_d, rings, missing = [], [], [], [], [], 0
    for m in ms:
        j = tid.index(m["id"])
        s = subs.get(m["id"]) or {}
        if not s:
            missing += 1
        patch = s.get("patch")
        R_p.append(unit(patch_to_vector(patch)) if patch is not None else np.zeros(768))
        T_p.append(unit(truth["pred"][j]))
        dmap = np.zeros((16, 16))
        for r, c, w in s.get("deps") or []:
            if 0 <= r < 16 and 0 <= c < 16 and POSITION[(r, c)] < m["n"] and w > 0:
                dmap[r, c] += w
        if dmap.sum() <= 0:  # nothing usable: an even spread over the patches seen so far
            for k in range(m["n"]):
                dmap[ORDER[k]] = 1.0
        R_d.append((dmap / dmap.sum()).ravel())
        T_d.append(truth["dep"][j].ravel())
        rings.append(m["ring"])
    R_p, T_p, R_d, T_d, rings = map(np.array, (R_p, T_p, R_d, T_d, rings))
    # FVE as natural language autoencoders use it, plus two robust scores that stay meaningful when the network's
    # outputs hardly vary between moments (an untrained network): the cosine between the drawn and the predicted patch
    # (0 = no information, 1 = exact) and the overlap of the dependence maps (sum of the smaller weight, 0 to 1)
    def scores(sel):
        return {"cos_patch": float(np.mean((R_p[sel] * T_p[sel]).sum(1))), "fve_patch": fve(R_p[sel], T_p[sel]),
                "overlap_deps": float(np.mean(np.minimum(R_d[sel], T_d[sel]).sum(1))), "fve_deps": fve(R_d[sel], T_d[sel])}
    out = {"moments": len(ms), "missing": missing, **scores(np.ones(len(ms), bool)), "by_ring": {}}
    for r in sorted(set(rings.tolist())):
        if (rings == r).sum() >= 2:
            out["by_ring"][int(r)] = scores(rings == r)
    return out


# ------------------------------------------------------------------ experiment 4
def exp4_job(ck, agent, level, cfg):
    meta, e = meta_of(ck), cfg["exp4"]
    with open(f"{ck}/truth/physics_base.json") as f:
        base = json.load(f)
    runs = 1 if level == "L0" else e["runs"]
    job = base_job(ck, agent, cfg, f"exp4-{level}")
    job["persist"] = ["edits"]
    job["prelude"] = "import numpy as np\n" + pl.code(meta) + "\nW = load_weights()\n" + EDIT_PRELUDE
    job["phases"] = [{"name": f"exp4-{level}", "kind": "chat", "max_runs": runs,
                      "message": pl.exp4(meta, level, base["dev"], blind=cfg["blind"], runs=runs,
                                         evals=e["evaluations"])}]
    if level == "L2":
        job["evaluate"] = {"budget": e["evaluations"], "timeout": 3600,
                           "cmd": [ASTROPT_PY, f"{HERE}/physics.py", f"{ck}/weights.bin", "--split", "dev", "--edits",
                                   "{edits}", "--brief", "--base", f"{ck}/truth/physics_base.json"]}
    return job


def score_exp4(ck, edits, out):
    """Test-set physics of the agent's edits and of a random edit of the same size, against the unedited weights."""
    import physics
    with open(f"{ck}/truth/physics_base.json") as f:
        base_all = json.load(f)
    base = base_all["test"]
    edited = physics.evaluate(f"{ck}/weights.bin", "test", edits)
    rand = physics.evaluate(f"{ck}/weights.bin", "test", edits, random_like=True, seed=0) if edited["edited_tensors"] \
        else edited
    def summarise(r):
        guard = r["loss"] <= 1.02 * base["loss"]
        rise = r["score"] - base["score"]
        return {"physics": r["score"], "rise": rise, "loss": r["loss"], "loss_change": r["loss"] / base["loss"] - 1,
                "guard_ok": guard, "score": rise if guard else 0.0, "r2": r["r2"]}
    res = {"base": {"physics": base["score"], "loss": base["loss"], "r2": base["r2"],
                    "random_init_physics": (base_all.get("random_init") or {}).get("test", {}).get("score")},
           "edit": summarise(edited),
           "random_edit": summarise(rand), "edited_tensors": edited["edited_tensors"], "change_norm": edited["change_norm"],
           "problems": edited["problems"]}
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    return res


# ------------------------------------------------------------------ the judge (experiment 1)
def head_line(k, h):
    return (f"{k}: current patch {h['current']:.0%}, previous patch {h['previous']:.0%}, grid neighbours of the patch "
            f"being predicted {h['neighbours']:.0%}, the 4 centre patches {h['centre']:.0%}, same ring {h['same_ring']:.0%}, "
            f"the ring just inside {h['ring_inside']:.0%}; spread {h['spread']:.2f} (0 = all on one patch, 1 = even over "
            f"all earlier patches); mean distance from the patch being predicted {h['distance']:.1f} patches")


def judge_facts_items(ck, variant):
    meta = meta_of(ck)
    a, ts = meta["arch"], meta["training_state"]
    with open(f"{ck}/truth/head_facts.json") as f:
        heads = json.load(f)
    loss = meta["measured"]
    state = {"untrained": "untrained: its predictions are no better than predicting zeros (random weights)",
             "partly trained": "partly trained: better than predicting zeros, but far from a fully trained model",
             "well trained": "well trained: it predicts the next patch far better than predicting zeros"}[ts["state"]]
    w, w4 = a["width"], 4 * a["width"]
    codec = (f"an MLP encoder (768 -> {w4} -> {w}) and decoder ({w} -> {w4} -> 768)" if a.get("tokeniser", "aim") == "aim"
             else f"a single linear encoder (768 -> {w}) and a single linear decoder ({w} -> 768)")
    facts = [f"Architecture: a pre-LayerNorm transformer (no biases) with {a['layers']} layers, {a['heads']} attention heads "
             f"per layer, width {w}, a GELU MLP of width {w4}; {codec}; learned position embeddings with "
             f"{a['block_size']} rows, of which only rows 0-255 are used; causal attention; it predicts the next of 256 "
             f"patches; {meta['numbers']:,} float32 numbers in all, stored without a header.",
             "Data: each input is a 16 x 16 x 3 patch (768 numbers) of a 256 x 256 colour image of a galaxy (DESI Legacy "
             "Imaging Surveys), each patch normalised on its own; the 256 patches are ordered starting at the image centre "
             "and spiralling outward clockwise, one square ring at a time.",
             f"Training state: {state} (mean next-patch Huber loss {loss['model']:.3f}; predicting zeros gives "
             f"{loss['zero']:.3f}).",
             "Measured attention of each head (averaged over 64 held-out galaxies, for positions 16-254):"] + \
            [head_line(k, h) for k, h in heads.items()]
    if variant == "bare":
        items = [("I1", "the file is raw little-endian float32 numbers with no header"),
                 ("I2", "the numbers are the weights of a transformer (attention and MLP blocks)"),
                 ("I3", f"it has {a['layers']} layers"), ("I4", f"its width (residual stream) is {a['width']}"),
                 ("I5", f"it has {a['heads']} attention heads per layer"),
                 ("I6", "the MLPs are 4 times wider than the residual stream with a GELU, with LayerNorm and no biases"),
                 ("I7", "each input and output is 768 numbers: a 16 x 16 x 3 image patch"),
                 ("I8", f"it has {a['block_size']} learned position embeddings of which only the first 256 are used"),
                 ("I9", "it predicts the next element of a sequence (autoregressive, causal attention)"),
                 ("I10", "the data are colour images"), ("I11", "the images are of galaxies or other astronomical objects"),
                 ("I12", "the patches are ordered from the centre of the image outward (a spiral)"),
                 ("I13", f"how trained the network is: {ts['state']}")]
    else:
        items = [(f"H{i + 1}", f"what the answer says head {k} attends to agrees with its measured attention")
                 for i, k in enumerate(heads)]
        items.append(("T1", f"how trained the network is: {ts['state']}"))
    return "\n".join(facts), items


def judge_job(ck, judge_agent, answer, variant, cfg):
    facts, items = judge_facts_items(ck, variant)
    job = base_job(ck, judge_agent, cfg, f"judge-{variant}", effort=cfg["judge"]["effort"])
    job["weights"], job["prelude"] = None, ""
    job["phases"] = [{"name": "judge", "kind": "chat", "max_runs": 0, "validate": {"type": "json"},
                      "message": pl.judge(answer, facts, items)}]
    job["items"] = [k for k, _ in items]
    return job


def score_judge(run_dir, items):
    with open(f"{run_dir}/phase_01_judge.json") as f:
        parsed = json.load(f).get("parsed") or {}
    scores = parsed.get("scores") or {}
    vals = [float(scores.get(k, 0) or 0) for k in items]
    head_vals = [float(scores.get(k, 0) or 0) for k in items if k.startswith("H")]
    return {"score": float(np.mean(vals)) if vals else 0.0, "heads": float(np.mean(head_vals)) if head_vals else None,
            "training_state": float(scores.get("T1", scores.get("I13", 0)) or 0),
            "confabulations": int(parsed.get("confabulations") or 0), "note": parsed.get("note", "")}


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "score-exp4":
        import fcntl
        lock = open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[4])), ".lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("already being scored by another process", flush=True)
            sys.exit(73)
        res = score_exp4(sys.argv[2], sys.argv[3], sys.argv[4])
        print(json.dumps({k: res[k] for k in ("edit", "random_edit")}, indent=1))
        sys.stdout.flush()
        os._exit(0)
    sys.exit(__doc__)
