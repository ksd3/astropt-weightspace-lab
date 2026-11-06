"""The physics score of a checkpoint, optionally with an agent's edits (experiment 4).

astropt's standard galaxy embedding (the residual stream after block L // 2, averaged over the 256 positions) is read
out by eight linear readouts (ridge regression on the standardised embedding, refitted every time on the same 2,000
galaxies): six physical properties, redshift, stellar mass (log), specific star-formation rate (log), physical size
in kpc (log) and the colours g-r and r-z, and two Galaxy Zoo morphology measures, the fractions of volunteers who
classed the galaxy as featured or a disc, and as a merger. The physics score is their mean R^2 on the dev (500) or
test (1,000) galaxies (also reported: the physical and the morphology parts separately). The guard is the mean
next-patch Huber loss on the same galaxies.

  python physics.py WEIGHTS --split dev|test [--edits DIR] [--random-like] [--seed N] [--out FILE] [--brief]

--random-like replaces every edit by a random change of the same size (Frobenius norm) to the same tensor: the
baseline an agent's edit is compared with.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astro_model as am  # noqa: E402
import galaxy_data as gd  # noqa: E402

ALPHAS = np.logspace(-3, 4, 15)


def scaling_problem(w, e):
    """True if e is the whole tensor w multiplied by one number (not allowed as an edit)."""
    w, e = w.astype(np.float64).ravel(), e.astype(np.float64).ravel()
    ww = w @ w
    if ww == 0:
        return False
    c = (w @ e) / ww
    return abs(c - 1) > 1e-6 and np.linalg.norm(e - c * w) <= 1e-6 * max(np.linalg.norm(e), 1e-30)


def embed_split(net, imgs, idx, chunk=64):
    import torch
    out = []
    for i in range(0, len(idx), chunk):
        x = torch.as_tensor(am.images_to_sequences(imgs[idx[i:i + chunk]]), device=net.device)
        out.append(net.embed(x).float().cpu().numpy())
    return np.concatenate(out)


def loss_split(net, imgs, idx, chunk=32):
    total, n = 0.0, 0
    for i in range(0, len(idx), chunk):
        seqs = am.images_to_sequences(imgs[idx[i:i + chunk]])
        total += net.loss(seqs) * len(seqs)
        n += len(seqs)
    return total / n


def evaluate_net(net, split):
    """Physics score, per-property R^2, morphology R^2 and next-patch loss of a network on the dev or test galaxies."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    imgs, labels, splits = gd.load()
    tr, ev = splits["probe_train"], splits[split]
    z_tr, z_ev = embed_split(net, imgs, tr), embed_split(net, imgs, ev)
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    z_tr, z_ev = (z_tr - mu) / sd, (z_ev - mu) / sd
    ok = np.isfinite(z_tr).all() and np.isfinite(z_ev).all()

    def readout(get):
        y_tr = np.array([get(labels[i]) for i in tr], dtype=np.float64)
        y_ev = np.array([get(labels[i]) for i in ev], dtype=np.float64)
        m_tr, m_ev = np.isfinite(y_tr), np.isfinite(y_ev)
        if not ok:
            return 0.0
        probe = RidgeCV(alphas=ALPHAS).fit(z_tr[m_tr], y_tr[m_tr])
        return float(max(-1.0, r2_score(y_ev[m_ev], probe.predict(z_ev[m_ev]))))

    r2 = {k: readout(lambda row, k=k: gd.target(row, k)) for k in gd.PHYSICS}
    return {"split": split, "score": float(np.mean(list(r2.values()))),
            "score_physical": float(np.mean([r2[k] for k in gd.PHYSICAL])),
            "score_morphology": float(np.mean([r2[k] for k in gd.MORPHOLOGY])), "r2": r2,
            "loss": loss_split(net, imgs, ev)}


def evaluate(weights, split, edit_dir=None, random_like=False, seed=0, device=None):
    t0 = time.time()
    W = am.load_numpy(weights)
    edits, problems = am.read_edits(edit_dir, W)
    for name in [k for k, v in edits.items() if scaling_problem(W[k], v)]:
        problems.append(f"{name}: multiplies the whole tensor by one number, ignored")
        edits.pop(name)
    change = {k: float(np.linalg.norm(v.astype(np.float64) - W[k])) for k, v in edits.items()}
    if random_like:
        rng = np.random.default_rng(seed)
        for k in list(edits):
            r = rng.standard_normal(W[k].shape)
            edits[k] = (W[k] + r * (change[k] / max(np.linalg.norm(r), 1e-30))).astype(np.float32)
    net = am.Net.load(weights, edits=edits, device=device)
    res = evaluate_net(net, split)
    res.update(edited_tensors=sorted(edits), change_norm=change, problems=problems, random_like=bool(random_like),
               seconds=round(time.time() - t0, 1))
    return res


def brief(res, base=None):
    """The text an agent sees after EVALUATE."""
    r2 = ", ".join(f"{gd.PHYSICS_NAMES[k]} {v:.3f}" for k, v in res["r2"].items())
    s = f"physics score {res['score']:.4f} (R^2: {r2}); next-patch loss {res['loss']:.4f}"
    if base:
        s += (f". Before any edit: physics score {base['score']:.4f}, next-patch loss {base['loss']:.4f} "
              f"(the limit is {1.02 * base['loss']:.4f}).")
    if res["edited_tensors"]:
        s += f" Edited tensors: {', '.join(res['edited_tensors'])}."
    else:
        s += " No edits saved yet."
    if res["problems"]:
        s += " Problems: " + "; ".join(res["problems"]) + "."
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--edits")
    ap.add_argument("--random-like", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--brief", action="store_true", help="print the line an agent sees, with --base FILE")
    ap.add_argument("--base")
    args = ap.parse_args()
    res = evaluate(args.weights, args.split, args.edits, args.random_like, args.seed)
    if args.out:
        with open(args.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(args.out + ".tmp", args.out)
    if args.brief:
        base = json.load(open(args.base))[args.split] if args.base else None
        print(brief(res, base))
    else:
        print(json.dumps(res, indent=1))
