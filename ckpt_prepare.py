"""Prepare one AstroPT checkpoint for every experiment. Works for any checkpoint with astropt's standard layout
(image modality, 16 x 16 patches, AIM encoder/decoder, causal attention, no biases), any size, any training step.

  $ASTROPT_PY ckpt_prepare.py SOURCE OUT_DIR
  SOURCE: a local checkpoint file, or hf://<repo>/<path in the repo> (e.g. hf://Smith42/astroPT_v2.0/astropt/015M/ckpt.pt)

writes OUT_DIR/ (meta.json last, so a missing meta.json means "not ready"):
  weights.bin, layout.json  numbers only, little-endian float32, fixed tensor order; the layout (never shown to agents)
  meta.json                 architecture, training settings, step, measured losses, how trained it looks, the checks
  truth/attention.npy       each head's attention averaged over 64 held-out galaxies, float32 (L, H, 256, 256)
  truth/head_facts.json     per head: where its attention goes (current/previous patch, grid neighbours, centre, rings)
  truth/moments.npz         for the 40 experiment-2 moments: the predicted patch and the dependence map
  truth/physics_base.json   the physics score and next-patch loss before any edit, on dev and test
Exit code 3 = this checkpoint is not a kind we can describe (the reason is printed and saved in OUT_DIR/unsupported.txt).
"""

import fcntl
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import astro_model as am  # noqa: E402
import galaxy_data as gd  # noqa: E402
import physics  # noqa: E402
import prompts_lib as pl  # noqa: E402


class Unsupported(Exception):
    pass


def canonical_names(L, tokeniser="aim"):
    """(our name, checkpoint key) in the order of weights.bin. The AIM tokeniser has a two-layer MLP encoder and decoder
    (enc_fc, enc_proj / dec_fc, dec_proj); the affine one a single linear layer each way (enc_fc / dec_fc)."""
    aim = tokeniser == "aim"
    names = [("enc_fc", "encoders.images.c_fc.weight")] + \
        ([("enc_proj", "encoders.images.c_proj.weight")] if aim else []) + [("pos_emb", "embedders.images.wpe.weight")]
    for l in range(L):
        names += [(f"ln1.{l}", f"transformer.h.{l}.ln_1.weight"), (f"W_qkv.{l}", f"transformer.h.{l}.attn.c_attn.weight"),
                  (f"W_o.{l}", f"transformer.h.{l}.attn.c_proj.weight"), (f"ln2.{l}", f"transformer.h.{l}.ln_2.weight"),
                  (f"W_fc.{l}", f"transformer.h.{l}.mlp.c_fc.weight"), (f"W_proj.{l}", f"transformer.h.{l}.mlp.c_proj.weight")]
    return names + [("lnf", "transformer.ln_f.weight"), ("dec_fc", "decoders.images.c_fc.weight")] + \
        ([("dec_proj", "decoders.images.c_proj.weight")] if aim else [])


def hf_parts(source):
    """hf://owner/repo/path or hf://datasets/owner/repo/path -> (repo, path, repo_type)."""
    rest, kind = (source[len("hf://datasets/"):], "dataset") if source.startswith("hf://datasets/") else (source[5:], "model")
    parts = rest.split("/")
    return "/".join(parts[:2]), "/".join(parts[2:]), kind


def fetch(source):
    if source.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        repo, path, kind = hf_parts(source)
        try:
            return hf_hub_download(repo, path, repo_type=kind)
        except Exception:  # a stale saved token can break even public downloads: try without it
            return hf_hub_download(repo, path, repo_type=kind, token=False)
    return source


def standard_registry():
    """The image modality astropt's image-only training uses (what some training scripts rebuild in code)."""
    from astropt.model import ModalityConfig, ModalityRegistry
    return ModalityRegistry([ModalityConfig(name="images", input_size=768, patch_size=16, embed_pos=True, pos_input_size=1)])


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(ck, dict) or "model" not in ck:
        raise Unsupported("not an astropt training checkpoint (no 'model' entry)")
    ma, cfg = ck.get("model_args") or {}, ck.get("config") or {}
    try:
        L, H, d, B = ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"]
    except KeyError as e:
        raise Unsupported(f"model_args lacks {e}")
    if ma.get("modalities"):
        try:
            (mod,) = ma["modalities"]
            kind = (mod.name, mod.input_size, mod.patch_size, mod.embed_pos, mod.vocab_size)
        except Exception as e:
            raise Unsupported(f"unusual modalities ({type(e).__name__}: {e})")
        if kind != ("images", 768, 16, True, 0):
            raise Unsupported(f"modality {kind}, expected images of 16 x 16 x 3 patches with learned positions")
    objective = ma.get("objective", cfg.get("objective", "ar"))
    attn = ma.get("attn_type", cfg.get("attn_type", "causal"))
    tokeniser = ma.get("tokeniser", cfg.get("tokeniser", "aim"))
    if objective != "ar" or attn != "causal":
        raise Unsupported(f"a '{objective}' network with '{attn}' attention: the experiments are written for "
                          "next-patch (autoregressive, causal) networks")
    if tokeniser not in ("aim", "affine"):
        raise Unsupported(f"tokeniser '{tokeniser}' (supported: aim, affine)")
    if ma.get("n_chan", 3) != 3 or d % H or B < 256 or cfg.get("bias", False) or cfg.get("spiral", True) is not True \
            or cfg.get("image_size", 256) != 256 or ma.get("lora_r", 0):
        raise Unsupported("needs 3 colours, no biases, spiral order, 256 x 256 images, no LoRA")
    old = {"transformer.wte.images.": "encoders.images.", "lm_head.images.": "decoders.images.",
           "transformer.wpe.images.wpe.": "embedders.images.wpe."}
    sd = {}
    for k, v in ck["model"].items():
        k = k.removeprefix("_orig_mod.")
        for a, b in old.items():
            if k.startswith(a):
                k = b + k[len(a):]
        sd[k] = v
    want = {k for _, k in canonical_names(L, tokeniser)}
    if set(sd) != want:
        raise Unsupported(f"unexpected tensors: {sorted(set(sd) ^ want)[:6]}")
    if tuple(sd["encoders.images.c_fc.weight"].shape)[1] != 768:
        raise Unsupported("the encoder does not take 768-number patches")
    registry = ck.get("modality_registry") or standard_registry()
    return ck, ma, cfg, sd, tokeniser, registry


def head_facts(A):
    """A: (L, H, 256, 256) mean attention. For each head, where the attention of positions 16..254 goes, relative to
    the patch being predicted (ORDER[i + 1]) and the current one (ORDER[i])."""
    order = np.array(am.ORDER)
    facts = {}
    rows = range(16, 255)
    for l in range(A.shape[0]):
        for h in range(A.shape[1]):
            acc = dict(current=0.0, previous=0.0, neighbours=0.0, centre=0.0, same_ring=0.0, ring_inside=0.0,
                       spread=0.0, distance=0.0)
            for i in rows:
                a = A[l, h, i, :i + 1].astype(np.float64)
                tgt = order[i + 1]
                cheb = np.abs(order[:i + 1] - tgt).max(1)
                acc["current"] += a[i]
                acc["previous"] += a[i - 1]
                acc["neighbours"] += a[cheb == 1].sum()
                acc["centre"] += a[:4].sum()
                acc["same_ring"] += a[am.RING[:i + 1] == am.RING[i + 1]].sum()
                acc["ring_inside"] += a[am.RING[:i + 1] == am.RING[i + 1] - 1].sum()
                p = a[a > 0]
                acc["spread"] += float(-(p * np.log(p)).sum() / np.log(i + 1))
                acc["distance"] += float((a * np.sqrt(((order[:i + 1] - tgt) ** 2).sum(1))).sum())
            facts[f"L{l}H{h}"] = {k: round(v / len(rows), 4) for k, v in acc.items()}
    return facts


def drop_download(source, path):
    """Delete a checkpoint downloaded from Hugging Face (the cached file and its link): weights.bin has all we need."""
    if not source.startswith("hf://"):
        return
    real = os.path.realpath(path)
    if "/huggingface/hub/" in real:
        for p in (path, real):
            if os.path.lexists(p):
                os.remove(p)


def main(source, out, drop=False):
    t0 = time.time()
    os.makedirs(f"{out}/truth", exist_ok=True)
    lock = open(f"{out}/.lock", "w")  # one process per checkpoint folder
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{out} is already being prepared by another process", flush=True)
        sys.exit(73)
    for f in ("meta.json", "unsupported.txt"):
        if os.path.exists(f"{out}/{f}"):
            os.remove(f"{out}/{f}")
    path = fetch(source)
    try:
        ck, ma, cfg, sd, tokeniser, registry = load_checkpoint(path)
    except Unsupported as e:
        with open(f"{out}/unsupported.txt", "w") as f:
            f.write(str(e) + "\n")
        print(f"unsupported checkpoint: {e}", flush=True)
        if drop:
            drop_download(source, path)
        sys.exit(3)
    L, H, d, B = ma["n_layer"], ma["n_head"], ma["n_embd"], ma["block_size"]

    # weights.bin, written to a new file and swapped in (runs that hard-link the old file keep it unchanged)
    layout, off = [], 0
    sha = hashlib.sha256()
    with open(f"{out}/weights.bin.tmp", "wb") as f:
        for name, key in canonical_names(L, tokeniser):
            a = np.ascontiguousarray(sd[key].detach().to(torch.float32).numpy(), dtype="<f4")
            if not np.isfinite(a).all():
                raise SystemExit(f"{name} has non-finite numbers")
            f.write(a.tobytes())
            sha.update(a.tobytes())
            layout.append({"name": name, "ckpt_key": key, "shape": list(a.shape), "offset": off, "count": a.size})
            off += a.size
    arch = {"layers": L, "heads": H, "width": d, "block_size": B, "tokeniser": tokeniser}
    with open(f"{out}/layout.json", "w") as f:
        json.dump({"source": source, "numbers": off, "format": "float32 little-endian, row-major, no header",
                   "sha256": sha.hexdigest(), "arch": arch, "tensors": layout}, f, indent=1)
    os.replace(f"{out}/weights.bin.tmp", f"{out}/weights.bin")
    print(f"weights.bin: {off:,} numbers ({4 * off / 1e9:.2f} GB), {L} layers x {H} heads, width {d}", flush=True)

    # checks: our GPU network against the official model; the prompt's numpy code and image pipeline against both
    from astropt.model import GPT, GPTConfig
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = am.Net.load(f"{out}/weights.bin", device=dev)
    gpt_config = GPTConfig(**{k: v for k, v in ma.items() if k in GPTConfig.__dataclass_fields__ and k != "modalities"})
    gpt_config.tokeniser = tokeniser
    model = GPT(gpt_config, registry, master_process=False)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model = model.to(dev).eval()
    iter_num = ck.get("iter_num")
    del ck, sd
    imgs, labels, splits = gd.load()
    vseq = am.images_to_sequences(imgs[splits["verify"]])
    x = torch.as_tensor(vseq[:8], device=dev)
    with torch.no_grad():
        off_pred, _ = model({"images": x, "images_positions": torch.arange(256, device=dev)[None].expand(8, -1)})
        ours = net.forward(x)
    scale = max(1.0, off_pred["images"].abs().max().item())
    diff_net = (ours - off_pred["images"]).abs().max().item() / scale
    del model
    torch.cuda.empty_cache()
    meta_partial = {"arch": arch, "numbers": off, "training": {}, "measured": {}}
    ns = {}
    exec(pl.code(meta_partial), ns)
    Wnp = ns["load_weights"](f"{out}/weights.bin")
    pred_np = np.stack([ns["predict"](vseq[g].astype(np.float64), Wnp) for g in range(2)])
    diff_prompt = float(np.abs(pred_np - ours[:2].cpu().numpy()).max() / scale)
    diff_pipe = float(np.abs(np.stack([ns["image_to_sequence"](imgs[i]) for i in splits["verify"][:2]]) - vseq[:2]).max())
    ok = diff_net < 1e-3 and diff_prompt < 1e-3 and diff_pipe < 1e-5
    print(f"checks: GPU network vs official {diff_net:.1e}, prompt code vs GPU network {diff_prompt:.1e}, "
          f"image pipeline {diff_pipe:.1e} -> {'ok' if ok else 'FAILED'}", flush=True)
    if not ok:
        raise SystemExit("check failed")

    # measured losses (the full prompt quotes them), and how trained the network looks
    zero = float(np.mean([physics_huber(np.zeros_like(s[1:]), s[1:]) for s in vseq]))
    prev = float(np.mean([physics_huber(s[:-1], s[1:]) for s in vseq]))
    model_loss = net.loss(vseq)
    ratio = model_loss / zero
    state = "untrained" if ratio > 0.95 else ("well trained" if ratio < 0.6 else "partly trained")

    # truth: attention, head facts, moments, physics before editing
    aseq = am.images_to_sequences(imgs[splits["attn"]])
    sums = None
    for i in range(0, len(aseq), 8):
        _, maps = net.forward(torch.as_tensor(aseq[i:i + 8], device=dev), attn=True)
        s = torch.stack([m.sum(0) for m in maps]).double().cpu().numpy()
        sums = s if sums is None else sums + s
    A = (sums / len(aseq)).astype(np.float32)
    np.save(f"{out}/truth/attention.npy", A)
    with open(f"{out}/truth/head_facts.json", "w") as f:
        json.dump(head_facts(A), f, indent=1)
    moms, mseq = gd.moments()
    pred_patch, dep = [], []
    for m, s in zip(moms, mseq):
        n = m["n"]
        pred_patch.append(net.forward(torch.as_tensor(s[:n][None], device=dev))[0, n - 1].cpu().numpy())
        dep.append(net.dependence(s, n))
    np.savez(f"{out}/truth/moments.npz", ids=np.array([m["id"] for m in moms]), pred=np.stack(pred_patch),
             dep=np.stack(dep))
    del net
    torch.cuda.empty_cache()
    base = {s: physics.evaluate(f"{out}/weights.bin", s, device=dev) for s in ("dev", "test")}
    # the same architecture with fresh random weights: the physics score a network gets before any training
    torch.manual_seed(0)
    rnd = GPT(gpt_config, registry, master_process=False).state_dict()
    rW = {name: rnd[key].detach().float().numpy() for name, key in canonical_names(L, tokeniser)}
    del rnd
    rnet = am.Net(rW, H, dev)
    base["random_init"] = {s: physics.evaluate_net(rnet, s) for s in ("dev", "test")}
    del rnet, rW
    torch.cuda.empty_cache()
    with open(f"{out}/truth/physics_base.json", "w") as f:
        json.dump(base, f, indent=1)

    t = cfg
    training = {"learning_rate": t.get("learning_rate"), "min_lr": t.get("min_lr"), "warmup_iters": t.get("warmup_iters"),
                "beta1": t.get("beta1"), "beta2": t.get("beta2"), "weight_decay": t.get("weight_decay"),
                "grad_clip": t.get("grad_clip"), "dtype": t.get("dtype", "bfloat16"),
                "images_per_step": int(t.get("batch_size", 16)) * int(t.get("gradient_accumulation_steps", 40)),
                "steps": int(iter_num) if iter_num is not None else None}
    nparams = off
    label = (f"{nparams / 1e6:.1f}M params{', affine' if tokeniser == 'affine' else ''}, "
             f"step {iter_num if iter_num is not None else '?'}")
    meta = {"source": source, "file": os.path.abspath(path), "label": label, "arch": arch, "numbers": off,
            "sha256": sha.hexdigest(), "training": training,
            "measured": {"model": model_loss, "zero": zero, "prev": prev, "galaxies": len(vseq)},
            "training_state": {"state": state, "loss_over_zero_baseline": round(ratio, 4)},
            "checks": {"gpu_vs_official": diff_net, "prompt_vs_gpu": diff_prompt, "image_pipeline": diff_pipe},
            "physics_base": {s: {"score": base[s]["score"], "loss": base[s]["loss"]} for s in ("dev", "test")},
            "physics_random_init": {s: base["random_init"][s]["score"] for s in ("dev", "test")},
            "seconds": round(time.time() - t0), "ready": True}
    with open(f"{out}/meta.json.tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(f"{out}/meta.json.tmp", f"{out}/meta.json")
    print(f"ready in {meta['seconds']} s: {label}; loss {model_loss:.4f} (zeros {zero:.4f}) -> {state}; "
          f"physics score dev {base['dev']['score']:.4f}, test {base['test']['score']:.4f}", flush=True)
    if drop:
        drop_download(source, path)


def physics_huber(a, b):
    d = np.abs(np.asarray(a, dtype=np.float64) - b)
    return float(np.where(d < 1, 0.5 * d ** 2, d - 0.5).mean())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--drop-download"]
    if len(args) != 2:
        sys.exit(__doc__)
    main(args[0], os.path.abspath(args[1]), drop="--drop-download" in sys.argv)
    sys.stdout.flush()
    os._exit(0)
