"""AstroPT computed straight from a numbers-only weights.bin (PyTorch, any device), plus the image pipeline.

Matches the numpy code the agents are given and the official astropt model (ckpt_prepare.py checks both on every
checkpoint). Used for everything we measure: attention, dependence maps, losses, embeddings, physics scores.
Needs only numpy and torch, so it also runs inside the sandbox.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from spiral import GRID, NPATCH, ORDER, ORDER_IDX, PATCH, PATCH_DIM, RING, patch_to_vector, vector_to_patch  # noqa: F401


def images_to_sequences(imgs):
    """(N, 256, 256, 3) RGB images, row 0 at the top -> (N, 256, 768) float32 input sequences, exactly like
    image_to_sequence in the prompt: entry (x * 16 + y) * 3 + colour, each patch to mean 0, std 1 (ddof 1) + 1e-8."""
    x = np.asarray(imgs, dtype=np.float64)
    if x.ndim == 3:
        x = x[None]
    n = x.shape[0]
    x = x.reshape(n, GRID, PATCH, GRID, PATCH, 3).transpose(0, 1, 3, 4, 2, 5)  # (N, R, C, x, y, colour)
    x = x.reshape(n, GRID * GRID, PATCH_DIM)[:, ORDER_IDX]
    mu, sd = x.mean(-1, keepdims=True), x.std(-1, ddof=1, keepdims=True)
    return ((x - mu) / (sd + 1e-8)).astype(np.float32)


def read_layout(weights_path):
    with open(os.path.join(os.path.dirname(os.path.abspath(weights_path)), "layout.json")) as f:
        return json.load(f)


def load_numpy(weights_path, layout=None):
    """name -> numpy array (read-only views of the file)."""
    layout = layout or read_layout(weights_path)
    flat = np.fromfile(weights_path, dtype="<f4")
    assert flat.size == layout["numbers"], (flat.size, layout["numbers"])
    return {t["name"]: flat[t["offset"]: t["offset"] + t["count"]].reshape(t["shape"]) for t in layout["tensors"]}


def read_edits(edit_dir, W):
    """Load an agent's saved edits (<name>.npy); returns (edits, problems). Invalid files are skipped and reported."""
    edits, problems = {}, []
    if not edit_dir or not os.path.isdir(edit_dir):
        return edits, problems
    for f in sorted(os.listdir(edit_dir)):
        if not f.endswith(".npy"):
            continue
        name = f[:-4]
        if name not in W:
            problems.append(f"{name}: not a tensor of this network")
            continue
        try:
            a = np.load(os.path.join(edit_dir, f), allow_pickle=False).astype(np.float32)
        except Exception as e:
            problems.append(f"{name}: unreadable ({type(e).__name__})")
            continue
        if a.shape != W[name].shape or not np.isfinite(a).all():
            problems.append(f"{name}: wrong shape or not finite")
            continue
        edits[name] = a
    return edits, problems


class Net:
    """The network. forward() returns next-patch predictions; it can also return attention maps, switch heads off,
    or stop after a block (the residual stream there)."""

    def __init__(self, W, heads, device="cuda"):
        self.device = device
        self.W = {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=device) for k, v in W.items()}
        self.L = sum(1 for k in W if k.startswith("ln1."))
        self.d = self.W["lnf"].shape[0]
        self.H = heads
        self.hd = self.d // heads
        self.affine = "enc_proj" not in W  # the affine tokeniser: one linear layer in (enc_fc) and one out (dec_fc)

    @classmethod
    def load(cls, weights_path, edits=None, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        layout = read_layout(weights_path)
        W = dict(load_numpy(weights_path, layout))
        for k, v in (edits or {}).items():
            W[k] = v
        return cls(W, layout["arch"]["heads"], device)

    @torch.no_grad()
    def forward(self, x, attn=False, head_scale=None, stop_after=None):
        """x: (B, T, 768). Returns predictions (B, T, 768) [row t = prediction of patch t + 1]; with attn=True also a
        list of (B, H, T, T) attention maps per layer; head_scale: (L, H) multipliers of each head's output;
        stop_after=l: return the residual stream (B, T, d) right after block l instead."""
        W, B, T = self.W, x.shape[0], x.shape[1]
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        h = (x @ W["enc_fc"].T if self.affine else
             F.gelu(x @ W["enc_fc"].T, approximate="tanh") @ W["enc_proj"].T) + W["pos_emb"][:T]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=self.device), 1)
        maps = []
        for l in range(self.L):
            a = F.layer_norm(h, (self.d,), W[f"ln1.{l}"], None, 1e-5)
            q, k, v = (a @ W[f"W_qkv.{l}"].T).split(self.d, dim=-1)
            q, k, v = (t.view(B, T, self.H, self.hd).transpose(1, 2) for t in (q, k, v))
            p = torch.softmax(q @ k.transpose(-1, -2) / self.hd ** 0.5 + mask, dim=-1)
            if attn:
                maps.append(p)
            o = p @ v
            if head_scale is not None:
                o = o * torch.as_tensor(head_scale, dtype=torch.float32, device=self.device)[l].view(1, self.H, 1, 1)
            h = h + o.transpose(1, 2).reshape(B, T, self.d) @ W[f"W_o.{l}"].T
            h = h + F.gelu(F.layer_norm(h, (self.d,), W[f"ln2.{l}"], None, 1e-5) @ W[f"W_fc.{l}"].T,
                           approximate="tanh") @ W[f"W_proj.{l}"].T
            if stop_after == l:
                return h
        h = F.layer_norm(h, (self.d,), W["lnf"], None, 1e-5)
        out = h @ W["dec_fc"].T if self.affine else F.gelu(h @ W["dec_fc"].T, approximate="tanh") @ W["dec_proj"].T
        return (out, maps) if attn else out

    @torch.no_grad()
    def embed(self, x):
        """astropt's standard galaxy embedding: the residual stream after block L // 2, averaged over the positions."""
        return self.forward(x, stop_after=self.L // 2).mean(1)

    @torch.no_grad()
    def batched(self, seqs, fn, batch=16):
        """Apply fn (e.g. self.embed) to a big numpy array of sequences in batches; returns numpy."""
        out = [fn(torch.as_tensor(seqs[i:i + batch], device=self.device)).float().cpu().numpy()
               for i in range(0, len(seqs), batch)]
        return np.concatenate(out)

    @torch.no_grad()
    def loss(self, seqs, batch=16):
        """Mean Huber loss (delta 1) of the predictions for patches 1..255, as in training."""
        total, count = 0.0, 0
        for i in range(0, len(seqs), batch):
            x = torch.as_tensor(seqs[i:i + batch], device=self.device)
            pred = self.forward(x[:, :-1])
            total += F.huber_loss(pred, x[:, 1:], reduction="sum").item()
            count += pred.numel()
        return total / count

    @torch.no_grad()
    def dependence(self, seq, n, chunk=32):
        """How much the prediction of patch n (from patches 0..n-1) depends on each earlier patch: blank (zero) each
        one in turn and measure the squared change of the prediction. Returns a (16, 16) map summing to 1."""
        x = torch.as_tensor(seq[:n], dtype=torch.float32, device=self.device)[None]
        base = self.forward(x)[0, n - 1]
        delta = torch.zeros(n, device=self.device)
        for s in range(0, n, chunk):
            idx = torch.arange(s, min(n, s + chunk), device=self.device)
            xs = x.repeat(len(idx), 1, 1)
            xs[torch.arange(len(idx)), idx] = 0.0
            delta[idx] = ((self.forward(xs)[:, n - 1] - base) ** 2).sum(-1)
        m = np.zeros((GRID, GRID))
        for j, v in enumerate(delta.cpu().numpy()):
            m[ORDER[j]] = v
        return m / m.sum() if m.sum() > 0 else m
