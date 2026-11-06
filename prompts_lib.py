"""Every prompt the agents get, generated for a given checkpoint (architecture, training details, measurements).

blind=True hides how far the network has been trained (no step count, no measured loss) and says it may come from
any point of training, from the random start to the end: used when sweeping training checkpoints (experiment 3).
"""

import string

from spiral import ORDER

CODE = '''import numpy as np

# weights.bin is ${total} float32 numbers (little-endian) and nothing else: the tensors below, one after
# another in this order, each stored row-major. Matrices are (out, in) and are applied as x @ M.T.
${shapes_head}
for l in range(${L}):
    SHAPES += [(f"ln1.{l}", (${d},)), (f"W_qkv.{l}", (${d3}, ${d})), (f"W_o.{l}", (${d}, ${d})),
               (f"ln2.{l}", (${d},)), (f"W_fc.{l}", (${F}, ${d})), (f"W_proj.{l}", (${d}, ${F}))]
${shapes_tail}


def load_weights(path="weights.bin"):
    flat, W, i = np.fromfile(path, dtype="<f4"), {}, 0
    for name, shape in SHAPES:
        n = int(np.prod(shape))
        W[name] = flat[i:i + n].reshape(shape)
        i += n
    assert i == flat.size
    return W


# The image is cut into a 16 x 16 grid of patches of 16 x 16 pixels. ORDER[k] = (row, col) on that grid of the
# k-th patch of the sequence (row 0 = top, col 0 = left).
ORDER = [
${order}
]


def image_to_sequence(img):
    """img: (256, 256, 3) RGB image, row 0 at the top. Returns the (256, 768) input sequence."""
    seq = np.zeros((256, 768))
    for k, (r, c) in enumerate(ORDER):
        p = img[16 * r:16 * r + 16, 16 * c:16 * c + 16].astype(np.float64)  # (16 rows, 16 columns, 3 colours)
        v = p.transpose(1, 0, 2).reshape(768)    # entry (x * 16 + y) * 3 + colour, x = column and y = row in the patch
        seq[k] = (v - v.mean()) / (v.std(ddof=1) + 1e-8)                    # each patch normalised on its own
    return seq


def sequence_to_image(seq):
    """(n, 768) sequence -> (256, 256, 3) image; each patch keeps its own normalisation, patches beyond n stay 0."""
    img = np.zeros((256, 256, 3))
    for k, (r, c) in enumerate(ORDER[:len(seq)]):
        img[16 * r:16 * r + 16, 16 * c:16 * c + 16] = np.asarray(seq[k]).reshape(16, 16, 3).transpose(1, 0, 2)
    return img


def gelu(x):                                          # tanh approximation
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, w):                                 # over the last axis, eps 1e-5, no bias
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * w


def predict(seq, W):
    """seq: (n, 768) normalised patches in sequence order, 1 <= n <= 256.
    Returns (n, 768): row t is the network's prediction of patch t + 1."""
    n, d, H = len(seq), ${d}, ${H}
    hd = d // H
    ${encode}
    for l in range(${L}):
        h = layer_norm(x, W[f"ln1.{l}"])
        q, k, v = np.split(h @ W[f"W_qkv.{l}"].T, 3, axis=-1)           # rows 0-${d_1} of W_qkv give q, ${d}-${d2_1} k, ${d2}-${d3_1} v
        out = np.zeros((n, d))
        for j in range(H):                                                # head j uses dimensions ${hd}j .. ${hd}j+${hd_1}
            s = slice(j * hd, (j + 1) * hd)
            att = q[:, s] @ k[:, s].T / np.sqrt(hd)                       # (n, n)
            att = att + np.triu(np.full((n, n), -np.inf), 1)              # causal: position i sees positions 0..i
            att = np.exp(att - att.max(-1, keepdims=True))
            att = att / att.sum(-1, keepdims=True)
            out[:, s] = att @ v[:, s]
        x = x + out @ W[f"W_o.{l}"].T
        h = layer_norm(x, W[f"ln2.{l}"])
        x = x + gelu(h @ W[f"W_fc.{l}"].T) @ W[f"W_proj.{l}"].T
    h = layer_norm(x, W["lnf"])
    ${decode}'''

INTRO = {False: "weights.bin holds the trained weights of a neural network that generates images of galaxies. It is a "
                "transformer with ${L} layers and ${H} attention heads per layer.",
         True: "weights.bin holds the weights of a neural network that is being trained to generate images of galaxies; "
               "this copy may come from any point of its training, from the very start (random weights) to the end. "
               "It is a transformer with ${L} layers and ${H} attention heads per layer."}
INTRO_TAIL = (" It reads a galaxy image as a sequence of 256 patches, starting at the centre of the image and spiralling "
              "outward, and at every position it predicts the next patch. This is exactly how it computes (numpy):")

DATA = ("The data: colour (RGB) JPEG images of galaxies from the DESI Legacy Imaging Surveys, 256 x 256 pixels, each "
        "centred on a galaxy. ORDER starts at (7, 8), just above and to the right of the image centre, and goes clockwise "
        "round the centre one square ring at a time: ring r (r = 1..8) is the border of the central 2r x 2r block of "
        "patches and takes positions (2r-2)^2 to (2r)^2 - 1 (ring 1 is positions 0-3, ring 2 is 4-15, ..., ring 8, the "
        "edge of the image, is 196-255). Each ring starts in its top row and ends at its top-left corner; the last patch, "
        "ORDER[255], is the top-left corner of the image. Each patch is normalised on its own (mean 0, standard deviation "
        "1 over its 768 numbers), so the network never sees how bright a patch is, only its pattern. Every image is 256 "
        "patches long, so only rows 0-255 of pos_emb are ever used.")

TRAINING = ("Training: from random initialization (every matrix entry drawn from a normal distribution with standard "
            "deviation 0.02, or ${cproj_std} for ${cproj_names}; LayerNorm weights 1), next-patch "
            "prediction: for t = 0..254, output row t is compared with patch t + 1 of the real image by the Huber loss "
            "(delta 1, averaged over all numbers). AdamW (betas ${b1} and ${b2}, weight decay ${wd} on all matrices "
            "including pos_emb and none on the LayerNorm weights, gradient clipping at ${clip}), learning rate warmed up "
            "to ${lr} over ${warmup} steps and then cosine-decayed towards ${min_lr}; ")
TRAINING_END = {False: "${steps} steps of ${imgs} images each, in ${dtype} mixed precision.",
                True: "batches of ${imgs} images, in ${dtype} mixed precision. This copy was saved after an unknown "
                      "number of training steps, possibly none."}

MEASURED = ("Measured on 32 held-out galaxy images (sequences from image_to_sequence): the mean Huber loss of "
            "predict(seq, W)[:-1] against seq[1:] is ${loss_model}. For comparison, predicting all zeros gives "
            "${loss_zero}, and predicting that each patch repeats the one before it gives ${loss_prev}.")

PROTOCOL = ("To run code, reply with exactly one code block that starts with ```python and ends with ```, and nothing "
            "else. We run it and reply with its printed output. Available: numpy, scipy, scikit-learn, PyTorch (with a "
            "CUDA GPU), pandas, sympy, networkx.")
LIMITS = ("Each run is a fresh process (nothing is kept between runs), may take up to 600 seconds, and its output is cut "
          "at 20,000 characters.")

EXP1_TAIL = '''WHAT YOU CAN USE. weights.bin, which holds all the weights. You may use any computation on these weights, with any method you like: multiply them, decompose them, compare, cluster or fit them, write any analysis you want. Two things are NOT allowed:
1. A forward pass: do not run the network or any of its parts (${enc_dec_parts}, LayerNorm, attention with its softmax, the MLP) on any input, whether real, random or constructed, in code, in your head or on paper. Treating the weight matrices as linear-algebra objects (products, eigenvalues, singular vectors, applying one weight matrix to directions taken from other weights, such as rows of pos_emb) is fine; simulating what the network does to an input is not.
2. Search: do not look anything up (there is no internet access in any case).

HOW YOU COMPUTE. The weights are not written out here: you read them by running Python code. ${protocol} Before your code runs, the code above has been run (so SHAPES, load_weights, ORDER, image_to_sequence, sequence_to_image, gelu, layer_norm and predict are defined) and W = load_weights() holds the weights: W["W_qkv.0"] is ${d3} x ${d}, W["pos_emb"] is ${B} x ${d}, and so on; matrices are (out, in), as used in the code above. Your code runs in a folder that holds weights.bin and nothing else (no other files, no network). ${limits} You may run code up to ${runs} times. When you are done, reply WITHOUT a code block, with your final answer in the order given in the question.

QUESTION. How does this network generate a galaxy, and what does each attention head do? Answer in this order:
Observations: <what you see in the weights and what it suggests; quote the numbers you rely on>
Heads: <one line per head, L0H0 .. L0H${H_1}, L1H0 .. L${L_1}H${H_1} (L = layer, H = head): which earlier patches it mostly attends to, in terms of the sequence and of the 16 x 16 grid (for example the previous patch, a particular neighbour on the grid, the centre, or everything so far), what it carries over, and your evidence>
Encoder and decoder: <what ${enc_names} ${pick} up from a patch, and what ${dec_names} can draw>
Mechanism: <how the network builds up a galaxy from the centre outward, in plain words>
'''

EXP1_BARE = '''weights.bin contains the raw weights of a neural network and nothing else. Figure out what's going on: what this network is, how it is built and what it does.

HOW YOU COMPUTE. You read weights.bin by running Python code. ${protocol} numpy is imported as np before your code runs. Your code runs in a folder that holds weights.bin and nothing else (no other files, no network). ${limits} You may run code up to ${runs} times. When you are done, reply WITHOUT a code block, with what you found.
'''

GENERATE = '''def generate(seq, n_total=256, seed=0):
    """Continue a sequence (n, 768) to n_total patches, one patch at a time: each new patch is the network's predicted
    patch plus noise of the size the prediction leaves unexplained, normalised like a real patch."""
    rng = np.random.default_rng(seed)
    seq = list(np.asarray(seq, dtype=np.float64))
    while len(seq) < n_total:
        p = predict(np.array(seq), W)[-1]
        p = p + np.sqrt(max(0.0, 1.0 - p.var())) * rng.standard_normal(768)
        seq.append((p - p.mean()) / (p.std(ddof=1) + 1e-8))
    return np.array(seq)'''

EXP2_STUDY = '''GENERATION. The network builds an image one patch at a time: given patches 0..n-1 (in ORDER), row n-1 of predict() is its prediction of patch n, which is appended, and so on. predict() gives the expected (average) next patch, which is smoother than a real patch; generate() adds noise of the size the prediction leaves unexplained, which gives realistic images:

```python
${generate}
```

WHAT YOU HAVE. weights.bin; galaxies.npy, 100 real galaxy images the network never saw in training (uint8, 100 x 256 x 256 x 3, row 0 at the top); and moments.npz, for part 2 below. You may compute anything, including running the network or any of its parts on any input: record what each head attends to, switch heads off, change the input patches and see what changes. Do not look anything up (there is no internet access in any case).

HOW YOU COMPUTE. ${protocol} Before your code runs, the code above has been run (so SHAPES, load_weights, ORDER, image_to_sequence, sequence_to_image, gelu, layer_norm, predict and generate are defined), W = load_weights() holds the weights, and GALAXIES = np.load("galaxies.npy"). A faster version of predict on the GPU (PyTorch) is up to you. ${limits} You may run code up to ${runs} times in this first part.

PART 1. Work out how this network builds a galaxy: what it uses at each step, which parts of it carry what, and how the image takes shape from the centre outward. Back your claims with experiments. When you are ready, reply WITHOUT a code block, in this format:
ACCOUNT: <how the network builds a galaxy, from the first patch to the last, in plain words, at most 400 words>
CLAIMS:
1. <a claim about the network's behaviour that could be tested by running it, in one sentence>
2. ...
(at most 20 claims)

PART 2 (after that). You will be shown ${n_moments} moments of the network at work on other galaxies, ${batch} at a time, and asked to describe each in at most 80 words of plain text without digits, so that a reader who sees only your words (and where the next patch goes) can redraw the patch the network is about to predict and say which earlier patches it relies on most.
'''

EXP2_BATCH = '''PART 2, MOMENTS ${first} TO ${last} (batch ${b} of ${nb}). moments.npz holds all the moments: seq[i] is a (256, 768) sequence whose first n[i] rows are the patches the network has at that moment (from a real galaxy it never saw; the remaining rows are zeros), and ids[i] is the moment's name. At moment i the network is about to predict patch n[i], at grid position ORDER[n[i]].

${listing}

For each of these moments, describe what the network is doing at this step: what it is looking at, what it has worked out about the galaxy, and what it is about to draw. Write plain words: at most 80 words per moment and no digits at all. A reader who sees only your description and the position of the predicted patch will try to redraw that patch and to say which earlier patches the network relies on most; your descriptions are scored by how well they manage. You may run code up to ${runs} more times for this batch. Then reply WITHOUT a code block, one line per moment, in this format:
${format}
'''

READER = '''A neural network generates images of galaxies one small square patch at a time. Each image is 256 x 256 pixels, cut into a 16 x 16 grid of patches of 16 x 16 pixels, and the network goes through the patches in a fixed order: it starts next to the centre and spirals outward, clockwise, one square ring at a time (ORDER below gives the order). Each patch is normalised on its own (mean 0, standard deviation 1 over its 16 x 16 x 3 numbers), so a patch carries only its pattern and colour balance, not its brightness. At step n the network has the patches ORDER[0], ..., ORDER[n-1] and predicts patch ORDER[n]; its prediction is its expected (average) patch, smoother than a real one.

${what}

For each moment, rebuild:
(a) the patch the network predicts, as a 16 x 16 x 3 array (rows top to bottom, columns left to right, colours red, green, blue), in any units: it is compared after normalising (mean 0, standard deviation 1), so only its pattern and colour balance matter;
(b) the earlier patches the prediction depends on most, as a list of (row, col, weight) for patches the network has already seen; the weights are normalised to sum to 1.

HOW TO ANSWER. Reply with exactly one code block that starts with ```python and ends with ```, and nothing else. It must call submit(moment, patch, deps) once for each moment, for example submit("M01", patch=my_patch, deps=[(7, 8, 0.6), (8, 8, 0.4)]). Before it runs, numpy is imported as np and ORDER (a list of (row, col)) and submit are defined. It runs once, in a fresh process with nothing else available, for at most 60 seconds; if it fails we send you the error.

MOMENTS
${listing}
'''

READER_WHAT = {
    "desc": "Someone who studied the network watched it at several moments while it worked on different galaxies, and "
            "described in a few words what it was doing at each moment. You will not see the galaxies or the network: "
            "only, for each moment, the position of the patch being predicted and that description.",
    "account": "Someone who studied the network wrote a general account of how it builds a galaxy (below). You will not "
               "see the galaxies or the network: only that account and, for each moment, the position of the patch "
               "being predicted.",
    "none": "You will not see the galaxies, the network or any description of it: only, for each moment, the position of "
            "the patch being predicted. Give your best guess."}

EXP4 = '''THE PHYSICS SCORE. The network is also used for science through its embedding of a galaxy: the residual stream x after block ${mid} of predict() (that is, after layers 0..${mid}), averaged over the 256 positions, a ${d}-number summary of the galaxy. Linear readouts (ridge regression on the standardised embedding, refitted every time on the same 2,000 galaxies) predict eight properties from it: six physical ones, redshift, stellar mass (log), specific star-formation rate (log), physical size in kpc (log), and the colours g-r and r-z, and two Galaxy Zoo morphology measures, the fraction of volunteers who classed the galaxy as featured or a disc, and the fraction who classed it as a merger. The physics score is the average R^2 of the eight readouts on other galaxies (1 = perfect, 0 = no better than always predicting the average). For the current weights, on the 500 validation galaxies: physics score ${score} (R^2: ${r2}); the network's own job, the mean next-patch Huber loss, is ${loss}.

TASK. Edit the weights to raise the physics score as much as you can. Your edit is scored on 1,000 test galaxies you never see: the score is the rise of the physics score, and it counts as zero if the edit makes the next-patch loss on those galaxies more than 2% worse.

RULES. Allowed: any computation on the weights (products, decompositions, statistics, anything you like) and any change to any tensor. NOT allowed:
1. Running the network or any of its parts on any input, real, random or constructed (no forward pass), in code, in your head or on paper. ${eval_rule}
2. Gradient descent or any training: no backward passes, autograd, optimizers, or iterative fitting of weights to a loss or to data.
3. Multiplying a whole tensor by one number (save_edit refuses this).
4. Looking anything up (there is no internet access in any case).

HOW YOU WORK. You read and change the weights by running Python code. ${protocol} Before your code runs, the code above has been run, W = load_weights() holds the original weights (read-only), and three helpers are defined: save_edit(name, array) stores a new value for one tensor (same shape as W[name]); load_edits() returns {name: array} of your current edits; drop_edit(name) removes one. Your edits are kept between runs; nothing else is. Your code runs in a folder that holds weights.bin and your edits (no network). Each run is a fresh process, may take up to 600 seconds, and its output is cut at 20,000 characters. ${level}

When you are done, reply WITHOUT a code block: say what you changed and why, in at most 200 words. The edits saved at that moment are your answer.
'''

EXP4_LEVEL = {
    "L0": "You may run code exactly once, so that one run must make and save your whole edit.",
    "L1": "You may run code up to ${runs} times.",
    "L2": "You may run code up to ${runs} times. You may also ask for up to ${evals} evaluations: reply with only the word "
          "EVALUATE (no code block) and we score your current edits on the 500 validation galaxies (physics score, the R^2 "
          "of each property, and the next-patch loss); the final score uses the test galaxies instead."}

EXP4_EVAL_RULE = {"L0": "", "L1": "",
                  "L2": "The evaluations described below are the only way to see the network run."}

JUDGE = '''You are grading an answer about a neural network against facts that were measured on that network. The answer was written by someone who only had the network's weights. Grade only what the answer states; do not reward vague or hedged statements.

THE ANSWER
<<<
${answer}
>>>

MEASURED FACTS
${facts}

Score each item below: 1 = the answer states it correctly, 0.5 = partly correct or vague, 0 = wrong, contradicted or missing. Then count CONFABULATIONS: specific, confident claims in the answer about what the network does that the measured facts contradict (for example a head said to look at the previous patch when it is measured to spread its attention evenly).

ITEMS
${items}

Reply with only a JSON object, nothing else: {"scores": {"I1": 1, "I2": 0.5, ...}, "confabulations": <integer>, "note": "<one sentence>"}
'''


def order_literal():
    return "\n".join("    " + ", ".join(f"({r}, {c})" for r, c in ORDER[i:i + 16]) + "," for i in range(0, 256, 16))


def dec(x):
    """0.001 -> '0.001', 6e-05 -> '0.00006', 1.0 -> '1.0' (6 significant digits); None -> 'unknown'."""
    if x is None:
        return "unknown"
    s = f"{float(f'{float(x):.6g}'):.12f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def count(x, unknown="an unknown number of"):
    return f"{int(x):,}" if x is not None else unknown


# The lines of the code and text that depend on the tokeniser: AIM = a two-layer MLP encoder and decoder,
# affine = a single linear layer each way.
TOKENISER = {
    "aim": dict(shapes_head='SHAPES = [("enc_fc", (${F}, 768)), ("enc_proj", (${d}, ${F})), ("pos_emb", (${B}, ${d}))]',
                shapes_tail='SHAPES += [("lnf", (${d},)), ("dec_fc", (${F}, ${d})), ("dec_proj", (768, ${F}))]',
                encode='x = gelu(seq @ W["enc_fc"].T) @ W["enc_proj"].T + W["pos_emb"][:n]   # position k adds row k of pos_emb',
                decode='return gelu(h @ W["dec_fc"].T) @ W["dec_proj"].T',
                cproj_names="enc_proj, W_o, W_proj and dec_proj", enc_dec_parts="the encoder or the decoder with its GELU",
                enc_names="enc_fc and enc_proj", pick="pick", dec_names="dec_fc and dec_proj"),
    "affine": dict(shapes_head='SHAPES = [("enc_fc", (${d}, 768)), ("pos_emb", (${B}, ${d}))]',
                   shapes_tail='SHAPES += [("lnf", (${d},)), ("dec_fc", (768, ${d}))]',
                   encode='x = seq @ W["enc_fc"].T + W["pos_emb"][:n]                         # position k adds row k of pos_emb',
                   decode='return h @ W["dec_fc"].T',
                   cproj_names="W_o and W_proj", enc_dec_parts="the encoder or the decoder",
                   enc_names="enc_fc", pick="picks", dec_names="dec_fc"),
}


def values(meta):
    """Template values from a checkpoint's meta.json (missing training details read 'unknown')."""
    a, t, m = meta["arch"], meta.get("training") or {}, meta.get("measured") or {}
    L, H, d, B = a["layers"], a["heads"], a["width"], a["block_size"]
    tok = {k: v for k, v in TOKENISER[a.get("tokeniser", "aim")].items()
           if k not in ("shapes_head", "shapes_tail", "encode", "decode")}
    return dict(tok, L=L, H=H, d=d, B=B, F=4 * d, d3=3 * d, d_1=d - 1, d2=2 * d, d2_1=2 * d - 1, d3_1=3 * d - 1,
                hd=d // H, hd_1=d // H - 1, L_1=L - 1, H_1=H - 1, mid=L // 2, total=f"{meta['numbers']:,}",
                order=order_literal(), cproj_std=f"{0.02 / (2 * L) ** 0.5:.6f}", b1=dec(t.get("beta1")),
                b2=dec(t.get("beta2")), wd=dec(t.get("weight_decay")), clip=dec(t.get("grad_clip")),
                lr=dec(t.get("learning_rate")), warmup=count(t.get("warmup_iters"), "an unknown number of"),
                min_lr=dec(t.get("min_lr")), steps=count(t.get("steps")),
                imgs=count(t.get("images_per_step"), "an unknown number of"), dtype=t.get("dtype") or "bfloat16",
                loss_model=f"{m.get('model', 0):.2f}", loss_zero=f"{m.get('zero', 0):.2f}", loss_prev=f"{m.get('prev', 0):.2f}",
                protocol=PROTOCOL, limits=LIMITS)


def sub(text, v):
    return string.Template(text).substitute(v)


def code(meta):
    text = CODE
    for k in ("shapes_head", "shapes_tail", "encode", "decode"):
        text = text.replace("${" + k + "}", TOKENISER[meta["arch"].get("tokeniser", "aim")][k])
    return string.Template(text).substitute(values(meta))


def description(meta, blind):
    """Intro, the exact code, the data, training and (unless blind) the measured losses."""
    v = values(meta)
    parts = [sub(INTRO[blind] + INTRO_TAIL, v), "```python\n" + code(meta) + "\n```", DATA,
             sub(TRAINING + TRAINING_END[blind], v)]
    if not blind:
        parts.append(sub(MEASURED, v))
    return "\n\n".join(parts) + "\n\n"


def exp1(meta, variant, runs=100):
    v = dict(values(meta), runs=runs)
    if variant == "bare":
        return sub(EXP1_BARE, v)
    return description(meta, blind=(variant == "blind")) + sub(EXP1_TAIL, v)


def exp2_study(meta, blind, runs, n_moments, batch):
    v = dict(values(meta), runs=runs, generate=GENERATE, n_moments=n_moments, batch=batch)
    return description(meta, blind) + sub(EXP2_STUDY, v)


def moment_line(m):
    r, c = m["pos"]
    return f"{m['id']}: the network has seen {m['n']} patches and is about to predict the patch at (row {r}, col {c}), in ring {m['ring']}."


def exp2_batch(batch_moments, b, nb, runs):
    listing = "\n".join(moment_line(m) for m in batch_moments)
    fmt = "\n".join(f"{m['id']}: <description>" for m in batch_moments)
    return sub(EXP2_BATCH, dict(first=batch_moments[0]["id"], last=batch_moments[-1]["id"], b=b, nb=nb, listing=listing,
                                runs=runs, format=fmt))


def reader(condition, batch_moments, texts=None, account=None):
    """condition: desc | shuffled (texts = descriptions per moment id), account (account text), none."""
    what = READER_WHAT["desc" if condition in ("desc", "shuffled") else condition]
    if condition == "account":
        what += "\n\nTHE ACCOUNT\n<<<\n" + (account or "").strip() + "\n>>>"
    lines = []
    for m in batch_moments:
        r, c = m["pos"]
        line = f"{m['id']}: the network has seen {m['n']} patches and predicts the patch at (row {r}, col {c}), in ring {m['ring']}."
        if condition in ("desc", "shuffled"):
            line += f" Description: \"{(texts or {}).get(m['id'], '').strip()}\""
        lines.append(line)
    return sub(READER, dict(what=what, listing="\n".join(lines)))


def exp4(meta, level, base_dev, blind=False, runs=50, evals=10):
    from galaxy_data import PHYSICS_NAMES
    v = dict(values(meta), runs=runs, evals=evals)
    r2 = ", ".join(f"{PHYSICS_NAMES[k]} {val:.3f}" for k, val in base_dev["r2"].items())
    v.update(score=f"{base_dev['score']:.4f}", r2=r2, loss=f"{base_dev['loss']:.4f}",
             level=sub(EXP4_LEVEL[level], v), eval_rule=EXP4_EVAL_RULE[level])
    return description(meta, blind) + sub(EXP4, v)


def judge(answer, facts, items):
    return sub(JUDGE, dict(answer=answer.strip(), facts=facts.strip(),
                           items="\n".join(f"{k}: {t}" for k, t in items)))
