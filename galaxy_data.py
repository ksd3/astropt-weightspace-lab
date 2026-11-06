"""A fixed local set of held-out galaxies (the Hugging Face test split of Smith42/galaxies v2.0, never used in
training), downloaded once, with physical labels. Every experiment and every checkpoint uses the same galaxies.

  $ASTROPT_PY galaxy_data.py     (builds data/galaxies/, about 5 minutes)

data/galaxies/
  images.npy      (N, 256, 256, 3) uint8, row 0 at the top
  labels.json     one dict per galaxy: id, catalogue columns, and the derived physics targets
  splits.json     index lists:
                    verify       32  the first 32 test galaxies (the losses quoted in the full prompt)
                    attn         64  measuring what each head attends to
                    study       100  given to the experiment-2 explainers
                    moments      10  the experiment-2 moments (5 with a disc or features, 5 others)
                    probe_train 2000 fitting the physics readouts
                    dev         500  experiment-4 evaluations the agents may request
                    test       1000  experiment-4 final scores
  study_images.npy, moments.npz   files handed to the experiment-2 explainers
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/galaxies"
SIZES = {"verify": 32, "attn": 64, "study": 100, "moments": 10, "probe_train": 2000, "dev": 500, "test": 1000}
PHYSICAL = ["redshift", "log_mass", "log_ssfr", "log_size_kpc", "g_r", "r_z"]
MORPHOLOGY = {"featured": "smooth-or-featured_featured-or-disk_fraction", "merger": "merging_merger_fraction"}
PHYSICS = PHYSICAL + list(MORPHOLOGY)  # the physics score's eight properties: six physical, two Galaxy Zoo morphology
PHYSICS_NAMES = {"redshift": "redshift", "log_mass": "stellar mass (log)", "log_ssfr": "specific star-formation rate (log)",
                 "log_size_kpc": "physical size in kpc (log)", "g_r": "colour g-r", "r_z": "colour r-z",
                 "featured": "featured-or-disc vote fraction", "merger": "merger vote fraction"}


def target(row, key):
    """A galaxy's value of one physics-score property (NaN if it has none)."""
    v = row["physics"][key] if key in PHYSICAL else num(row.get(MORPHOLOGY[key]))
    return float("nan") if v is None else v
def moment_steps(g):
    """Patches already seen at galaxy g's four moments (the network then predicts patch ORDER[n]): one in each of
    rings 2, 4, 6 and 8 (inner, middle, outer, edge), at a different place round the ring for each galaxy."""
    return [12 + g % 4, 40 + 2 * g, 100 + 4 * g, 200 + 5 * g]
KEEP = ["dr8_id", "ra", "dec", "photo_z", "mass_med_photoz", "ssfr_med_photoz", "est_petro_th50_kpc", "mag_g_desi",
        "mag_r_desi", "mag_z_desi", "galaxy_size", "smooth-or-featured_featured-or-disk_fraction",
        "merging_merger_fraction", "disk-edge-on_yes_fraction", "has-spiral-arms_yes_fraction", "bar_strong_fraction"]


def num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def physics_targets(row):
    z, m, s, size = num(row.get("photo_z")), num(row.get("mass_med_photoz")), num(row.get("ssfr_med_photoz")), \
        num(row.get("est_petro_th50_kpc"))
    g, r, zz = num(row.get("mag_g_desi")), num(row.get("mag_r_desi")), num(row.get("mag_z_desi"))
    ok = (z is not None and 0 < z < 1.5 and m is not None and 6 < m < 13 and s is not None and -16 < s < -6
          and size is not None and size > 0 and None not in (g, r, zz) and max(g, r, zz) < 30)
    if not ok:
        return None
    return {"redshift": z, "log_mass": m, "log_ssfr": s, "log_size_kpc": math.log10(size), "g_r": g - r, "r_z": r - zz}


def build():
    from datasets import load_dataset
    sys.path.insert(0, HERE)
    import astro_model as am

    os.makedirs(OUT, exist_ok=True)
    total = sum(SIZES.values())
    images = np.lib.format.open_memmap(f"{OUT}/images.npy.tmp", mode="w+", dtype=np.uint8, shape=(total, 256, 256, 3))
    labels, splits = [], {k: [] for k in SIZES}
    featured_quota = SIZES["moments"] // 2
    ds = load_dataset("Smith42/galaxies", split="test", revision="v2.0", streaming=True)
    seen = 0
    for ex in ds:
        seen += 1
        need = {k: SIZES[k] - len(v) for k, v in splits.items()}
        if sum(need.values()) == 0:
            break
        img = np.array(ex["image"])
        if img.shape != (256, 256, 3):
            continue
        row = {k: ex.get(k) for k in KEEP}
        phys = physics_targets(row)
        feat = num(row.get("smooth-or-featured_featured-or-disk_fraction")) or 0.0
        if need["verify"]:
            split = "verify"
        elif need["attn"]:
            split = "attn"
        elif need["study"]:
            split = "study"
        elif need["moments"] and (feat >= 0.5) == (len([1 for i in splits["moments"] if labels[i]["featured"] >= 0.5]) < featured_quota):
            split = "moments"
        elif phys is not None and need["probe_train"]:
            split = "probe_train"
        elif phys is not None and need["dev"]:
            split = "dev"
        elif phys is not None and need["test"]:
            split = "test"
        else:
            continue
        i = len(labels)
        images[i] = img
        labels.append({**row, "featured": feat, "merger": num(row.get("merging_merger_fraction")), "physics": phys,
                       "split": split})
        splits[split].append(i)
        if len(labels) % 500 == 0:
            print(f"  {len(labels)} of {total} galaxies kept ({seen} read)", flush=True)
    assert all(len(v) == SIZES[k] for k, v in splits.items()), {k: len(v) for k, v in splits.items()}
    images.flush()
    del images
    os.replace(f"{OUT}/images.npy.tmp", f"{OUT}/images.npy")
    with open(f"{OUT}/labels.json", "w") as f:
        json.dump(labels, f)
    with open(f"{OUT}/splits.json", "w") as f:
        json.dump(splits, f)

    imgs = np.load(f"{OUT}/images.npy", mmap_mode="r")
    np.save(f"{OUT}/study_images.npy", np.ascontiguousarray(imgs[splits["study"]]))
    build_moments()
    print(f"done: {len(labels)} galaxies from {seen} read -> {OUT}", flush=True)


def build_moments():
    """moments.npz for the experiment-2 explainers: for each moment only the patches seen so far (the rest zeros)."""
    import astro_model as am
    imgs, labels, splits = load()
    seqs = am.images_to_sequences(imgs[splits["moments"]])
    mom_seq, mom_n, mom_gal = [], [], []
    for g in range(len(seqs)):
        for n in moment_steps(g):
            s = np.zeros_like(seqs[g])
            s[:n] = seqs[g][:n]
            mom_seq.append(s), mom_n.append(n), mom_gal.append(g)
    ids = [f"M{i + 1:02d}" for i in range(len(mom_n))]
    np.savez(f"{OUT}/moments.npz.tmp.npz", seq=np.stack(mom_seq), n=np.array(mom_n), galaxy=np.array(mom_gal),
             ids=np.array(ids))
    os.replace(f"{OUT}/moments.npz.tmp.npz", f"{OUT}/moments.npz")
    print(f"{len(ids)} moments -> {OUT}/moments.npz", flush=True)


def load():
    """(images memmap, labels, splits)."""
    with open(f"{OUT}/labels.json") as f:
        labels = json.load(f)
    with open(f"{OUT}/splits.json") as f:
        splits = json.load(f)
    return np.load(f"{OUT}/images.npy", mmap_mode="r"), labels, splits


def moments(with_seq=True):
    """The experiment-2 moments: [{id, n, galaxy, pos, ring}], and their (40, 256, 768) prefixes."""
    from spiral import ORDER, RING
    m = np.load(f"{OUT}/moments.npz")
    info = [{"id": str(i), "n": int(n), "galaxy": int(g), "pos": ORDER[int(n)], "ring": int(RING[int(n)])}
            for i, n, g in zip(m["ids"], m["n"], m["galaxy"])]
    return (info, m["seq"]) if with_seq else info


def ready():
    return all(os.path.exists(f"{OUT}/{f}") for f in ("images.npy", "labels.json", "splits.json", "study_images.npy",
                                                      "moments.npz"))


if __name__ == "__main__":
    if ready():
        print("already built:", OUT)
    else:
        build()
    sys.stdout.flush()
    os._exit(0)  # the HF streaming iterator crashes at interpreter teardown otherwise
