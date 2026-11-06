"""The patch order and layout of AstroPT images (numpy only, so it loads anywhere fast)."""

import numpy as np

PATCH, GRID, NPATCH, PATCH_DIM = 16, 16, 256, 768


def spiral_order():
    """ORDER[k] = (row, col) of the k-th patch, exactly as astropt's GalaxyImageDataset._spiral + spiralise build it
    (the raster axes are swapped there because images are read as (colour, column, row))."""
    n = GRID
    a = np.arange(n * n)
    b = a.reshape((n, n))
    m = None
    for _ in range(n, 0, -2):
        m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
        b = b[1:-1, 1:-1]
    a[list(m[1:])] = list(a)
    sp = abs(a - n * n + 1).reshape((n, n))
    order = [None] * (n * n)
    for h in range(n):
        for w in range(n):
            order[sp[h, w]] = (int(w), int(h))
    return order


ORDER = spiral_order()
ORDER_IDX = np.array([r * GRID + c for r, c in ORDER])  # row-major grid index of the k-th patch
RING = np.array([max(abs(2 * r - 15), abs(2 * c - 15)) // 2 + 1 for r, c in ORDER])  # ring 1 (centre) .. 8 (edge)
POSITION = {rc: k for k, rc in enumerate(ORDER)}  # (row, col) -> position in the sequence


def patch_to_vector(patch):
    """(16 rows, 16 columns, 3) patch in image orientation -> its 768 numbers in sequence layout (not normalised)."""
    return np.asarray(patch, dtype=np.float64).transpose(1, 0, 2).reshape(PATCH_DIM)


def vector_to_patch(v):
    return np.asarray(v, dtype=np.float64).reshape(PATCH, PATCH, 3).transpose(1, 0, 2)
