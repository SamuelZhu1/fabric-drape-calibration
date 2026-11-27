"""Total energy of a batch in NumPy (float64), used by the tests to check the Warp kernels."""

import numpy as np

from cloth.sim import GRAVITY, StripBatch


def potential(batch: StripBatch, x: np.ndarray, rigidity) -> float:
    """Spring + bending + gravity energy, J. x is (B*n, 3); rigidity is (warp, weft) in N m."""
    m = batch.mesh
    x = x.reshape(batch.B, m.n, 3).astype(np.float64)
    k = batch.spring_k.numpy().astype(np.float64)
    rest = batch.spring_rest.numpy().astype(np.float64)
    coeff = batch.bend_coeff.numpy().reshape(batch.B, -1).astype(np.float64)
    mass = batch.mass.numpy().reshape(batch.B, m.n).astype(np.float64)
    idx = batch.param_index.numpy().reshape(batch.B, 2)
    g = np.asarray(GRAVITY)
    e = 0.0
    for b in range(batch.B):
        p = x[b]
        d = p[m.springs[:, 1]] - p[m.springs[:, 0]]
        e += 0.5 * np.sum(k * (np.linalg.norm(d, axis=1) - rest) ** 2)
        t = m.bend_triplets
        u = p[t[:, 0]] - 2 * p[t[:, 1]] + p[t[:, 2]]
        gb = np.asarray(rigidity, dtype=np.float64)[idx[b][m.bend_dir]]
        e += 0.5 * np.sum(gb * coeff[b] * np.sum(u * u, axis=1))
        e -= np.sum(mass[b] * (p @ g))
    return e


def kinetic(batch: StripBatch, v: np.ndarray) -> float:
    mass = batch.mass.numpy().astype(np.float64)
    return 0.5 * float(np.sum(mass * np.sum(v.astype(np.float64) ** 2, axis=1)))
