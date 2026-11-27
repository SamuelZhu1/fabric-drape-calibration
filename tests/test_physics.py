"""The kernels have to agree with the energy they claim to minimise."""

import numpy as np
import warp as wp

from cloth import kernels as K
from cloth.energy import kinetic, potential
from cloth.sim import SimConfig, StripBatch, StripCase
from conftest import log_params

RIGIDITY = (120e-6, 40e-6)


def forces(batch, x_np, rigidity):
    m, d = batch.mesh, batch.device
    N = batch.B * m.n
    x = wp.array(x_np.astype(np.float32), dtype=wp.vec3, device=d)
    v = wp.zeros(N, dtype=wp.vec3, device=d)
    f = wp.zeros(N, dtype=wp.vec3, device=d)
    lg = batch.expand_params(log_params(rigidity, d))
    wp.launch(K.init_forces, N, [v, batch.mass, wp.vec3(0.0, 0.0, 0.0), 0.0, f], device=d)
    wp.launch(K.spring_forces, batch.B * len(m.springs),
              [x, batch.springs, batch.spring_rest, batch.spring_k, m.n, len(m.springs), f], device=d)
    wp.launch(K.bending_forces, batch.B * len(m.bend_triplets),
              [x, batch.triplets, batch.bend_dir, batch.bend_coeff, lg, m.n, len(m.bend_triplets), f], device=d)
    return f.numpy().astype(np.float64)


def test_forces_are_minus_energy_gradient(device):
    batch = StripBatch([StripCase(0.10, 0), StripCase(0.15, 1)], 0.2, device=device)
    rng = np.random.default_rng(3)
    x = batch.x0.numpy().astype(np.float64) + rng.normal(0, 2e-3, (batch.B * batch.mesh.n, 3))
    f = forces(batch, x, RIGIDITY)

    def internal(xx):  # drop gravity: forces() was launched without it
        mass = batch.mass.numpy().astype(np.float64)
        return potential(batch, xx, RIGIDITY) - np.sum(mass * xx[:, 1]) * 9.81

    h = 1e-6
    picks = rng.choice(len(x), 25, replace=False)
    for p in picks:
        for axis in range(3):
            xp, xm = x.copy(), x.copy()
            xp[p, axis] += h
            xm[p, axis] -= h
            grad = (internal(xp) - internal(xm)) / (2 * h)
            assert abs(f[p, axis] + grad) <= 1e-3 * max(abs(grad), 1e-3)


def test_energy_never_rises_while_settling(device):
    batch = StripBatch([StripCase(0.15, 0)], 0.2, device=device)
    lg = batch.expand_params(log_params(RIGIDITY, device))
    energies = []
    for steps in range(250, 6001, 250):
        x = batch.rollout_fast(lg, steps=steps)
        energies.append(potential(batch, x.numpy(), RIGIDITY) + kinetic(batch, batch.final_v.numpy()))
    rises = np.diff(energies)
    assert np.all(rises <= 1e-9), rises.max()


def test_pinned_particles_never_move(device):
    batch = StripBatch([StripCase(0.20, 1)], 0.2, device=device)
    x = batch.rollout_fast(batch.expand_params(log_params(RIGIDITY, device)), steps=2000).numpy()
    pinned = batch.inv_mass.numpy() == 0.0
    assert pinned.sum() == 5 * 5  # 5 cm on the table = 5 columns at or behind the edge, 5 across
    assert np.array_equal(x[pinned], batch.x0.numpy()[pinned])


def test_strip_settles(device):
    batch = StripBatch([StripCase(0.20, 0)], 0.2, config=SimConfig(), device=device)
    batch.rollout_fast(batch.expand_params(log_params((20e-6, 20e-6), device)))
    assert batch.max_speed() < 1e-3  # m/s, the "settled" bar the fitter relies on
