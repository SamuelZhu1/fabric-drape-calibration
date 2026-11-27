"""Warp kernels for the mass-spring strip.

Every kernel works on a batch of B identical strips laid out back to back: particle p of strip
b lives at index b * n + p. The same kernels compile to C++ on the CPU and to CUDA on a GPU.
"""

import warp as wp


@wp.kernel
def init_forces(
    v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    gravity: wp.vec3,
    damping: float,
    f: wp.array(dtype=wp.vec3),
):
    """f = m g - c m v. Mass-proportional damping only removes energy."""
    tid = wp.tid()
    m = mass[tid]
    f[tid] = m * gravity - damping * m * v[tid]


@wp.kernel
def spring_forces(
    x: wp.array(dtype=wp.vec3),
    springs: wp.array(dtype=wp.vec2i),
    rest: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    n_per_strip: int,
    springs_per_strip: int,
    f: wp.array(dtype=wp.vec3),
):
    """Linear springs on stretch and shear edges: E = k/2 (|d| - L)^2."""
    tid = wp.tid()
    b = tid / springs_per_strip
    s = tid - b * springs_per_strip
    off = b * n_per_strip
    i = springs[s][0] + off
    j = springs[s][1] + off
    d = x[j] - x[i]
    length = wp.length(d)
    if length < 1.0e-9:
        return
    fs = stiffness[s] * (length - rest[s]) * (d / length)
    wp.atomic_add(f, i, fs)
    wp.atomic_sub(f, j, fs)


@wp.kernel
def bending_forces(
    x: wp.array(dtype=wp.vec3),
    triplets: wp.array(dtype=wp.vec3i),
    bend_dir: wp.array(dtype=int),
    coeff: wp.array(dtype=float),
    log_rigidity: wp.array(dtype=float),
    n_per_strip: int,
    triplets_per_strip: int,
    f: wp.array(dtype=wp.vec3),
):
    """Discrete bending energy E = (G w / 2 h^3) |x_a - 2 x_b + x_c|^2.

    With curvature kappa ~ |x_a - 2 x_b + x_c| / h^2 this is the elastica energy G w kappa^2 / 2
    integrated over one spacing h, where G is the flexural rigidity per unit width (N m) and w
    the width this row stands for. coeff = w / h^3 is precomputed per strip and triplet. The rigidity
    is passed as log(G) per strip and direction, which is what the optimizer works in.
    """
    tid = wp.tid()
    b = tid / triplets_per_strip
    t = tid - b * triplets_per_strip
    off = b * n_per_strip
    a = triplets[t][0] + off
    c = triplets[t][1] + off
    e = triplets[t][2] + off
    g = wp.exp(log_rigidity[b * 2 + bend_dir[t]])
    u = x[a] - 2.0 * x[c] + x[e]
    k = g * coeff[tid]
    wp.atomic_sub(f, a, k * u)
    wp.atomic_add(f, c, 2.0 * k * u)
    wp.atomic_sub(f, e, k * u)


@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    dt: float,
    x_out: wp.array(dtype=wp.vec3),
    v_out: wp.array(dtype=wp.vec3),
):
    """Semi-implicit Euler: update velocity first, then move with the new velocity.

    Pinned particles have inv_mass 0, so they never move.
    """
    tid = wp.tid()
    v1 = v[tid] + f[tid] * inv_mass[tid] * dt
    v_out[tid] = v1
    x_out[tid] = x[tid] + v1 * dt
