"""Topology for a rectangular fabric strip lying along +x with its width along z.

The strip is a grid of nx particles along its length and nz across its width. Particle (i, k)
has index i * nz + k, so column i is one slice across the width at distance i * dx from the
back of the strip.
"""

from dataclasses import dataclass

import numpy as np

# Spring types, stored per spring so the stiffness kernel can look them up.
STRETCH = 0
SHEAR = 1


@dataclass
class StripMesh:
    nx: int
    nz: int
    length: float  # m
    width: float  # m
    rest_positions: np.ndarray  # (n, 3)
    springs: np.ndarray  # (s, 2) particle pairs
    spring_rest: np.ndarray  # (s,)
    spring_type: np.ndarray  # (s,)
    bend_triplets: np.ndarray  # (t, 3) particles (i-1, i, i+1) along one row or column
    bend_dir: np.ndarray  # (t,) 0 = bending along the length, 1 = across the width
    bend_spacing: np.ndarray  # (t,) rest spacing of the triplet, m

    @property
    def n(self) -> int:
        return self.nx * self.nz

    @property
    def dx(self) -> float:
        return self.length / (self.nx - 1)

    @property
    def dz(self) -> float:
        return self.width / (self.nz - 1)

    def index(self, i: int, k: int) -> int:
        return i * self.nz + k


def make_strip(length: float = 0.25, width: float = 0.05, nx: int = 21, nz: int = 5) -> StripMesh:
    """Build a flat strip in the y = 0 plane, x in [0, length], z in [-width/2, width/2]."""
    xs = np.linspace(0.0, length, nx)
    zs = np.linspace(-width / 2, width / 2, nz)
    pos = np.array([[x, 0.0, z] for x in xs for z in zs], dtype=np.float32)

    def idx(i, k):
        return i * nz + k

    springs, rest, kind = [], [], []

    def add(a, b, t):
        springs.append((a, b))
        rest.append(float(np.linalg.norm(pos[a] - pos[b])))
        kind.append(t)

    for i in range(nx):
        for k in range(nz):
            if i + 1 < nx:
                add(idx(i, k), idx(i + 1, k), STRETCH)
            if k + 1 < nz:
                add(idx(i, k), idx(i, k + 1), STRETCH)
            if i + 1 < nx and k + 1 < nz:
                add(idx(i, k), idx(i + 1, k + 1), SHEAR)
                add(idx(i + 1, k), idx(i, k + 1), SHEAR)

    trip, tdir, tspace = [], [], []
    dx, dz = length / (nx - 1), width / (nz - 1)
    for k in range(nz):
        for i in range(1, nx - 1):
            trip.append((idx(i - 1, k), idx(i, k), idx(i + 1, k)))
            tdir.append(0)
            tspace.append(dx)
    for i in range(nx):
        for k in range(1, nz - 1):
            trip.append((idx(i, k - 1), idx(i, k), idx(i, k + 1)))
            tdir.append(1)
            tspace.append(dz)

    return StripMesh(
        nx=nx,
        nz=nz,
        length=length,
        width=width,
        rest_positions=pos,
        springs=np.array(springs, dtype=np.int32),
        spring_rest=np.array(rest, dtype=np.float32),
        spring_type=np.array(kind, dtype=np.int32),
        bend_triplets=np.array(trip, dtype=np.int32),
        bend_dir=np.array(tdir, dtype=np.int32),
        bend_spacing=np.array(tspace, dtype=np.float32),
    )
