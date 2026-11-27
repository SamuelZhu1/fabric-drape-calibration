"""Batched cantilever strip simulation on NVIDIA Warp.

Each strip in a batch rests on a table that ends at x = 0 and hangs `overhang` metres past the
edge. Particles on the table (x <= 0) are pinned, which clamps the strip flat at the edge the way
the table does in a cantilever bending test. The hanging part sags under gravity until the
damping settles it, and the settled shape depends on the flexural rigidity G.
"""

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from cloth import kernels as K
from cloth.mesh import SHEAR, STRETCH, StripMesh, make_strip

GRAVITY = (0.0, -9.81, 0.0)


@wp.kernel
def gather_log_rigidity(
    params: wp.array(dtype=float),
    index: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    """Map the shared fit parameters onto each strip's (lengthwise, crosswise) slots."""
    tid = wp.tid()
    out[tid] = params[index[tid]]


@dataclass
class SimConfig:
    dt: float = 2.5e-4  # s; stable for the stiffest springs on the lightest fabric we fit
    settle_time: float = 1.5  # s of simulated time before the shape is read
    damping: float = 12.0  # 1/s, mass-proportional
    k_stretch: float = 20.0  # N/m per stretch edge
    k_shear: float = 5.0  # N/m per shear edge


@dataclass
class StripCase:
    """One photographed configuration: which way the strip was cut and how far it hangs."""

    overhang: float  # m
    direction: int  # 0 = cut along the warp, 1 = cut along the weft
    target_xy: np.ndarray = field(default=None)  # (cols, 2) measured centerline points, m


class StripBatch:
    """B strips of one fabric, each with its own overhang and cut direction."""

    def __init__(
        self,
        cases: list,
        areal_density: float,
        mesh: StripMesh = None,
        config: SimConfig = None,
        device: str = None,
    ):
        self.mesh = mesh or make_strip()
        self.cfg = config or SimConfig()
        self.device = device or wp.get_preferred_device()
        self.cases = cases
        self.B = len(cases)
        m = self.mesh
        n = m.n
        dx = m.dx

        for c in cases:
            steps = c.overhang / dx
            if abs(steps - round(steps)) > 1e-6:
                raise ValueError(f"overhang {c.overhang} m must be a multiple of {dx} m")
            if c.overhang >= m.length - dx:
                raise ValueError("at least two columns must stay on the table")

        # Particle masses from areal density and each particle's share of the area.
        wx = np.full(m.nx, m.dx)
        wx[[0, -1]] *= 0.5
        wz = np.full(m.nz, m.dz)
        wz[[0, -1]] *= 0.5
        area = np.outer(wx, wz).reshape(-1)
        mass1 = (areal_density * area).astype(np.float32)

        x0, inv_mass, cols = [], [], []
        self.hang_cols = []
        for c in cases:
            shift = m.length - c.overhang
            p = m.rest_positions.copy()
            p[:, 0] -= shift
            x0.append(p)
            pinned = p[:, 0] <= 1e-6
            im = np.where(pinned, 0.0, 1.0 / mass1).astype(np.float32)
            inv_mass.append(im)
            first = int(round(shift / dx))
            hang = list(range(first, m.nx))
            self.hang_cols.append(hang)
            cols.extend(hang)
        self.cols_per_strip = None  # strips differ, so columns are listed flat
        self.col_index = np.array(cols, dtype=np.int32)
        self.col_strip = np.concatenate(
            [np.full(len(h), b, dtype=np.int32) for b, h in enumerate(self.hang_cols)]
        )

        d = self.device
        self.x0 = wp.array(np.concatenate(x0), dtype=wp.vec3, device=d)
        self.v0 = wp.zeros(self.B * n, dtype=wp.vec3, device=d)
        self.mass = wp.array(np.tile(mass1, self.B), dtype=float, device=d)
        self.inv_mass = wp.array(np.concatenate(inv_mass), dtype=float, device=d)
        self.springs = wp.array(m.springs, dtype=wp.vec2i, device=d)
        self.spring_rest = wp.array(m.spring_rest, dtype=float, device=d)
        k = np.where(m.spring_type == STRETCH, self.cfg.k_stretch, 0.0)
        k = np.where(m.spring_type == SHEAR, self.cfg.k_shear, k).astype(np.float32)
        self.spring_k = wp.array(k, dtype=float, device=d)

        # Width (or length) each bending triplet stands for, divided by h^3, per strip.
        share = np.empty(len(m.bend_triplets), dtype=np.float32)
        for t, (a, b_, c) in enumerate(m.bend_triplets):
            i, kz = divmod(int(b_), m.nz)
            share[t] = wz[kz] if m.bend_dir[t] == 0 else wx[i]
        coeff = []
        for b, c in enumerate(cases):
            cb = share / m.bend_spacing**3
            # The triplet centred on the last pinned column (the table edge) gets twice the
            # weight. Pinning the column behind the edge only fixes the slope half a spacing
            # back, which moves the clamp to x = -h/2 and makes the strip sag as if it were
            # h/2 longer. Doubling this one term matches a mirrored ghost node, the standard
            # second-order clamp, and removes that first-order error.
            edge_col = int(round((m.length - c.overhang) / dx))
            centre_col = m.bend_triplets[:, 1] // m.nz
            cb = np.where((m.bend_dir == 0) & (centre_col == edge_col), 2.0 * cb, cb)
            coeff.append(cb)
        self.bend_coeff = wp.array(np.concatenate(coeff).astype(np.float32), dtype=float, device=d)
        self.triplets = wp.array(m.bend_triplets, dtype=wp.vec3i, device=d)
        self.bend_dir = wp.array(m.bend_dir, dtype=int, device=d)

        # Strip b slot 0 is bending along its length (its cut direction), slot 1 across it.
        idx = []
        for c in cases:
            idx += [c.direction, 1 - c.direction]
        self.param_index = wp.array(np.array(idx, dtype=np.int32), dtype=int, device=d)

    # ------------------------------------------------------------------ simulation

    @property
    def steps(self) -> int:
        return int(round(self.cfg.settle_time / self.cfg.dt))

    def _step(self, x, v, f, log_g, x_out, v_out):
        m, cfg, d = self.mesh, self.cfg, self.device
        N = self.B * m.n
        wp.launch(K.init_forces, N, [v, self.mass, wp.vec3(*GRAVITY), cfg.damping, f], device=d)
        wp.launch(
            K.spring_forces,
            self.B * len(m.springs),
            [x, self.springs, self.spring_rest, self.spring_k, m.n, len(m.springs), f],
            device=d,
        )
        wp.launch(
            K.bending_forces,
            self.B * len(m.bend_triplets),
            [x, self.triplets, self.bend_dir, self.bend_coeff, log_g, m.n, len(m.bend_triplets), f],
            device=d,
        )
        wp.launch(K.integrate, N, [x, v, f, self.inv_mass, cfg.dt, x_out, v_out], device=d)

    def expand_params(self, log_params: wp.array) -> wp.array:
        """(2,) shared log G (warp, weft) -> (B*2,) per-strip log G, recorded on any open tape."""
        out = wp.zeros(self.B * 2, dtype=float, device=self.device, requires_grad=True)
        wp.launch(gather_log_rigidity, self.B * 2, [log_params, self.param_index, out], device=self.device)
        return out

    def rollout_differentiable(self, log_g: wp.array) -> wp.array:
        """Run to settle, keeping every intermediate state so a wp.Tape can differentiate it."""
        N = self.B * self.mesh.n
        d = self.device
        x, v = self.x0, self.v0
        for _ in range(self.steps):
            f = wp.zeros(N, dtype=wp.vec3, device=d, requires_grad=True)
            x1 = wp.zeros(N, dtype=wp.vec3, device=d, requires_grad=True)
            v1 = wp.zeros(N, dtype=wp.vec3, device=d, requires_grad=True)
            self._step(x, v, f, log_g, x1, v1)
            x, v = x1, v1
        self.final_v = v
        return x

    def rollout_fast(self, log_g: wp.array, steps: int = None) -> wp.array:
        """Same physics with two ping-pong buffers and no gradient bookkeeping."""
        N = self.B * self.mesh.n
        d = self.device
        xa = wp.clone(self.x0)
        va = wp.clone(self.v0)
        xb = wp.zeros_like(xa)
        vb = wp.zeros_like(va)
        f = wp.zeros(N, dtype=wp.vec3, device=d)
        for _ in range(steps or self.steps):
            self._step(xa, va, f, log_g, xb, vb)
            xa, xb = xb, xa
            va, vb = vb, va
        self.final_v = va
        return xa

    # ------------------------------------------------------------------ measurement

    def centerlines(self, x: wp.array) -> list:
        """Per strip, the (x, y) mid point of each hanging column, table edge first, in metres."""
        p = x.numpy().reshape(self.B, self.mesh.nx, self.mesh.nz, 3)
        mid = p.mean(axis=2)[:, :, :2]
        return [mid[b, cols] for b, cols in enumerate(self.hang_cols)]

    def max_speed(self) -> float:
        return float(np.linalg.norm(self.final_v.numpy(), axis=1).max())

    def loss_setup(self):
        """Flatten every strip's target points for the loss kernel."""
        targets = []
        for c, cols in zip(self.cases, self.hang_cols):
            if c.target_xy is None or len(c.target_xy) != len(cols):
                raise ValueError("each case needs one target point per hanging column")
            targets.append(c.target_xy)
        d = self.device
        self._targets = wp.array(np.concatenate(targets).astype(np.float32), dtype=wp.vec2, device=d)
        self._cols = wp.array(self.col_index, dtype=int, device=d)
        self._strip_arr = wp.array(self.col_strip, dtype=int, device=d)
        self._n_cols = len(self.col_index)

    def loss(self, x: wp.array) -> wp.array:
        """Mean squared centerline distance over all strips, m^2.

        Two kernels instead of one: a loop that reassigns a variable over a runtime range is not
        differentiable in Warp, so the column means are gathered with atomics first.
        """
        d = self.device
        mid = wp.zeros(self._n_cols, dtype=wp.vec3, device=d, requires_grad=True)
        wp.launch(
            column_means,
            self._n_cols * self.mesh.nz,
            [x, self._cols, self._strip_arr, self.mesh.nz, self.mesh.n, mid],
            device=d,
        )
        out = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        wp.launch(centerline_sq_error, self._n_cols, [mid, self._targets, 1.0 / self._n_cols, out], device=d)
        return out


@wp.kernel
def column_means(
    x: wp.array(dtype=wp.vec3),
    col: wp.array(dtype=int),
    strip: wp.array(dtype=int),
    nz: int,
    n_per_strip: int,
    mid: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    c = tid / nz
    k = tid - c * nz
    p = strip[c] * n_per_strip + col[c] * nz + k
    wp.atomic_add(mid, c, x[p] / float(nz))


@wp.kernel
def centerline_sq_error(
    mid: wp.array(dtype=wp.vec3),
    target: wp.array(dtype=wp.vec2),
    weight: float,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    m = mid[tid]
    d = wp.vec2(m[0], m[1]) - target[tid]
    wp.atomic_add(loss, 0, weight * wp.dot(d, d))
