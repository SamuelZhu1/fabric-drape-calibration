"""Fit a fabric's bending rigidity (warp and weft) to measured cantilever curves.

The whole settle rollout is recorded on a wp.Tape, so one backward pass gives the gradient of
the centerline loss with respect to both log rigidities. Adam works on log G because rigidity
spans three orders of magnitude between a silk scarf and denim.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp
from scipy.signal import savgol_filter
from warp.optim import Adam

from cloth.mesh import make_strip
from cloth.sim import StripBatch, StripCase

DEFAULT_RIGIDITY = 20e-6  # N m, the one stiffness every fabric got before calibration


def resample_to_columns(points: np.ndarray, overhang: float, dx: float) -> np.ndarray:
    """Pick the photo point at each simulated column's arc length from the table edge.

    The strip barely stretches, so column i sits at arc length i * dx along the measured curve.
    Point noise makes a curve look longer than it is (every wiggle adds length), which shifted
    every column outward by up to 4 mm on a 20 cm overhang with 0.5 mm noise, so the curve is
    smoothed before its arc length is measured.
    """
    if len(points) >= 9:
        smooth = savgol_filter(points, 9, 3, axis=0)
        smooth[0] = points[0]
        points = smooth
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n_cols = int(round(overhang / dx)) + 1
    want = np.arange(n_cols) * dx
    want = np.minimum(want, s[-1])
    return np.stack([np.interp(want, s, points[:, 0]), np.interp(want, s, points[:, 1])], axis=1)


def loss_and_grad(batch: StripBatch, params: wp.array):
    tape = wp.Tape()
    with tape:
        log_g = batch.expand_params(params)
        x = batch.rollout_differentiable(log_g)
        loss = batch.loss(x)
    tape.backward(loss)
    value = float(loss.numpy()[0])
    grad = params.grad.numpy().copy()
    tape.zero()
    return value, grad, tape


def rms_mm(batch: StripBatch, log_params: np.ndarray) -> float:
    """RMS centerline distance to the targets over every hanging column, mm."""
    x = batch.rollout_fast(batch.expand_params(wp.array(log_params.astype(np.float32), dtype=float, device=batch.device)))
    errs = []
    for c, cl in zip(batch.cases, batch.centerlines(x)):
        errs.append(np.sum((cl - c.target_xy) ** 2, axis=1))
    return 1000.0 * math.sqrt(np.concatenate(errs).mean())


def fit(
    batch: StripBatch,
    init_rigidity=(DEFAULT_RIGIDITY, DEFAULT_RIGIDITY),
    iters: int = 150,
    lr: float = 0.15,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """Adam on (log G_warp, log G_weft). Stops when both steps fall below tol in log space."""
    batch.loss_setup()
    params = wp.array(np.log(init_rigidity).astype(np.float32), dtype=float, device=batch.device, requires_grad=True)
    opt = Adam([params], lr=lr)
    history = []
    best = (math.inf, params.numpy().copy())
    t0 = time.time()
    for it in range(iters):
        before = params.numpy().copy()
        value, grad, tape = loss_and_grad(batch, params)
        if not math.isfinite(value) or not np.all(np.isfinite(grad)):
            # Too big a step made the rollout blow up: go back to the best point, smaller steps.
            params.assign(best[1].astype(np.float32))
            opt = Adam([params], lr=opt.lr * 0.5)
            continue
        if value < best[0]:
            best = (value, before)
        # Directions with no strips cut that way get no gradient; leave them alone.
        opt.step([wp.array(grad.astype(np.float32), dtype=float, device=batch.device)])
        history.append({"iter": it, "loss": value, "rigidity": np.exp(before).tolist()})
        if verbose:
            print(f"{it:4d} loss {value:.3e} G {np.exp(before)} grad {grad}")
        if np.max(np.abs(params.numpy() - before)) < tol:
            break
    final = params.numpy().astype(np.float64)
    return {
        "rigidity": np.exp(final).tolist(),
        "log_rigidity": final.tolist(),
        "iterations": len(history),
        "seconds": time.time() - t0,
        "history": history,
    }


def load_fabric(name: str, curves_dir: Path, fabrics_file: Path, overhangs=None):
    """Read fabrics.yaml and every curves/<name>_<dir>_<overhang>.json for one fabric."""
    import yaml

    spec = yaml.safe_load(fabrics_file.read_text())[name]
    rho = spec["areal_density_g_m2"] / 1000.0
    cases = []
    for path in sorted(curves_dir.glob(f"{name}_*.json")):
        d = json.loads(path.read_text())
        o = d["overhang_cm"] / 100.0
        if overhangs is not None and round(o * 1000) not in {round(v * 1000) for v in overhangs}:
            continue
        pts = np.asarray(d["points_cm"], dtype=float) / 100.0
        cases.append(StripCase(o, 0 if d["direction"] == "warp" else 1, pts))
    return rho, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fabric", required=True)
    ap.add_argument("--curves", type=Path, default=Path("curves"))
    ap.add_argument("--fabrics", type=Path, default=Path("fabrics.yaml"))
    ap.add_argument("--fit-overhangs", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("params"))
    args = ap.parse_args()

    rho, cases = load_fabric(args.fabric, args.curves, args.fabrics, args.fit_overhangs)
    dx = make_strip().dx
    for c in cases:
        c.target_xy = resample_to_columns(c.target_xy, c.overhang, dx)
    batch = StripBatch(cases, rho, device=args.device)
    batch.loss_setup()
    before = rms_mm(batch, np.log([DEFAULT_RIGIDITY, DEFAULT_RIGIDITY]))
    result = fit(batch, iters=args.iters, verbose=True)
    after = rms_mm(batch, np.array(result["log_rigidity"]))

    # Overhangs the fit never saw: the honest error to quote.
    _, every = load_fabric(args.fabric, args.curves, args.fabrics)
    fit_mm = {round(o * 1000) for o in args.fit_overhangs}
    held = [c for c in every if round(c.overhang * 1000) not in fit_mm]
    held_out = None
    if held:
        for c in held:
            c.target_xy = resample_to_columns(c.target_xy, c.overhang, dx)
        hb = StripBatch(held, rho, device=args.device)
        held_out = {
            "overhangs_m": sorted({c.overhang for c in held}),
            "rms_mm_default": rms_mm(hb, np.log([DEFAULT_RIGIDITY, DEFAULT_RIGIDITY])),
            "rms_mm_fit": rms_mm(hb, np.array(result["log_rigidity"])),
        }
    args.out.mkdir(exist_ok=True)
    out = {
        "fabric": args.fabric,
        "flexural_rigidity_uNm_fit": {"warp": result["rigidity"][0] * 1e6, "weft": result["rigidity"][1] * 1e6},
        "rms_mm_default": before,
        "rms_mm_fit": after,
        "iterations": result["iterations"],
        "fit_overhangs_m": args.fit_overhangs,
        "held_out": held_out,
    }
    (args.out / f"{args.fabric}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
