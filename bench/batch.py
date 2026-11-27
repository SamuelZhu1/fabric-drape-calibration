"""Time batched strip rollouts on CPU and GPU.

Every strip in a batch is an independent cantilever, so a batch of B strips is one launch of
B times as many threads. One strip has 105 particles, far too few to fill a GPU, so the GPU only
pays off once many strips share each launch.

    python -m bench.batch --device cpu cuda:0 --batch 1 64 256 1024 --steps 500
"""

import argparse
import time

import numpy as np
import warp as wp

from cloth.sim import StripBatch, StripCase


def time_rollout(device: str, batch: int, steps: int, repeats: int = 3) -> float:
    """Best of `repeats` wall-clock seconds per simulated step."""
    cases = [StripCase(0.05 + 0.0125 * (b % 13), b % 2) for b in range(batch)]
    sb = StripBatch(cases, areal_density=0.2, device=device)
    log_g = sb.expand_params(wp.array(np.log([100e-6, 50e-6]).astype(np.float32), dtype=float, device=device))
    sb.rollout_fast(log_g, steps=10)  # compile and warm up
    wp.synchronize_device(device)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        sb.rollout_fast(log_g, steps=steps)
        wp.synchronize_device(device)
        best = min(best, (time.perf_counter() - t) / steps)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", nargs="+", default=["cpu"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 64, 256, 1024])
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    wp.init()
    rows = {}
    print(f"{'device':<8} {'batch':>6} {'ms/step':>9} {'strip-steps/s':>14}")
    for dev in args.device:
        for b in args.batch:
            s = time_rollout(dev, b, args.steps)
            rows[(dev, b)] = s
            print(f"{dev:<8} {b:>6} {1000 * s:>9.3f} {b / s:>14.0f}")
    devs = args.device
    if len(devs) > 1:
        for b in args.batch:
            print(f"batch {b}: {devs[1]} is {rows[(devs[0], b)] / rows[(devs[1], b)]:.1f}x {devs[0]}")


if __name__ == "__main__":
    main()
