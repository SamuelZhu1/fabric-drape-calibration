"""Check a fitted rigidity against Peirce's cantilever formula (the ASTM D1388 method).

ASTM D1388 slides the strip off the table until the line from the edge to the tip sits 41.5
degrees below horizontal. At that angle Peirce's bending length is half the overhang, and the
flexural rigidity is G = W c^3. Peirce's formula is a fit to the exact elastica that is only
accurate near 41.5 degrees (about 6% high there, worse at steeper angles), so this uses the
overhang interpolated to 41.5 degrees when the measured overhangs bracket it, and otherwise the
curve whose tip angle is closest.

    python -m validate.peirce --fabric denim
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from validate.elastica import GRAVITY, peirce_rigidity, tip_angle

ASTM_ANGLE = math.radians(41.5)


def peirce_estimate(overhangs, tips, areal_density) -> dict:
    """overhangs in m, tips as (x, y) in m relative to the table edge."""
    angles = np.array([tip_angle(*t) for t in tips])
    order = np.argsort(overhangs)
    o = np.asarray(overhangs)[order]
    a = angles[order]
    w = areal_density * GRAVITY
    if a.min() <= ASTM_ANGLE <= a.max():
        l = float(np.interp(ASTM_ANGLE, a, o))
        return {"rigidity": w * (l / 2) ** 3, "method": "astm_41.5", "overhang_m": l}
    i = int(np.argmin(np.abs(a - ASTM_ANGLE)))
    return {
        "rigidity": peirce_rigidity(o[i], a[i], areal_density),
        "method": "closest_angle",
        "overhang_m": float(o[i]),
        "angle_deg": math.degrees(a[i]),
    }


def main():
    from fit.calibrate import load_fabric

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fabric", required=True)
    ap.add_argument("--curves", type=Path, default=Path("curves"))
    ap.add_argument("--fabrics", type=Path, default=Path("fabrics.yaml"))
    ap.add_argument("--params", type=Path, default=Path("params"))
    args = ap.parse_args()

    rho, cases = load_fabric(args.fabric, args.curves, args.fabrics)
    fitted = json.loads((args.params / f"{args.fabric}.json").read_text())["flexural_rigidity_uNm_fit"]
    out = {}
    for d, name in ((0, "warp"), (1, "weft")):
        cs = [c for c in cases if c.direction == d]
        if not cs:
            continue
        est = peirce_estimate([c.overhang for c in cs], [c.target_xy[-1] for c in cs], rho)
        g_p = est["rigidity"] * 1e6
        out[name] = {"peirce_uNm": g_p, "fit_uNm": fitted[name], "diff_pct": 100 * (fitted[name] / g_p - 1), **est}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
