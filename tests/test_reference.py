"""The simulator against the exact heavy elastica, and Peirce's formula against both."""

import math

import numpy as np
import pytest

from cloth.sim import StripBatch, StripCase
from validate.elastica import heavy_elastica, peirce_rigidity, tip_angle
from validate.peirce import peirce_estimate
from conftest import log_params

RHO = 0.2  # kg/m^2, a light shirting cotton


@pytest.mark.parametrize("rigidity,overhang", [(100e-6, 0.10), (100e-6, 0.15), (400e-6, 0.20)])
def test_sim_matches_exact_elastica(rigidity, overhang):
    batch = StripBatch([StripCase(overhang, 0)], RHO, device="cpu")
    x = batch.rollout_fast(batch.expand_params(log_params((rigidity, rigidity))))
    cl = batch.centerlines(x)[0]
    s, ex, ey = heavy_elastica(rigidity, RHO, overhang)
    # Compare at each column's arc length along the exact curve.
    want = np.stack([np.interp(np.arange(len(cl)) * batch.mesh.dx, s, ex),
                     np.interp(np.arange(len(cl)) * batch.mesh.dx, s, ey)], axis=1)
    err = np.linalg.norm(cl - want, axis=1)
    assert err.max() < 3e-3  # 3 mm on a 25 cm strip with 1.25 cm spacing


def test_peirce_is_close_only_near_41_5_degrees():
    g = 400e-6
    w = RHO * 9.81
    c = (g / w) ** (1 / 3)
    ratios = {}
    for l in (2 * c, 3.5 * c):
        _, x, y = heavy_elastica(g, RHO, l)
        th = tip_angle(x[-1], y[-1])
        ratios[round(math.degrees(th))] = peirce_rigidity(l, th, RHO) / g
    near, steep = sorted(ratios)
    assert abs(ratios[near] - 1) < 0.10
    assert ratios[steep] > 1.15  # why the check uses the 41.5 degree overhang


def test_astm_interpolation_brackets_41_5():
    g = 400e-6
    overhangs = [0.10, 0.125, 0.15, 0.175, 0.20]
    tips = []
    for l in overhangs:
        _, x, y = heavy_elastica(g, RHO, l)
        tips.append((x[-1], y[-1]))
    est = peirce_estimate(overhangs, tips, RHO)
    assert est["method"] == "astm_41.5"
    assert abs(est["rigidity"] / g - 1) < 0.10
