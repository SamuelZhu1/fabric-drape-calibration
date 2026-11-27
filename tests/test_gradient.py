"""wp.Tape gradients against central finite differences, and fitting a strip whose answer we know."""

import numpy as np
import warp as wp

from cloth.sim import SimConfig, StripBatch, StripCase
from fit.calibrate import fit, loss_and_grad
from conftest import log_params

TRUE = np.array([150e-6, 60e-6])


def synthetic_batch(cfg, overhangs=(0.10, 0.20), noise_m=0.0, seed=0):
    cases = [StripCase(o, d) for d in (0, 1) for o in overhangs]
    gen = StripBatch(cases, 0.2, config=cfg, device="cpu")
    x = gen.rollout_fast(gen.expand_params(log_params(TRUE)))
    rng = np.random.default_rng(seed)
    for c, cl in zip(cases, gen.centerlines(x)):
        c.target_xy = cl + rng.normal(0.0, noise_m, cl.shape)
    batch = StripBatch(cases, 0.2, config=cfg, device="cpu")
    batch.loss_setup()
    return batch


def loss_at(batch, p):
    x = batch.rollout_fast(batch.expand_params(wp.array(p.astype(np.float32), dtype=float)))
    return float(batch.loss(x).numpy()[0])


def test_tape_gradient_matches_finite_differences():
    batch = synthetic_batch(SimConfig(settle_time=1.5))
    p0 = np.log([40e-6, 40e-6])
    _, grad, _ = loss_and_grad(batch, log_params(np.exp(p0), grad=True))
    h = 1e-2
    for i in range(2):
        e = np.eye(2)[i]
        fd = (loss_at(batch, p0 + h * e) - loss_at(batch, p0 - h * e)) / (2 * h)
        assert abs(grad[i] - fd) / abs(fd) < 1e-3


def test_recovers_known_rigidity_from_noisy_curves():
    # 0.5 mm of noise on every point is about what the photo pipeline adds.
    batch = synthetic_batch(SimConfig(settle_time=1.5), noise_m=0.5e-3)
    result = fit(batch, iters=60, lr=0.2)
    err = np.abs(np.array(result["rigidity"]) / TRUE - 1)
    assert np.all(err < 0.02), err
