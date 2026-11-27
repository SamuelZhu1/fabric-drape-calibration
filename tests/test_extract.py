"""Draw a strip and a checkerboard into a fake photo and read the curve back."""

import cv2
import numpy as np

from capture.extract import extract_curve
from validate.elastica import heavy_elastica

PX_PER_CM = 20.0


def fake_photo(curve_cm, edge_px):
    img = np.full((960, 1280), 235, np.uint8)
    # Board plane (cm, y up) to pixels, with a little perspective like a phone slightly off level.
    H = np.array([[PX_PER_CM, 0.8, edge_px[0]], [0.5, -PX_PER_CM, edge_px[1]], [2e-4, 1e-4, 1.0]])

    def px(p):
        q = np.hstack([p, np.ones((len(p), 1))]) @ H.T
        return np.round(q[:, :2] / q[:, 2:] * 16).astype(np.int32)

    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                x0, y0 = -13 + c, -6 - r
                sq = np.array([[x0, y0], [x0 + 1, y0], [x0 + 1, y0 - 1], [x0, y0 - 1]], float)
                cv2.fillConvexPoly(img, px(sq), 20, cv2.LINE_AA, 4)
    cv2.polylines(img, [px(curve_cm)], False, 30, 8, cv2.LINE_AA, 4)
    cv2.line(img, (0, int(edge_px[1])), (int(edge_px[0]), int(edge_px[1])), 30, 8, cv2.LINE_AA)
    noise = np.random.default_rng(1).normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_curve_comes_back_within_a_millimetre():
    _, x, y = heavy_elastica(100e-6, 0.2, 0.15)
    truth = np.stack([x, y], axis=1) * 100
    edge = [300.0, 250.0]
    res = extract_curve(fake_photo(truth, edge), edge)
    pts = np.array(res["points_cm"])
    assert len(pts) == 60
    assert res["reprojection_err_px"] < 0.5
    dist = np.min(np.linalg.norm(pts[:, None] - truth[None], axis=2), axis=1)
    assert dist.max() < 0.1  # cm
    assert np.linalg.norm(pts[-1] - truth[-1]) < 0.1
