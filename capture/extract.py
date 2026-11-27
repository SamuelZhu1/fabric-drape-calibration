"""Turn a side photo of a strip hanging off a table into a centerline curve in centimetres.

Setup: the phone sits level with the table and looks along the strip's width, so the strip
shows up as a thin dark curve. A printed checkerboard is taped flat in the same plane as the
strip. The board gives the pixel to centimetre mapping (a homography), and a separate set of
board photos gives the lens distortion (`calibrate`).

    python -m capture.extract calibrate calib/*.jpg --board 9 6 --square 1.0 --out camera.json
    python -m capture.extract curve photo.jpg --camera camera.json --edge 812 640 \
        --fabric denim --direction warp --overhang 15 --out curves/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

N_POINTS = 60
MAX_REPROJ_PX = 1.5


def find_board(gray: np.ndarray, board: tuple):
    ok, corners = cv2.findChessboardCorners(gray, board, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)


def board_points(board: tuple, square_cm: float) -> np.ndarray:
    """Board corner coordinates in cm, x to the right and y up, matching OpenCV's corner order."""
    cols, rows = board
    g = np.array([[c * square_cm, -r * square_cm] for r in range(rows) for c in range(cols)], dtype=np.float32)
    return g


def calibrate_camera(paths, board, square_cm) -> dict:
    obj, img, size = [], [], None
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        corners = find_board(gray, board)
        if corners is None:
            print(f"no board found in {p}, skipping")
            continue
        size = gray.shape[::-1]
        pts = board_points(board, square_cm)
        obj.append(np.hstack([pts, np.zeros((len(pts), 1), np.float32)]))
        img.append(corners)
    if len(obj) < 3:
        raise RuntimeError("need at least 3 board photos to calibrate the lens")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms), "size": list(size)}


def strip_mask(gray: np.ndarray, board_quad: np.ndarray) -> np.ndarray:
    """Dark pixels outside the checkerboard. Otsu picks the threshold per photo."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hull = cv2.convexHull(board_quad.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 0, lineType=cv2.LINE_8)
    # Grow the board cut-out a little so its border squares don't leak into the mask.
    cv2.polylines(mask, [hull], True, 0, thickness=15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def order_skeleton(skel: np.ndarray, start_px) -> np.ndarray:
    """Walk the one pixel wide skeleton from the pixel nearest the table edge to its far end."""
    ys, xs = np.nonzero(skel)
    pts = np.stack([xs, ys], axis=1)
    on = {tuple(p) for p in pts}
    cur = tuple(pts[np.argmin(np.linalg.norm(pts - np.asarray(start_px), axis=1))])
    path = [cur]
    seen = {cur}
    while True:
        nxt = None
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            q = (cur[0] + dx, cur[1] + dy)
            if q in on and q not in seen:
                nxt = q
                break
        if nxt is None:
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return np.array(path, dtype=np.float32)


def resample(points: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], axis=1)


def extract_curve(image: np.ndarray, edge_px, board=(9, 6), square_cm=1.0, camera: dict = None) -> dict:
    """Return the centerline in cm relative to the table edge, plus quality numbers."""
    if camera is not None:
        K = np.array(camera["K"])
        dist = np.array(camera["dist"])
        image = cv2.undistort(image, K, dist)
        edge_px = cv2.undistortPoints(np.array([[edge_px]], np.float32), K, dist, P=K)[0, 0]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = find_board(gray, board)
    if corners is None:
        raise RuntimeError("checkerboard not found")
    obj = board_points(board, square_cm)
    H, _ = cv2.findHomography(corners.reshape(-1, 2), obj)
    back = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    reproj = float(np.sqrt(np.mean(np.sum((back - corners.reshape(-1, 2)) ** 2, axis=1))))

    cols, rows = board
    c = corners.reshape(rows, cols, 2)
    quad = np.array([c[0, 0], c[0, -1], c[-1, -1], c[-1, 0]])
    centre = quad.mean(axis=0)
    # The outer squares stick out one square past the inner corners.
    quad = centre + (quad - centre) * np.array([(cols + 1) / (cols - 1), (rows + 1) / (rows - 1)])
    mask = strip_mask(gray, quad)
    # Only the hanging part: drop the strip lying on the table behind the edge.
    mask[:, : int(round(edge_px[0]))] = 0
    skel = cv2.ximgproc.thinning(mask)
    path_px = order_skeleton(skel, edge_px)
    if len(path_px) < 10:
        raise RuntimeError("strip not found next to the table edge")
    # Thinning eats about half the stroke width off the free end, which is 1 to 2 mm of a
    # 15 cm strip. Put it back along the last tangent, using the stroke's median half width
    # (less one pixel, since the distance transform counts the centre pixel too).
    half = float(np.median(cv2.distanceTransform(mask, cv2.DIST_L2, 5)[path_px[:, 1].astype(int), path_px[:, 0].astype(int)]))
    tangent = path_px[-1] - path_px[max(0, len(path_px) - 8)]
    tangent /= np.linalg.norm(tangent) + 1e-9
    path_px = np.vstack([np.asarray(edge_px, np.float32)[None], path_px, path_px[-1] + max(half - 1.0, 0.0) * tangent])
    to_cm = lambda p: cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
    path_cm = to_cm(path_px) - to_cm(np.array([edge_px]))[0]
    px_per_cm = float(np.linalg.norm(corners[1] - corners[0]) / square_cm)
    return {
        "points_cm": resample(path_cm).round(4).tolist(),
        "px_per_cm": px_per_cm,
        "reprojection_err_px": reproj,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate")
    cal.add_argument("photos", nargs="+", type=Path)
    cal.add_argument("--board", type=int, nargs=2, default=[9, 6])
    cal.add_argument("--square", type=float, default=1.0)
    cal.add_argument("--out", type=Path, default=Path("camera.json"))
    cur = sub.add_parser("curve")
    cur.add_argument("photo", type=Path)
    cur.add_argument("--camera", type=Path)
    cur.add_argument("--edge", type=float, nargs=2, required=True, help="table edge pixel (x y)")
    cur.add_argument("--board", type=int, nargs=2, default=[9, 6])
    cur.add_argument("--square", type=float, default=1.0)
    cur.add_argument("--fabric", required=True)
    cur.add_argument("--direction", choices=["warp", "weft"], required=True)
    cur.add_argument("--overhang", type=float, required=True, help="cm")
    cur.add_argument("--width", type=float, default=5.0, help="cm")
    cur.add_argument("--out", type=Path, default=Path("curves"))
    a = ap.parse_args()

    if a.cmd == "calibrate":
        cam = calibrate_camera(a.photos, tuple(a.board), a.square)
        a.out.write_text(json.dumps(cam, indent=2))
        print(f"lens calibrated, rms {cam['rms_px']:.2f} px")
        return
    camera = json.loads(a.camera.read_text()) if a.camera else None
    res = extract_curve(cv2.imread(str(a.photo)), a.edge, tuple(a.board), a.square, camera)
    if res["reprojection_err_px"] > MAX_REPROJ_PX:
        raise SystemExit(f"board reprojection {res['reprojection_err_px']:.2f} px > {MAX_REPROJ_PX}, retake the photo")
    res.update(fabric=a.fabric, direction=a.direction, overhang_cm=a.overhang, width_cm=a.width, photo=a.photo.name)
    a.out.mkdir(exist_ok=True)
    name = f"{a.fabric}_{a.direction}_{a.overhang:g}.json"
    (a.out / name).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out / name}")


if __name__ == "__main__":
    main()
