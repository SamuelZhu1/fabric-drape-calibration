# fabric-drape-calibration

A cloth simulator in [NVIDIA Warp](https://github.com/NVIDIA/warp) that learns a real fabric's
bending stiffness from photos of it hanging off a table.

At a hackathon I built a virtual try-on app whose cloth draped like cardboard, because I had
picked the stiffness by eye. Measuring stiffness properly usually needs a lab instrument, but the
old cantilever bending test (Peirce 1930, ASTM D1388) only needs a table edge, a ruler and a
camera. This project photographs that test and fits the simulator to it by differentiating
through the whole simulation.

## How it works

```
 side photos of a 25 x 5 cm strip at several overhangs
        |
 capture/extract.py   lens undistortion + checkerboard homography -> pixels to cm
                      Otsu threshold -> thinning -> ordered centerline, 60 points
        |
 curves/<fabric>_<warp|weft>_<overhang>.json
        |
 fit/calibrate.py     wp.Tape records a 1.5 s settle rollout of every strip in one batch,
                      tape.backward() gives d(loss)/d(log G), Adam steps log G (warp, weft)
        |
 params/<fabric>.json  fitted rigidity, RMS error before/after, held-out error
        |
 validate/peirce.py   Peirce / ASTM D1388 estimate at the 41.5 degree overhang
 validate/elastica.py exact heavy-elastica curve (scipy solve_bvp), used by the tests
```

**Simulator (`cloth/`).** Each strip is a 21 x 5 grid of particles with stretch and shear springs
and a discrete bending energy `E = (G w / 2 h^3) |x_a - 2 x_b + x_c|^2`, so the parameter being
fitted is the flexural rigidity G (N m) that the cantilever test measures, not an arbitrary spring
constant. Semi-implicit Euler at dt = 0.25 ms with mass-proportional damping. Particles on the
table are pinned. A batch of B strips with different overhangs and cut directions runs in the
same kernel launches (one thread per particle, spring or bending triplet), which is what makes a
GPU worth it.

**The clamp.** Pinning the column behind the table edge only fixes the slope half a grid spacing
back, so the strip sagged as if it were h/2 longer: the tip was 11.7 mm off the exact solution at
21 columns, and only halved when I doubled the resolution (first order). Doubling the bending
weight of the triplet centred on the edge column is the same as a mirrored ghost node, and the
error dropped to 1.8 mm and stopped scaling with resolution.

**Gradients.** The first tape gradients came out wrong, one with the wrong sign. The loss kernel averaged each column
with a loop over a runtime range that reassigned a variable, which Warp can't differentiate.
Splitting it into an atomic column-mean kernel and a squared-error kernel fixed it, and the test
that compares against central finite differences now guards it.

## Results

Measured on a 4-core Intel Xeon (2.1 GHz) cloud machine, Warp 1.19 CPU backend.

| Check | Result |
| --- | --- |
| Tape gradient vs central finite differences, 6,000 steps | relative error 1e-5 (warp), 1e-4 (weft) |
| Recover known rigidity, 6 strips, 0.5 mm noise, start 7.5x and 3x off | 0.07% and 0.15% error, RMS 18.8 mm -> 0.67 mm |
| Simulator vs exact heavy elastica, 5 stiffness / overhang cases | max 1.6 to 3.4 mm along a 10 to 20 cm overhang |
| Photo extraction on a synthetic photo with perspective and noise | RMS 0.22 mm, max 0.69 mm |
| Energy while settling | never rises (sampled every 250 steps over 6,000) |

CPU rollout speed (`python -m bench.batch --steps 300`):

| strips | ms per step | strip-steps per s |
| --- | --- | --- |
| 1 | 0.098 | 10,246 |
| 64 | 0.649 | 98,546 |
| 256 | 2.045 | 125,179 |
| 1,024 | 7.082 | 144,585 |

Still to do: GPU timing (same script with `--device cpu cuda:0`) and fitting real fabric photos.

## Running it

```bash
pip install -r requirements.txt
pytest                       # 12 tests, about 4 minutes on CPU (the recovery test is most of it)

python -m capture.extract calibrate calib/*.jpg --out camera.json
python -m capture.extract curve photos/denim_warp_15.jpg --camera camera.json --edge 812 640 \
    --fabric denim --direction warp --overhang 15
python -m fit.calibrate --fabric denim --fit-overhangs 0.10 0.15 0.20
python -m validate.peirce --fabric denim
python -m bench.batch --device cpu cuda:0 --batch 1 64 256 1024
```

Taking the photos: phone on a stack of books level with the table, looking along the strip's
width, with a printed checkerboard (9 x 6 inner corners, 1 cm squares) taped flat in the plane of
the strip. Cut one strip along the warp and one along the weft, and photograph each at several
overhangs. Click the table edge pixel once per setup and pass it as `--edge`.

## Limits

- The strip is modelled as a flat sheet. Knits like jersey curl at the edges, which stiffens them
  in a way a single rigidity can't capture.
- Peirce's formula is only accurate near a 41.5 degree tip angle (about 6% high there, 19% high
  at 67 degrees, 41% at 77 degrees against the exact elastica), which is why the ASTM test uses
  that angle and why `validate/peirce.py` interpolates to it.
- Springs stretch a little under the strip's own weight, which is most of the remaining error
  against the elastica for very floppy fabric.
