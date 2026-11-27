"""Reference solutions for a cantilevered strip under its own weight.

`heavy_elastica` solves the exact large-deflection equation for an inextensible strip clamped
flat at the table edge:

    G theta''(s) = -W (L - s) cos(theta(s)),  theta(0) = 0,  theta'(L) = 0

where theta is the angle below horizontal, s the arc length from the edge, G the flexural
rigidity per unit width (N m) and W the weight per unit area (N/m^2).

`peirce_rigidity` is Peirce's 1930 closed-form estimate from one overhang and the tip angle,
the formula behind the ASTM D1388 cantilever test.
"""

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp

GRAVITY = 9.81


def heavy_elastica(rigidity: float, areal_density: float, overhang: float, n: int = 400):
    """Return (s, x, y) of the settled centerline, metres, y negative below the table."""
    w = areal_density * GRAVITY
    a = w / rigidity

    def rhs(s, y):
        theta, dtheta = y
        return np.vstack([dtheta, -a * (overhang - s) * np.cos(theta)])

    def bc(ya, yb):
        return np.array([ya[0], yb[1]])

    s = np.linspace(0.0, overhang, n)
    # Start from the small-deflection solution, clipped so cos(theta) stays sensible.
    guess = np.zeros((2, n))
    lin = a * (overhang**2 * s / 2 - overhang * s**2 / 2 + s**3 / 6)
    guess[0] = np.clip(lin, 0, 1.5)
    guess[1] = np.gradient(guess[0], s)
    sol = solve_bvp(rhs, bc, s, guess, tol=1e-8, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)
    theta = sol.sol(s)[0]
    x = cumulative_trapezoid(np.cos(theta), s, initial=0.0)
    y = -cumulative_trapezoid(np.sin(theta), s, initial=0.0)
    return s, x, y


def tip_angle(tip_x: float, tip_y: float) -> float:
    """Angle of the line from the table edge to the tip, below horizontal, radians."""
    return math.atan2(-tip_y, tip_x)


def peirce_rigidity(overhang: float, theta: float, areal_density: float) -> float:
    """Peirce: bending length c = l (cos(theta/2) / (8 tan(theta)))^(1/3), G = W c^3."""
    c = overhang * (math.cos(theta / 2) / (8 * math.tan(theta))) ** (1.0 / 3.0)
    return areal_density * GRAVITY * c**3
