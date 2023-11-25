import sys

import numpy as np

from propagation import vallado as vallado_fast

def vallado(k, r0, v0, tof, *, numiter):
    # Compute Lagrange coefficients
    f, g, fdot, gdot = vallado_fast(k, r0, v0, tof, numiter)

    assert (
        np.abs(f * gdot - fdot * g - 1) < 1e-5
    ), "Internal error, solution is not consistent"  # Fixed tolerance

    # Return position and velocity vectors
    r = f * r0 + g * v0
    v = fdot * r0 + gdot * v0

    return r, v
