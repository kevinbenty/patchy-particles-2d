"""Periodic boundary helpers for a square 2D simulation box.

The box has side length ``box_length`` and coordinates in [0, box_length).
All functions are compiled with Numba so they can be called both from
Python and from the compiled force and dynamics loops.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def wrap_positions(positions, box_length):
    """Map positions back into the periodic box.

    Parameters
    ----------
    positions : float or ndarray
        Coordinates, e.g. an array of shape (N, 2).
    box_length : float
        Side length L of the square box.

    Returns
    -------
    float or ndarray
        Coordinates in the interval [0, L).
    """
    return positions - box_length * np.floor(positions / box_length)


@njit(cache=True)
def minimum_image(displacement, box_length):
    """Return the shortest periodic image of a displacement.

    Parameters
    ----------
    displacement : float or ndarray
        Displacement component(s) between two points, e.g. x_j - x_i.
    box_length : float
        Side length L of the square box.

    Returns
    -------
    float or ndarray
        Equivalent displacement with every component in [-L/2, L/2).
    """
    return displacement - box_length * np.floor(displacement / box_length + 0.5)
