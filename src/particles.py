"""Set-up of a 2D system of patchy particles.

Each particle is a disk of diameter ``sigma`` with an orientation angle and
two or three patches spaced evenly around its surface. Patches are stored in
flat arrays (one entry per patch) so that the compiled force loop can treat
all patches in the same way, whatever the valence of their particle.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PatchySystem:
    """State of a 2D system of patchy particles in a periodic square box.

    Attributes
    ----------
    positions : ndarray, shape (N, 2)
        Particle centres, in units of the particle diameter sigma.
    orientations : ndarray, shape (N,)
        Orientation angle of each particle (rad).
    valence : ndarray of int, shape (N,)
        Number of patches on each particle (2 or 3).
    patch_owner : ndarray of int, shape (M,)
        Index of the particle that carries each patch.
    patch_offset : ndarray, shape (M,)
        Angle of each patch relative to its particle's orientation (rad).
    box_length : float
        Side length L of the periodic box.
    """

    positions: np.ndarray
    orientations: np.ndarray
    valence: np.ndarray
    patch_owner: np.ndarray
    patch_offset: np.ndarray
    box_length: float

    @property
    def n_particles(self):
        """Number of particles N."""
        return len(self.positions)

    @property
    def n_patches(self):
        """Total number of patches M."""
        return len(self.patch_owner)

    def copy(self):
        """Return an independent copy of the system."""
        return PatchySystem(
            self.positions.copy(),
            self.orientations.copy(),
            self.valence.copy(),
            self.patch_owner.copy(),
            self.patch_offset.copy(),
            self.box_length,
        )


def patch_offsets(valence):
    """Return evenly spaced patch angles for a particle with ``valence`` patches.

    A two-patch particle has patches 180 degrees apart (it can only form
    straight-ish chains); a three-patch particle has patches 120 degrees
    apart (it can act as a branching point).

    Parameters
    ----------
    valence : int
        Number of patches on the particle.

    Returns
    -------
    ndarray, shape (valence,)
        Patch angles relative to the particle orientation (rad).
    """
    return 2.0 * np.pi * np.arange(valence) / valence


def box_length_for_area_fraction(n_particles, area_fraction, sigma=1.0):
    """Return the box side length that gives the requested area fraction.

    The area fraction is eta = N * pi * sigma^2 / 4 / L^2.

    Parameters
    ----------
    n_particles : int
        Number of particles N.
    area_fraction : float
        Fraction eta of the box area covered by the disks.
    sigma : float, optional
        Particle diameter.

    Returns
    -------
    float
        Box side length L.
    """
    return np.sqrt(n_particles * np.pi * sigma**2 / 4.0 / area_fraction)


def random_positions(n_particles, box_length, min_distance, rng, max_attempts=100000):
    """Place particles at random without overlaps (random sequential addition).

    Parameters
    ----------
    n_particles : int
        Number of particles to place.
    box_length : float
        Side length of the periodic box.
    min_distance : float
        Smallest allowed centre-to-centre distance.
    rng : numpy.random.Generator
        Random number generator.
    max_attempts : int, optional
        Maximum number of trial insertions before giving up.

    Returns
    -------
    ndarray, shape (n_particles, 2)
        Particle positions in [0, box_length).
    """
    positions = np.empty((n_particles, 2))
    n_placed = 0
    for _ in range(max_attempts):
        trial = rng.uniform(0.0, box_length, size=2)
        delta = positions[:n_placed] - trial
        delta -= box_length * np.round(delta / box_length)  # minimum image
        if n_placed == 0 or np.min(np.sum(delta**2, axis=1)) >= min_distance**2:
            positions[n_placed] = trial
            n_placed += 1
            if n_placed == n_particles:
                return positions
    raise RuntimeError("Could not place all particles; lower the area fraction.")


def create_system(n_particles, fraction_three_patch, area_fraction, seed, sigma=1.0):
    """Create a random, non-overlapping mixture of two- and three-patch particles.

    Parameters
    ----------
    n_particles : int
        Number of particles N.
    fraction_three_patch : float
        Fraction of particles with three patches (the rest have two).
    area_fraction : float
        Area fraction eta covered by the particles.
    seed : int
        Seed for the random number generator (positions, orientations, types).
    sigma : float, optional
        Particle diameter.

    Returns
    -------
    PatchySystem
        The initial state of the system.
    """
    rng = np.random.default_rng(seed)
    box_length = box_length_for_area_fraction(n_particles, area_fraction, sigma)
    positions = random_positions(n_particles, box_length, sigma, rng)
    orientations = rng.uniform(0.0, 2.0 * np.pi, size=n_particles)

    n_three_patch = int(round(fraction_three_patch * n_particles))
    valence = np.full(n_particles, 2, dtype=np.int64)
    valence[rng.permutation(n_particles)[:n_three_patch]] = 3

    patch_owner = np.repeat(np.arange(n_particles), valence)
    patch_offset = np.concatenate([patch_offsets(v) for v in valence])
    return PatchySystem(positions, orientations, valence, patch_owner, patch_offset, box_length)
