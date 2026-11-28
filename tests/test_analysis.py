"""Unit tests for the analysis functions in src/analysis.py."""

import numpy as np

from src.analysis import (bonded_patch_fraction, cluster_sizes, coordination_numbers,
                          find_bonds, mean_cluster_size, mean_squared_displacement)
from src.dynamics import SimulationParameters
from src.particles import PatchySystem

# Hand-made bonds (particle i, particle j, patch k, patch m) for 6 particles:
# a chain 0-1-2, a pair 3-4 and a single particle 5.
BONDS = np.array([[0, 1, 1, 2], [1, 2, 3, 4], [3, 4, 7, 8]])
N_PARTICLES = 6


def two_particle_system(orientations):
    """Two two-patch particles at contact distance along the x-axis."""
    return PatchySystem(np.array([[5.0, 5.0], [6.05, 5.0]]), np.array(orientations),
                        np.array([2, 2]), np.array([0, 0, 1, 1]),
                        np.array([0.0, np.pi, 0.0, np.pi]), 20.0)


def test_find_bonds_detects_facing_patches_only():
    params = SimulationParameters()
    facing = find_bonds(two_particle_system([0.0, 0.0]), params)
    turned = find_bonds(two_particle_system([np.pi / 2, np.pi / 2]), params)
    # Patch 0 of particle 0 points at patch 3 (angle pi) of particle 1.
    np.testing.assert_array_equal(facing, [[0, 1, 0, 3]])
    assert len(turned) == 0


def test_coordination_numbers_count_bonds_per_particle():
    np.testing.assert_array_equal(coordination_numbers(BONDS, N_PARTICLES), [1, 2, 1, 1, 1, 0])


def test_bonded_patch_fraction_counts_two_patches_per_bond():
    assert bonded_patch_fraction(BONDS, n_patches=12) == 0.5


def test_cluster_sizes_and_mean_cluster_size():
    sizes = np.sort(cluster_sizes(BONDS, N_PARTICLES))
    np.testing.assert_array_equal(sizes, [1, 2, 3])
    # C = (1 + 4 + 9) / (1 + 2 + 3)
    assert np.isclose(mean_cluster_size(sizes), 14 / 6)


def test_mean_squared_displacement_of_uniform_motion():
    # Every particle moves 1 sigma per frame along x, so MSD = t^2.
    times = np.arange(5.0)
    positions = np.zeros((5, 3, 2))
    positions[:, :, 0] = times[:, None]
    np.testing.assert_allclose(mean_squared_displacement(positions), times**2)
