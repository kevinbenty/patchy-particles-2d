"""Unit tests for the interactions in src/forces.py."""

import numpy as np

from src.dynamics import patch_start_indices
from src.forces import WCA_CUTOFF, angular_switch, compute_forces, radial_switch, wca_energy_force
from src.particles import create_system

EPSILON, DELTA, COS_MAX = 8.0, 0.15, np.cos(np.radians(20.0))


def pair_forces(positions, orientations, box_length=20.0):
    """Forces, torques and energy of two two-patch particles."""
    valence = np.array([2, 2])
    patch_offset = np.array([0.0, np.pi, 0.0, np.pi])
    return compute_forces(positions, orientations, patch_start_indices(valence), valence,
                          patch_offset, box_length, EPSILON, DELTA, COS_MAX, 1.0)


def test_wca_is_repulsive_inside_and_zero_beyond_cutoff():
    energy, force = wca_energy_force(0.95, 1.0)
    assert energy > 0.0 and force > 0.0
    assert wca_energy_force(WCA_CUTOFF + 1e-9, 1.0) == (0.0, 0.0)


def test_switches_go_smoothly_from_one_to_zero():
    # Full attraction at contact / perfect alignment, none beyond the range.
    assert radial_switch(1.0, DELTA) == (1.0, 0.0)
    assert radial_switch(1.0 + DELTA, DELTA) == (0.0, 0.0)
    assert np.isclose(angular_switch(1.0, COS_MAX)[0], 1.0)
    assert angular_switch(COS_MAX, COS_MAX) == (0.0, 0.0)


def test_facing_patches_attract_and_turned_away_patches_do_not():
    positions = np.array([[5.0, 5.0], [6.05, 5.0]])
    _, _, energy_facing = pair_forces(positions, np.array([0.0, 0.0]))
    # Turning both particles by 90 degrees points all patches away from each other.
    _, _, energy_away = pair_forces(positions, np.array([np.pi / 2, np.pi / 2]))
    # Perfect alignment gives f = 1 for both patches, so only g(r) and the
    # WCA repulsion remain; turned away, only the WCA repulsion is left.
    wca = wca_energy_force(1.05, 1.0)[0]
    assert np.isclose(energy_facing, -EPSILON * radial_switch(1.05, DELTA)[0] + wca)
    assert np.isclose(energy_away, wca)


def test_forces_and_torques_are_minus_energy_gradient():
    # Compare with central finite differences of the energy.
    positions = np.array([[5.0, 5.0], [6.05, 5.02]])
    orientations = np.array([0.05, np.pi - 0.1])
    forces, torques, _ = pair_forces(positions, orientations)
    h = 1e-6
    for i in range(2):
        for d in range(2):
            plus, minus = positions.copy(), positions.copy()
            plus[i, d] += h
            minus[i, d] -= h
            numeric = -(pair_forces(plus, orientations)[2] - pair_forces(minus, orientations)[2]) / (2 * h)
            assert np.isclose(forces[i, d], numeric, rtol=1e-5, atol=1e-6)
        plus, minus = orientations.copy(), orientations.copy()
        plus[i] += h
        minus[i] -= h
        numeric = -(pair_forces(positions, plus)[2] - pair_forces(positions, minus)[2]) / (2 * h)
        assert np.isclose(torques[i], numeric, rtol=1e-5, atol=1e-6)


def test_total_force_is_zero():
    # Newton's third law: internal forces cancel in a closed system.
    system = create_system(150, 0.3, 0.4, seed=5)
    forces, _, _ = compute_forces(system.positions, system.orientations,
                                  patch_start_indices(system.valence), system.valence,
                                  system.patch_offset, system.box_length,
                                  EPSILON, DELTA, COS_MAX, 1.0)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)


def test_energy_is_unchanged_by_shifting_the_periodic_box():
    # Shifting all particles moves pairs across the boundaries and between
    # cells; a correct cell list and minimum image give the same energy.
    system = create_system(150, 0.3, 0.4, seed=6)
    args = (patch_start_indices(system.valence), system.valence, system.patch_offset,
            system.box_length, EPSILON, DELTA, COS_MAX, 1.0)
    orientations = system.orientations
    energy = compute_forces(system.positions, orientations, *args)[2]
    shifted = np.mod(system.positions + np.array([3.7, 11.2]), system.box_length)
    assert np.isclose(compute_forces(shifted, orientations, *args)[2], energy)
