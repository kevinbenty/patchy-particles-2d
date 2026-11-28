"""Unit tests for the particle set-up in src/particles.py."""

import numpy as np

from src.geometry import minimum_image
from src.particles import box_length_for_area_fraction, create_system, patch_offsets


def test_patch_offsets_are_evenly_spaced():
    np.testing.assert_allclose(patch_offsets(2), [0.0, np.pi])
    np.testing.assert_allclose(patch_offsets(3), [0.0, 2 * np.pi / 3, 4 * np.pi / 3])


def test_box_length_reproduces_area_fraction():
    n_particles, area_fraction = 250, 0.1
    box_length = box_length_for_area_fraction(n_particles, area_fraction)
    covered = n_particles * np.pi / 4 / box_length**2
    assert np.isclose(covered, area_fraction)


def test_create_system_has_requested_mixture_and_patches():
    system = create_system(n_particles=100, fraction_three_patch=0.2, area_fraction=0.1, seed=1)
    assert np.sum(system.valence == 3) == 20
    assert np.sum(system.valence == 2) == 80
    # Every particle owns exactly as many patches as its valence.
    np.testing.assert_array_equal(np.bincount(system.patch_owner), system.valence)
    assert system.n_patches == 2 * 80 + 3 * 20


def test_create_system_has_no_overlaps():
    system = create_system(n_particles=200, fraction_three_patch=0.5, area_fraction=0.3, seed=2)
    delta = system.positions[:, None, :] - system.positions[None, :, :]
    distance = np.linalg.norm(minimum_image(delta, system.box_length), axis=-1)
    np.fill_diagonal(distance, np.inf)
    assert distance.min() >= 1.0


def test_create_system_is_reproducible_with_seed():
    first = create_system(50, 0.3, 0.1, seed=7)
    second = create_system(50, 0.3, 0.1, seed=7)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.valence, second.valence)
