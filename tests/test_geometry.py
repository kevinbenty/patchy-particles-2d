"""Unit tests for the periodic boundary helpers in src/geometry.py."""

import numpy as np

from src.geometry import minimum_image, wrap_positions

BOX_LENGTH = 10.0


def test_wrap_positions_maps_into_box():
    positions = np.array([[-0.5, 3.0], [10.5, 25.0], [4.0, -12.0]])
    wrapped = wrap_positions(positions, BOX_LENGTH)
    assert np.all(wrapped >= 0.0) and np.all(wrapped < BOX_LENGTH)
    np.testing.assert_allclose(wrapped, [[9.5, 3.0], [0.5, 5.0], [4.0, 8.0]])


def test_wrap_positions_shifts_by_whole_box_lengths():
    # Wrapping may only move a point by an integer number of box lengths,
    # otherwise the physical position would change.
    positions = np.random.default_rng(0).uniform(-50.0, 50.0, size=(100, 2))
    shift = (positions - wrap_positions(positions, BOX_LENGTH)) / BOX_LENGTH
    np.testing.assert_allclose(shift, np.round(shift), atol=1e-12)


def test_minimum_image_takes_shortest_path_across_boundary():
    # Points at x = 0.5 and x = 9.5 are 1.0 apart through the boundary.
    assert np.isclose(minimum_image(9.5 - 0.5, BOX_LENGTH), -1.0)
    assert np.isclose(minimum_image(0.5 - 9.5, BOX_LENGTH), 1.0)


def test_minimum_image_components_within_half_box():
    displacements = np.random.default_rng(1).uniform(-40.0, 40.0, size=(200, 2))
    reduced = minimum_image(displacements, BOX_LENGTH)
    assert np.all(np.abs(reduced) <= BOX_LENGTH / 2)
