"""Unit and system tests for the Brownian dynamics in src/dynamics.py."""

import numpy as np

from src.analysis import find_bonds, mean_squared_displacement
from src.dynamics import SimulationParameters, patch_start_indices, run_simulation
from src.particles import create_system


def test_patch_start_indices_point_to_first_patch_of_each_particle():
    np.testing.assert_array_equal(patch_start_indices(np.array([2, 3, 2])), [0, 2, 5])


def test_same_seed_gives_same_trajectory():
    params = SimulationParameters(epsilon=10.0)
    first = run_simulation(create_system(50, 0.2, 0.2, seed=1), params, 0.5, 0.1, seed=3)
    second = run_simulation(create_system(50, 0.2, 0.2, seed=1), params, 0.5, 0.1, seed=3)
    np.testing.assert_array_equal(first.positions, second.positions)


def test_free_particles_diffuse_with_msd_4dt():
    # System test: without patch attraction and at very low density the
    # particles are free Brownian walkers, so <r^2> = 4 D_t t and the
    # orientation spreads as <dphi^2> = 2 D_r t.
    system = create_system(400, 0.0, 0.005, seed=2)
    params = SimulationParameters(epsilon=0.0)
    traj = run_simulation(system, params, total_time=1.0, save_interval=0.25, seed=4)
    msd = mean_squared_displacement(traj.unwrapped_positions)
    angular_msd = np.mean((traj.orientations - traj.orientations[0]) ** 2, axis=1)
    # 400 particles give a statistical error of about 5 %; allow 15 %.
    np.testing.assert_allclose(msd[1:], 4.0 * traj.times[1:], rtol=0.15)
    np.testing.assert_allclose(angular_msd[1:], 2.0 * 3.0 * traj.times[1:], rtol=0.15)


def test_assembly_run_forms_limited_valency_bonds():
    # System test: a small version of the tutorial's assembly simulation.
    # Strong patches must lower the energy, form bonds, never bond one patch
    # twice, and bond more patches than weak patches do.
    bonded = {}
    for epsilon in (6.0, 12.0):
        system = create_system(120, 0.3, 0.15, seed=1)
        params = SimulationParameters(epsilon=epsilon)
        traj = run_simulation(system, params, total_time=3.0, save_interval=1.0, seed=1)
        bonds = find_bonds(system, params, threshold=0.0)
        assert np.bincount(bonds[:, 2:].ravel()).max() <= 1  # one bond per patch
        bonded[epsilon] = len(bonds)
        if epsilon == 12.0:
            assert traj.energies[-1] < traj.energies[0]
    assert bonded[12.0] > bonded[6.0] > 0
