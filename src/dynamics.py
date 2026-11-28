"""Brownian dynamics of patchy particles in 2D.

Each particle performs overdamped Langevin motion. With the Euler-Maruyama
scheme of Ermak & McCammon (1978), one time step dt reads

    r(t + dt)   = r(t)   + D_t F dt / (k_B T) + sqrt(2 D_t dt) xi
    phi(t + dt) = phi(t) + D_r T dt / (k_B T) + sqrt(2 D_r dt) eta

where F and T are the force and torque on the particle and xi, eta are
independent standard Gaussian random numbers.

Units: lengths in sigma, energies in k_B T and time in tau = sigma^2 / D_t,
so D_t = 1. For a sphere in a viscous fluid D_r = 3 D_t / sigma^2
(Stokes-Einstein), which is the default here.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from src.forces import compute_forces
from src.geometry import wrap_positions


@dataclass
class SimulationParameters:
    """Physical and numerical parameters of a simulation.

    Attributes
    ----------
    epsilon : float
        Patch bond strength (k_B T).
    delta : float
        Range of the patch attraction beyond contact (sigma).
    patch_half_angle : float
        Half-opening angle theta_max of a patch (rad).
    dt : float
        Time step (tau).
    diffusion_translational : float
        Translational diffusion coefficient D_t (sigma^2 / tau).
    diffusion_rotational : float
        Rotational diffusion coefficient D_r (1 / tau).
    wca_epsilon : float
        Strength of the WCA repulsion (k_B T).
    """

    epsilon: float = 8.0
    delta: float = 0.15
    patch_half_angle: float = np.radians(20.0)
    dt: float = 1e-4
    diffusion_translational: float = 1.0
    diffusion_rotational: float = 3.0
    wca_epsilon: float = 1.0


@dataclass
class Trajectory:
    """Saved frames of a simulation.

    Attributes
    ----------
    times : ndarray, shape (F,)
        Time of each frame (tau).
    positions : ndarray, shape (F, N, 2)
        Particle positions, wrapped into the box.
    unwrapped_positions : ndarray, shape (F, N, 2)
        Particle positions without periodic wrapping (for the MSD).
    orientations : ndarray, shape (F, N)
        Particle orientation angles (rad).
    energies : ndarray, shape (F,)
        Total potential energy (k_B T).
    """

    times: np.ndarray
    positions: np.ndarray
    unwrapped_positions: np.ndarray
    orientations: np.ndarray
    energies: np.ndarray


@njit(cache=True)
def _seed_numba(seed):
    """Seed Numba's internal random number generator."""
    np.random.seed(seed)


@njit(cache=True)
def _integrate(positions, orientations, patch_start, valence, patch_offset, box_length,
               epsilon, delta, cos_max, wca_epsilon, dt, diff_t, diff_r,
               n_steps, save_every):
    """Run the Euler-Maruyama loop and return the saved frames."""
    n_particles = len(positions)
    n_frames = n_steps // save_every + 1
    saved_pos = np.empty((n_frames, n_particles, 2))
    saved_unwrapped = np.empty((n_frames, n_particles, 2))
    saved_angle = np.empty((n_frames, n_particles))
    saved_energy = np.empty(n_frames)

    unwrapped = positions.copy()
    noise_t = np.sqrt(2.0 * diff_t * dt)
    noise_r = np.sqrt(2.0 * diff_r * dt)
    frame = 0
    for step in range(n_steps + 1):
        forces, torques, energy = compute_forces(positions, orientations, patch_start,
                                                 valence, patch_offset, box_length,
                                                 epsilon, delta, cos_max, wca_epsilon)
        if step % save_every == 0:
            saved_pos[frame] = positions
            saved_unwrapped[frame] = unwrapped
            saved_angle[frame] = orientations
            saved_energy[frame] = energy
            frame += 1
        if step == n_steps:
            break
        for i in range(n_particles):
            for d in range(2):
                move = diff_t * forces[i, d] * dt + noise_t * np.random.normal()
                unwrapped[i, d] += move
                positions[i, d] += move
            orientations[i] += diff_r * torques[i] * dt + noise_r * np.random.normal()
        positions = wrap_positions(positions, box_length)
    return saved_pos, saved_unwrapped, saved_angle, saved_energy


def patch_start_indices(valence):
    """Index of the first patch of every particle in the flat patch arrays.

    Parameters
    ----------
    valence : ndarray of int, shape (N,)
        Number of patches per particle.

    Returns
    -------
    ndarray of int, shape (N,)
        Start index of each particle's patches.
    """
    return np.concatenate(([0], np.cumsum(valence)[:-1])).astype(np.int64)


def run_simulation(system, params, total_time, save_interval, seed):
    """Run Brownian dynamics and update ``system`` to the final state.

    Parameters
    ----------
    system : PatchySystem
        Initial state; it is updated in place to the final state, so a run
        can be continued (e.g. with a different epsilon) by calling again.
    params : SimulationParameters
        Physical and numerical parameters.
    total_time : float
        Simulated time (tau).
    save_interval : float
        Time between saved frames (tau).
    seed : int
        Seed for the random noise, so that runs are reproducible.

    Returns
    -------
    Trajectory
        Saved frames, including the initial state at time 0.
    """
    n_steps = int(round(total_time / params.dt))
    save_every = max(int(round(save_interval / params.dt)), 1)
    _seed_numba(seed)
    positions, unwrapped, orientations, energies = _integrate(
        system.positions.copy(), system.orientations.copy(),
        patch_start_indices(system.valence), system.valence, system.patch_offset,
        system.box_length, params.epsilon, params.delta,
        np.cos(params.patch_half_angle), params.wca_epsilon, params.dt,
        params.diffusion_translational, params.diffusion_rotational,
        n_steps, save_every)
    system.positions = positions[-1].copy()
    system.orientations = orientations[-1].copy()
    times = np.arange(len(energies)) * save_every * params.dt
    return Trajectory(times, positions, unwrapped, orientations, energies)
