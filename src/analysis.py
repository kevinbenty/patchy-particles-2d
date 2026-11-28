"""Analysis of patchy-particle configurations: bonds, clusters and diffusion."""

import numpy as np
from numba import njit

from src.dynamics import patch_start_indices
from src.forces import angular_switch, build_cell_list, patch_pair_energy
from src.geometry import minimum_image


@njit(cache=True)
def _find_bonds(positions, orientations, patch_start, valence, patch_offset, box_length,
                epsilon, delta, cos_max, threshold):
    """Return patch-pair bonds as rows (particle_i, particle_j, patch_k, patch_m)."""
    n_particles = len(positions)
    cutoff = 1.0 + delta
    n_cells, head, next_in_cell = build_cell_list(positions, box_length, cutoff)
    bonds = []
    for i in range(n_particles):
        cx = int(positions[i, 0] / box_length * n_cells) % n_cells
        cy = int(positions[i, 1] / box_length * n_cells) % n_cells
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                if n_cells < 3 and (ox != 0 or oy != 0):
                    continue  # all particles share the single cell
                j = head[((cx + ox) % n_cells) * n_cells + (cy + oy) % n_cells]
                while j != -1:
                    if j > i:
                        dx = minimum_image(positions[j, 0] - positions[i, 0], box_length)
                        dy = minimum_image(positions[j, 1] - positions[i, 1], box_length)
                        r = np.sqrt(dx * dx + dy * dy)
                        if r < cutoff:
                            ux, uy = dx / r, dy / r
                            for k in range(patch_start[i], patch_start[i] + valence[i]):
                                a = orientations[i] + patch_offset[k]
                                cos_i = np.cos(a) * ux + np.sin(a) * uy
                                if angular_switch(cos_i, cos_max)[0] == 0.0:
                                    continue
                                for m in range(patch_start[j], patch_start[j] + valence[j]):
                                    b = orientations[j] + patch_offset[m]
                                    cos_j = -(np.cos(b) * ux + np.sin(b) * uy)
                                    energy = patch_pair_energy(cos_i, cos_j, r, epsilon,
                                                               delta, cos_max)
                                    if energy < -threshold * epsilon:
                                        bonds.append((i, j, k, m))
                    j = next_in_cell[j]
    result = np.empty((len(bonds), 4), dtype=np.int64)
    for n in range(len(bonds)):
        for c in range(4):
            result[n, c] = bonds[n][c]
    return result


def find_bonds(system, params, positions=None, orientations=None, threshold=0.5):
    """Find all patch-patch bonds in a configuration.

    Two patches count as bonded when their pair energy is below
    -threshold * epsilon, i.e. they are close and both point at each other.

    Parameters
    ----------
    system : PatchySystem
        System (supplies valence, patches and box size).
    params : SimulationParameters
        Interaction parameters (epsilon, delta, patch half-angle).
    positions, orientations : ndarray, optional
        Configuration to analyse; defaults to the current state of ``system``.
    threshold : float, optional
        Fraction of epsilon a patch pair must reach to count as a bond.

    Returns
    -------
    ndarray of int, shape (B, 4)
        One row per bond: particle i, particle j, patch of i, patch of j.
    """
    positions = system.positions if positions is None else positions
    orientations = system.orientations if orientations is None else orientations
    return _find_bonds(positions, orientations, patch_start_indices(system.valence),
                       system.valence, system.patch_offset, system.box_length,
                       params.epsilon, params.delta, np.cos(params.patch_half_angle),
                       threshold)


def coordination_numbers(bonds, n_particles):
    """Number of bonds Z of every particle.

    Parameters
    ----------
    bonds : ndarray of int, shape (B, 4)
        Bonds as returned by ``find_bonds``.
    n_particles : int
        Number of particles N.

    Returns
    -------
    ndarray of int, shape (N,)
        Z for each particle.
    """
    return np.bincount(bonds[:, :2].ravel(), minlength=n_particles)


def bonded_patch_fraction(bonds, n_patches):
    """Fraction p_b of all patches that take part in a bond.

    Parameters
    ----------
    bonds : ndarray of int, shape (B, 4)
        Bonds as returned by ``find_bonds``.
    n_patches : int
        Total number of patches M.

    Returns
    -------
    float
        p_b = 2 B / M (each bond uses two patches).
    """
    return 2.0 * len(bonds) / n_patches


def cluster_labels(bonds, n_particles):
    """Label the clusters of bonded particles (union-find).

    Parameters
    ----------
    bonds : ndarray of int, shape (B, 4)
        Bonds as returned by ``find_bonds``.
    n_particles : int
        Number of particles N.

    Returns
    -------
    ndarray of int, shape (N,)
        Cluster label of each particle, numbered 0, 1, 2, ...
    """
    parent = np.arange(n_particles)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving keeps the trees shallow
            i = parent[i]
        return i

    for i, j in bonds[:, :2]:
        root_i, root_j = root(i), root(j)
        if root_i != root_j:
            parent[root_i] = root_j
    roots = np.array([root(i) for i in range(n_particles)])
    return np.unique(roots, return_inverse=True)[1]


def cluster_sizes(bonds, n_particles):
    """Size (number of particles) of every cluster, single particles included.

    Parameters
    ----------
    bonds : ndarray of int, shape (B, 4)
        Bonds as returned by ``find_bonds``.
    n_particles : int
        Number of particles N.

    Returns
    -------
    ndarray of int
        Size of each cluster.
    """
    return np.bincount(cluster_labels(bonds, n_particles))


def mean_cluster_size(sizes):
    """Weight-average cluster size C = sum(s^2 N_s) / sum(s N_s).

    This is the size of the cluster that a randomly chosen particle belongs
    to, as used by Swinkels et al. (2024).

    Parameters
    ----------
    sizes : ndarray of int
        Size of each cluster.

    Returns
    -------
    float
        Mean cluster size C.
    """
    return np.sum(sizes**2) / np.sum(sizes)


def mean_squared_displacement(unwrapped_positions):
    """Mean squared displacement relative to the first frame.

    Parameters
    ----------
    unwrapped_positions : ndarray, shape (F, N, 2)
        Unwrapped particle positions over time.

    Returns
    -------
    ndarray, shape (F,)
        MSD averaged over particles (sigma^2).
    """
    displacement = unwrapped_positions - unwrapped_positions[0]
    return np.mean(np.sum(displacement**2, axis=-1), axis=1)
