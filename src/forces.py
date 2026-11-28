"""Interactions between patchy particles in 2D.

Two contributions act between each pair of particles i and j:

1. A Weeks-Chandler-Andersen (WCA) repulsion between the particle centres,
   which keeps the disks from overlapping (Weeks et al., 1971).

2. A patch-patch attraction of depth ``epsilon``. It is a smooth version of
   the Kern-Frenkel model (Kern & Frenkel, 2003):

       U_patch = -epsilon * g(r) * f(cos_i) * f(cos_j)

   g(r) is 1 for r <= sigma and goes smoothly to 0 at r = sigma (1 + delta).
   cos_i is the cosine of the angle between a patch direction on particle i
   and the vector pointing from i to j (and the same for j, pointing back).
   f(cos) is 1 when the patch points straight at the other particle and goes
   smoothly to 0 at the patch half-opening angle theta_max.

   Because the patches are narrow, at most one particle fits in front of each
   patch: if sin(theta_max) < sigma / (2 sigma (1 + delta)), two other disks
   cannot both be within range of the same patch (Kern & Frenkel, 2003). Each
   patch can therefore form at most one bond. Being smooth, the potential
   gives finite forces and torques, which Brownian dynamics needs.

Units: lengths in sigma, energies in k_B T.
"""

import numpy as np
from numba import njit

from src.geometry import minimum_image

WCA_CUTOFF = 2.0 ** (1.0 / 6.0)


@njit(cache=True)
def wca_energy_force(distance, wca_epsilon):
    """WCA pair energy and the magnitude of the repulsive force.

    Parameters
    ----------
    distance : float
        Centre-to-centre distance r (units of sigma).
    wca_epsilon : float
        Strength of the repulsion (k_B T).

    Returns
    -------
    energy : float
        Pair energy (zero beyond the cutoff 2^(1/6) sigma).
    force : float
        -dU/dr, positive when the particles are pushed apart.
    """
    if distance >= WCA_CUTOFF:
        return 0.0, 0.0
    inv_r6 = 1.0 / distance**6
    energy = 4.0 * wca_epsilon * (inv_r6 * inv_r6 - inv_r6) + wca_epsilon
    force = 24.0 * wca_epsilon * (2.0 * inv_r6 * inv_r6 - inv_r6) / distance
    return energy, force


@njit(cache=True)
def radial_switch(distance, delta):
    """Radial part g(r) of the patch attraction and its derivative.

    g = 1 for r <= 1, g = cos^2(pi s / 2) with s = (r - 1) / delta for
    1 < r < 1 + delta, and g = 0 beyond.

    Parameters
    ----------
    distance : float
        Centre-to-centre distance r (units of sigma).
    delta : float
        Range of the attraction beyond contact (units of sigma).

    Returns
    -------
    value : float
        g(r).
    derivative : float
        dg/dr.
    """
    if distance <= 1.0:
        return 1.0, 0.0
    if distance >= 1.0 + delta:
        return 0.0, 0.0
    s = (distance - 1.0) / delta
    return np.cos(0.5 * np.pi * s) ** 2, -0.5 * np.pi / delta * np.sin(np.pi * s)


@njit(cache=True)
def angular_switch(cosine, cos_max):
    """Angular part f(cos) of the patch attraction and its derivative.

    f = sin^2(pi u / 2) with u = (cos - cos_max) / (1 - cos_max), so f = 1
    when the patch points exactly at the other particle and f = 0 at the
    patch edge (cos = cos_max) and outside it.

    Parameters
    ----------
    cosine : float
        Cosine of the angle between the patch direction and the bond vector.
    cos_max : float
        Cosine of the patch half-opening angle theta_max.

    Returns
    -------
    value : float
        f(cos).
    derivative : float
        df/dcos.
    """
    if cosine <= cos_max:
        return 0.0, 0.0
    u = (cosine - cos_max) / (1.0 - cos_max)
    return np.sin(0.5 * np.pi * u) ** 2, 0.5 * np.pi / (1.0 - cos_max) * np.sin(np.pi * u)


@njit(cache=True)
def patch_pair_energy(cos_i, cos_j, distance, epsilon, delta, cos_max):
    """Energy of one patch-patch pair, -epsilon g(r) f(cos_i) f(cos_j).

    Parameters
    ----------
    cos_i, cos_j : float
        Patch alignment cosines on particle i and particle j.
    distance : float
        Centre-to-centre distance r.
    epsilon : float
        Patch bond strength (k_B T).
    delta : float
        Range of the attraction beyond contact.
    cos_max : float
        Cosine of the patch half-opening angle.

    Returns
    -------
    float
        Patch-pair energy (<= 0).
    """
    g, _ = radial_switch(distance, delta)
    f_i, _ = angular_switch(cos_i, cos_max)
    f_j, _ = angular_switch(cos_j, cos_max)
    return -epsilon * g * f_i * f_j


@njit(cache=True)
def build_cell_list(positions, box_length, cutoff):
    """Sort particles into square cells of side >= cutoff (linked-cell list).

    Only particles in the same or neighbouring cells can interact, so the
    force calculation costs O(N) instead of O(N^2).

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions inside the box.
    box_length : float
        Side length of the periodic box.
    cutoff : float
        Interaction cutoff distance.

    Returns
    -------
    n_cells : int
        Number of cells per box side.
    head : ndarray of int, shape (n_cells * n_cells,)
        Index of the first particle in each cell (-1 if empty).
    next_in_cell : ndarray of int, shape (N,)
        Index of the next particle in the same cell (-1 at the end).
    """
    n_cells = max(int(box_length / cutoff), 1)
    head = -np.ones(n_cells * n_cells, dtype=np.int64)
    next_in_cell = -np.ones(len(positions), dtype=np.int64)
    for i in range(len(positions)):
        cx = int(positions[i, 0] / box_length * n_cells) % n_cells
        cy = int(positions[i, 1] / box_length * n_cells) % n_cells
        cell = cx * n_cells + cy
        next_in_cell[i] = head[cell]
        head[cell] = i
    return n_cells, head, next_in_cell


@njit(cache=True)
def _add_pair(i, j, positions, orientations, patch_start, valence, patch_offset,
              box_length, epsilon, delta, cos_max, wca_epsilon, forces, torques):
    """Add the forces and torques of pair (i, j) and return its energy."""
    dx = minimum_image(positions[j, 0] - positions[i, 0], box_length)
    dy = minimum_image(positions[j, 1] - positions[i, 1], box_length)
    distance = np.sqrt(dx * dx + dy * dy)
    if distance >= 1.0 + delta and distance >= WCA_CUTOFF:
        return 0.0
    ux, uy = dx / distance, dy / distance  # unit vector from i to j
    energy, f_wca = wca_energy_force(distance, wca_epsilon)
    fx, fy = f_wca * ux, f_wca * uy  # force on j (and -force on i)

    g, dg = radial_switch(distance, delta)
    if g > 0.0:
        for k in range(patch_start[i], patch_start[i] + valence[i]):
            angle_i = orientations[i] + patch_offset[k]
            nx, ny = np.cos(angle_i), np.sin(angle_i)
            cos_i = nx * ux + ny * uy
            f_i, df_i = angular_switch(cos_i, cos_max)
            if f_i == 0.0:
                continue
            for m in range(patch_start[j], patch_start[j] + valence[j]):
                angle_j = orientations[j] + patch_offset[m]
                mx, my = np.cos(angle_j), np.sin(angle_j)
                cos_j = -(mx * ux + my * uy)
                f_j, df_j = angular_switch(cos_j, cos_max)
                if f_j == 0.0:
                    continue
                energy -= epsilon * g * f_i * f_j
                # Gradient of U_patch with respect to the separation vector
                # r = r_j - r_i, using d(cos_i)/dr = (n - cos_i u) / r and
                # d(cos_j)/dr = -(m + cos_j u) / r.
                a = -epsilon * dg * f_i * f_j
                b = -epsilon * g * df_i * f_j / distance
                c = epsilon * g * f_i * df_j / distance
                grad_x = a * ux + b * (nx - cos_i * ux) + c * (mx + cos_j * ux)
                grad_y = a * uy + b * (ny - cos_i * uy) + c * (my + cos_j * uy)
                fx -= grad_x
                fy -= grad_y
                # Torques -dU/dphi, with dn/dphi = (-n_y, n_x).
                torques[i] += epsilon * g * df_i * f_j * (-ny * ux + nx * uy)
                torques[j] += epsilon * g * f_i * df_j * (my * ux - mx * uy)
    forces[j, 0] += fx
    forces[j, 1] += fy
    forces[i, 0] -= fx
    forces[i, 1] -= fy
    return energy


@njit(cache=True)
def compute_forces(positions, orientations, patch_start, valence, patch_offset,
                   box_length, epsilon, delta, cos_max, wca_epsilon=1.0):
    """Total forces, torques and potential energy of the system.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Particle positions inside the box.
    orientations : ndarray, shape (N,)
        Particle orientation angles (rad).
    patch_start : ndarray of int, shape (N,)
        Index of the first patch of each particle in the patch arrays.
    valence : ndarray of int, shape (N,)
        Number of patches of each particle.
    patch_offset : ndarray, shape (M,)
        Patch angles relative to the particle orientation (rad).
    box_length : float
        Side length of the periodic box.
    epsilon : float
        Patch bond strength (k_B T).
    delta : float
        Range of the patch attraction beyond contact (sigma).
    cos_max : float
        Cosine of the patch half-opening angle.
    wca_epsilon : float, optional
        Strength of the WCA repulsion (k_B T).

    Returns
    -------
    forces : ndarray, shape (N, 2)
        Force on each particle (k_B T / sigma).
    torques : ndarray, shape (N,)
        Torque on each particle (k_B T).
    energy : float
        Total potential energy (k_B T).
    """
    n_particles = len(positions)
    forces = np.zeros((n_particles, 2))
    torques = np.zeros(n_particles)
    energy = 0.0
    cutoff = max(1.0 + delta, WCA_CUTOFF)
    n_cells, head, next_in_cell = build_cell_list(positions, box_length, cutoff)

    if n_cells < 3:
        # Box too small for a cell list: loop over all pairs.
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                energy += _add_pair(i, j, positions, orientations, patch_start, valence,
                                    patch_offset, box_length, epsilon, delta, cos_max,
                                    wca_epsilon, forces, torques)
        return forces, torques, energy

    for cx in range(n_cells):
        for cy in range(n_cells):
            i = head[cx * n_cells + cy]
            while i != -1:
                # Visit the cell itself and its 8 neighbours; count each pair once.
                for ox in range(-1, 2):
                    for oy in range(-1, 2):
                        cell = ((cx + ox) % n_cells) * n_cells + (cy + oy) % n_cells
                        j = head[cell]
                        while j != -1:
                            if j > i:
                                energy += _add_pair(i, j, positions, orientations,
                                                    patch_start, valence, patch_offset,
                                                    box_length, epsilon, delta, cos_max,
                                                    wca_epsilon, forces, torques)
                            j = next_in_cell[j]
                i = next_in_cell[i]
    return forces, torques, energy
