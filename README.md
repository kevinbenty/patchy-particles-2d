# Bond Dynamics and Network Formation of Patchy Particles in 2D

**Author:** Kevin Benty

**Affiliation:** Chalmers University of Technology, FFR120/FYM119 Simulation of Complex Systems

## Abstract

Patchy particles are colloids whose surfaces carry a small number of attractive sites, or patches. Each patch can form only one bond, so a particle with *n* patches forms at most *n* bonds (limited valency). Despite these simple rules, patchy particles self-assemble into chains and networks: in recent experiments, mixtures of two-patch and three-patch colloids in 2D first formed chains, which were later cross-linked into a system-spanning network (Swinkels et al., 2024). In this project I study how two parameters control this self-assembly in 2D: the patch bond strength ε (in units of k<sub>B</sub>T) and the fraction of three-patch particles in a mixture with two-patch particles. I simulate the system with Brownian dynamics and measure how bonds form and break, and how these microscopic events determine the resulting structure (chains versus networks).

## Methods

- **Model.** N hard-core disks of diameter σ in a periodic 2D box. Each particle carries 2 or 3 point patches at fixed angles on its surface (180° apart for two-patch, 120° apart for three-patch particles). The disks repel each other through a Weeks–Chandler–Andersen (WCA) potential (Weeks et al., 1971). Patches on different particles attract each other through a short-ranged, continuous potential of depth ε. Its range is small enough that each patch can bond to only one other patch, as in the Kern–Frenkel model (Kern & Frenkel, 2003).
- **Dynamics.** Overdamped Langevin (Brownian) dynamics with translational and rotational diffusion, integrated with the Ermak–McCammon scheme (Ermak & McCammon, 1978). The code is written in Python with NumPy and accelerated with Numba (Lam et al., 2015).
- **Bonds and clusters.** Two patches are bonded when their distance is below a cutoff. From the bonds I build clusters and compute the number of bonds per particle Z, the bonded fraction, the cluster-size distribution and the mean cluster size C = Σ s² N<sub>s</sub> / Σ s N<sub>s</sub> (Swinkels et al., 2024), and the bond lifetime.
- **Parameter study.** Sweep over ε and the three-patch fraction, with several independent, seeded runs per point to estimate uncertainties.
- **Validation.** Free particles must reproduce the diffusion law ⟨r²⟩ = 4Dt. The equilibrium bonded fraction is compared with Wertheim theory (Wertheim, 1984), which has been used for patchy-particle mixtures (Bianchi et al., 2006; Russo et al., 2009).
- **Comparison.** Two assembly routes: a sudden quench to large ε versus a slow stepwise increase of ε, as in the experiment of Swinkels et al. (2024).

## References

1. Swinkels, P. J. M., Sinaasappel, R., Gong, Z., Sacanna, S., Meyer, W. V., Sciortino, F., & Schall, P. (2024). Networks of Limited-Valency Patchy Particles. *Physical Review Letters*, 132(7), 078203. https://doi.org/10.1103/PhysRevLett.132.078203
2. Kern, N., & Frenkel, D. (2003). Fluid–fluid coexistence in colloidal systems with short-ranged strongly directional attraction. *The Journal of Chemical Physics*, 118(21), 9882–9889. https://doi.org/10.1063/1.1569473
3. Bianchi, E., Largo, J., Tartaglia, P., Zaccarelli, E., & Sciortino, F. (2006). Phase Diagram of Patchy Colloids: Towards Empty Liquids. *Physical Review Letters*, 97(16), 168301. https://doi.org/10.1103/PhysRevLett.97.168301
4. Russo, J., Tartaglia, P., & Sciortino, F. (2009). Reversible gels of patchy particles: Role of the valence. *The Journal of Chemical Physics*, 131(1), 014504. https://doi.org/10.1063/1.3153843
5. Wertheim, M. S. (1984). Fluids with highly directional attractive forces. I. Statistical thermodynamics. *Journal of Statistical Physics*, 35(1–2), 19–34. https://doi.org/10.1007/BF01017362
6. Ermak, D. L., & McCammon, J. A. (1978). Brownian dynamics with hydrodynamic interactions. *The Journal of Chemical Physics*, 69(4), 1352–1360. https://doi.org/10.1063/1.436761
7. Weeks, J. D., Chandler, D., & Andersen, H. C. (1971). Role of Repulsive Forces in Determining the Equilibrium Structure of Simple Liquids. *The Journal of Chemical Physics*, 54(12), 5237–5247. https://doi.org/10.1063/1.1674820

Software:

8. Lam, S. K., Pitrou, A., & Seibert, S. (2015). Numba: A LLVM-based Python JIT compiler. *Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC*, 1–6. https://doi.org/10.1145/2833157.2833162
