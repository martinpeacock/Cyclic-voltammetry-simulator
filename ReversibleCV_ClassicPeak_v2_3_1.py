# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 19:40:44 2025

@author: martp
"""

"""
ReversibleCV_ClassicPeak_v2.3.1
--------------------------------
Classic reversible cyclic voltammetry simulator (Model A).
Implements 1D diffusion with Nernst boundary condition at x=0.
"""

import numpy as np
from scipy.linalg import solve_banded

F = 96485.3329
R = 8.314462618


def run_classic_cv(
    E_start,
    E_vertex,
    E_end,
    v,
    dt,
    t_eq,
    D,
    C_bulk,
    A,
    E0,
    T,
    x_max,
    Nx,
    snapshot_times,
    progress_callback=None
):
    """
    Main wrapper that runs the reversible CV simulation.

    Parameters:
        E_start (float): Start potential (V)
        E_vertex (float): Vertex potential (V)
        E_end (float): End potential (V)
        v (float): Scan rate (V/s)
        dt (float): Time step (s)
        t_eq (float): Equilibration time at E_start (s)
        D (float): Diffusion coefficient (m²/s)
        C_bulk (float): Bulk concentration (mol/m³)
        A (float): Electrode area (m²)
        E0 (float): Formal potential (V)
        T (float): Temperature (K)
        x_max (float): Spatial domain size (m)
        Nx (int): Number of spatial grid points
        snapshot_times (list[float]): Times at which to store concentration profiles
        progress_callback (callable): Optional, called as progress_callback(k, Nt)

    Returns:
        E (ndarray): Potential waveform (V)
        i (ndarray): Current (A)
        t (ndarray): Time array (s)
        Cred_surf (ndarray): Surface reduced concentration vs time (mol/m³)
        Cox_surf (ndarray): Surface oxidized concentration vs time (mol/m³)
        snaps (dict): Snapshot data with keys:
            - "times": list of snapshot times
            - "x": spatial grid (ndarray)
            - "Cred_profiles": list of Cred(x) arrays
            - "Cox_profiles": list of Cox(x) arrays
    """

    # ------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------
    # Spatial grid
    x = np.linspace(0, x_max, Nx)
    dx = x[1] - x[0]

    # Time grid
    t_forward = abs(E_vertex - E_start) / v
    t_reverse = abs(E_end - E_vertex) / v
    total_time = t_eq + t_forward + t_reverse

    Nt = int(total_time / dt) + 1
    t = np.linspace(0, total_time, Nt)

    # ------------------------------------------------------------
    # Potential waveform
    # ------------------------------------------------------------
    E = np.zeros(Nt)
    for k in range(Nt):
        tk = t[k]
        if tk < t_eq:
            E[k] = E_start
        elif tk < t_eq + t_forward:
            E[k] = E_start + v * (tk - t_eq)
        else:
            E[k] = E_vertex - v * (tk - (t_eq + t_forward))

    # ------------------------------------------------------------
    # Initial concentrations
    # ------------------------------------------------------------
    # Total concentration assumed equal to C_bulk everywhere initially
    C = np.ones(Nx) * C_bulk

    # ------------------------------------------------------------
    # Crank–Nicolson setup
    # ------------------------------------------------------------
    alpha = D * dt / (2 * dx * dx)

    # Banded matrix for CN: main diagonal in middle row
    ab = np.zeros((3, Nx))
    ab[0, 1:] = -alpha         # upper diagonal
    ab[1, :] = 1 + 2 * alpha   # main diagonal
    ab[2, :-1] = -alpha        # lower diagonal

    # ------------------------------------------------------------
    # Storage arrays
    # ------------------------------------------------------------
    i = np.zeros(Nt)
    Cred_surf = np.zeros(Nt)
    Cox_surf = np.zeros(Nt)

    snaps = {
        "times": [],
        "x": x,
        "Cred_profiles": [],
        "Cox_profiles": []
    }

    # ------------------------------------------------------------
    # Time-stepping loop
    # ------------------------------------------------------------
    for k in range(Nt):

        # -----------------------------
        # Nernst boundary at x = 0
        # -----------------------------
        C0_total = C[0]
        ratio = np.exp((F / (R * T)) * (E[k] - E0))
        Cred0 = C0_total / (1 + ratio)
        Cox0 = C0_total - Cred0

        Cred_surf[k] = Cred0
        Cox_surf[k] = Cox0

        # -----------------------------
        # Flux and current (Fick's first law)
        # -----------------------------
        dCdx = (C[1] - C[0]) / dx
        i[k] = -F * A * D * dCdx

        # -----------------------------
        # Right-hand side for CN
        # -----------------------------
        rhs = np.copy(C)
        rhs[1:-1] = (
            alpha * C[:-2] +
            (1 - 2 * alpha) * C[1:-1] +
            alpha * C[2:]
        )

        # Boundary conditions:
        # x = 0: total concentration at surface = Cred0 + Cox0 = C0_total
        rhs[0] = Cred0 + Cox0
        # x = x_max: concentration fixed at bulk
        rhs[-1] = C_bulk

        # -----------------------------
        # Solve CN step
        # -----------------------------
        C = solve_banded((1, 1), ab, rhs)

        # -----------------------------
        # Snapshots
        # -----------------------------
        if snapshot_times is not None:
            for ts in snapshot_times:
                if abs(t[k] - ts) < dt / 2:
                    snaps["times"].append(t[k])
                    snaps["Cred_profiles"].append(C.copy())
                    snaps["Cox_profiles"].append(C_bulk - C.copy())

        # -----------------------------
        # Progress callback
        # -----------------------------
        if progress_callback is not None:
            progress_callback(k, Nt)

    return E, i, t, Cred_surf, Cox_surf, snaps