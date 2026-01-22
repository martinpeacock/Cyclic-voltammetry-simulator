# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:54:10 2026

@author: martp
"""

"""
ReversibleCV_ClassicPeak_v2_3_2
--------------------------------
Classic reversible cyclic voltammetry simulator (Model A).
Implements 1D diffusion with a correct Nernst boundary condition
and computes current from net Faradaic flux for physical accuracy.
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

    # Spatial grid
    x = np.linspace(0, x_max, Nx)
    dx = x[1] - x[0]

    # Time grid
    t_forward = abs(E_vertex - E_start) / v
    t_reverse = abs(E_end - E_vertex) / v
    total_time = t_eq + t_forward + t_reverse

    Nt = int(total_time / dt) + 1
    t = np.linspace(0, total_time, Nt)

    # Potential waveform
    E = np.zeros(Nt)
    for k in range(Nt):
        tk = t[k]
        if tk < t_eq:
            E[k] = E_start
        elif tk < t_eq + t_forward:
            E[k] = E_start + v * (tk - t_eq)
        else:
            E[k] = E_vertex - v * (tk - (t_eq + t_forward))

    # Initial concentrations
    Cred = np.ones(Nx) * C_bulk
    Cox = np.zeros(Nx)

    # Crank–Nicolson matrices
    alpha = D * dt / (2 * dx * dx)

    ab = np.zeros((3, Nx))
    ab[0, 1:] = -alpha
    ab[1, :] = 1 + 2 * alpha
    ab[2, :-1] = -alpha

    # Storage
    i = np.zeros(Nt)
    Cred_surf = np.zeros(Nt)
    Cox_surf = np.zeros(Nt)

    snaps = {
        "times": [],
        "x": x,
        "Cred_profiles": [],
        "Cox_profiles": []
    }

    # Time stepping
    for k in range(Nt):

        # Nernst boundary
        ratio = np.exp((F / (R * T)) * (E[k] - E0))
        Ctot0 = Cred[0] + Cox[0]
        Cred0 = Ctot0 / (1 + ratio)
        Cox0 = Ctot0 - Cred0

        Cred_surf[k] = Cred0
        Cox_surf[k] = Cox0

        # Net Faradaic current
        dCred_dx = (Cred[1] - Cred[0]) / dx
        dCox_dx  = (Cox[1]  - Cox[0])  / dx

        J_red = -D * dCred_dx
        J_ox  = -D * dCox_dx

        i[k] = F * A * (J_ox - J_red)

        # Diffusion update (CN)
        rhs_r = Cred.copy()
        rhs_r[1:-1] = (
            alpha * Cred[:-2] +
            (1 - 2 * alpha) * Cred[1:-1] +
            alpha * Cred[2:]
        )
        rhs_r[0] = Cred0
        rhs_r[-1] = C_bulk
        Cred = solve_banded((1, 1), ab, rhs_r)

        rhs_o = Cox.copy()
        rhs_o[1:-1] = (
            alpha * Cox[:-2] +
            (1 - 2 * alpha) * Cox[1:-1] +
            alpha * Cox[2:]
        )
        rhs_o[0] = Cox0
        rhs_o[-1] = 0.0
        Cox = solve_banded((1, 1), ab, rhs_o)

        # Snapshots
        if snapshot_times is not None:
            for ts in snapshot_times:
                if abs(t[k] - ts) < dt / 2:
                    snaps["times"].append(t[k])
                    snaps["Cred_profiles"].append(Cred.copy())
                    snaps["Cox_profiles"].append(Cox.copy())

        # Progress callback
        if progress_callback is not None:
            progress_callback(k, Nt)

    return E, i, t, Cred_surf, Cox_surf, snaps