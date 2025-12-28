# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 17:51:46 2025

@author: martp
"""

"""
ReversibleCV_ClassicPeak_v2.3.1
--------------------------------
Classic reversible cyclic voltammetry simulator (Model A).

Features:
- Semi-infinite diffusion domain
- Global Nernst boundary condition
- Transient peak-shaped CV (Randles–Ševčík regime)
- Depletion layer snapshots for teaching
- Streamlit-ready (no plotting)
- Optional progress callback for Streamlit progress bar

Author: Martin + Copilot
Version: v2.3.1
"""

import numpy as np
from scipy.linalg import solve_banded

F = 96485.3329
R = 8.314462618


# ------------------------------------------------------------
# 1. Generate CV waveform
# ------------------------------------------------------------
def generate_potential_waveform(E_start, E_vertex, E_end, v, dt, t_eq):
    n_eq = int(np.ceil(t_eq / dt))
    E_eq = np.full(n_eq, E_start)

    t_forward = abs(E_vertex - E_start) / v
    n_forward = int(np.ceil(t_forward / dt))
    E_forward = np.linspace(E_start, E_vertex, n_forward, endpoint=False)

    t_reverse = abs(E_vertex - E_end) / v
    n_reverse = int(np.ceil(t_reverse / dt))
    E_reverse = np.linspace(E_vertex, E_end, n_reverse + 1)

    E = np.concatenate([E_eq, E_forward, E_reverse])
    t = np.arange(len(E)) * dt
    return E, t


# ------------------------------------------------------------
# 2. Crank–Nicolson diffusion step
# ------------------------------------------------------------
def crank_nicolson_step(C_old, lam):
    N = len(C_old)
    ab = np.zeros((3, N - 2))
    ab[0, 1:] = -lam / 2.0
    ab[1, :] = 1.0 + lam
    ab[2, :-1] = -lam / 2.0

    rhs = C_old[1:-1].copy()
    rhs += lam / 2.0 * (C_old[2:] - 2.0 * C_old[1:-1] + C_old[:-2])

    C_inner_new = solve_banded((1, 1), ab, rhs)

    C_new = C_old.copy()
    C_new[1:-1] = C_inner_new
    return C_new


# ------------------------------------------------------------
# 3. Classic reversible CV (semi-infinite, global Nernst)
# ------------------------------------------------------------
def simulate_cv_reversible_classic(
    E, t,
    D=4e-11, C_bulk=1.0, A=1.96e-6, n=1,
    E0=0.1, T=298.15, x_max=7e-4, Nx=400,
    snapshot_times=None,
    progress_callback=None
):
    dt = t[1] - t[0]
    dx = x_max / (Nx - 1)
    lam = D * dt / dx**2

    C_red = np.full(Nx, C_bulk)
    C_ox = np.zeros(Nx)

    i = np.zeros_like(t)
    Cred_surf = np.zeros_like(t)
    Cox_surf = np.zeros_like(t)

    beta = n * F / (R * T)
    n_steps = len(t)

    # Snapshot setup
    if snapshot_times is None:
        snapshot_times = []
    snapshot_indices = [np.argmin(np.abs(t - ts)) for ts in snapshot_times]

    snapshots = {
        "times": [],
        "x": None,
        "Cred_profiles": [],
        "Cox_profiles": []
    }

    for k in range(n_steps):

        # Streamlit progress callback
        if progress_callback is not None:
            progress_callback(k, n_steps)

        E_k = E[k]
        exponent = np.clip(beta * (E_k - E0), -50, 50)
        K = np.exp(exponent)

        # Global Nernst: total analyte = C_bulk
        C_red0 = C_bulk / (1.0 + K)
        C_ox0 = C_bulk - C_red0

        # Boundary conditions
        C_red[0] = C_red0
        C_ox[0] = C_ox0
        C_red[-1] = C_bulk
        C_ox[-1] = 0.0

        # Diffusion update
        C_red = crank_nicolson_step(C_red, lam)
        C_ox = crank_nicolson_step(C_ox, lam)

        # Reapply BCs
        C_red[0] = C_red0
        C_ox[0] = C_ox0
        C_red[-1] = C_bulk
        C_ox[-1] = 0.0

        # Surface concentrations
        Cred_surf[k] = C_red[0]
        Cox_surf[k] = C_ox[0]

        # Flux and current
        dCred_dx_0 = (C_red[1] - C_red[0]) / dx
        j_red = -D * dCred_dx_0
        i[k] = -n * F * A * j_red

        # Save snapshots
        if k in snapshot_indices:
            if snapshots["x"] is None:
                snapshots["x"] = np.linspace(0, x_max, Nx)
            snapshots["times"].append(t[k])
            snapshots["Cred_profiles"].append(C_red.copy())
            snapshots["Cox_profiles"].append(C_ox.copy())

    return E, i, t, Cred_surf, Cox_surf, snapshots


# ------------------------------------------------------------
# 4. Streamlit-friendly wrapper
# ------------------------------------------------------------
def run_classic_cv(
    E_start=-0.2,
    E_vertex=0.4,
    E_end=-0.2,
    v=2.0,
    dt=2e-5,
    t_eq=1.0,
    D=4e-11,
    C_bulk=1.0,
    A=1.96e-6,
    n=1,
    E0=0.1,
    T=298.15,
    x_max=7e-4,
    Nx=400,
    snapshot_times=None,
    progress_callback=None
):
    E, t = generate_potential_waveform(E_start, E_vertex, E_end, v, dt, t_eq)

    return simulate_cv_reversible_classic(
        E, t,
        D=D, C_bulk=C_bulk, A=A, n=n,
        E0=E0, T=T, x_max=x_max, Nx=Nx,
        snapshot_times=snapshot_times,
        progress_callback=progress_callback
    )