# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:24:55 2026

@author: martp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:06:05 2026

@author: martp
"""

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

from ReversibleCV_ClassicPeak_v2_3_2 import run_classic_cv

st.set_page_config(page_title="Cyclic Voltammetry Simulator", layout="wide")

st.title("🔬 Cyclic Voltammetry Simulator")
st.markdown("### Classic Reversible CV (Model A, v2.3.9)")

# ---------------------------------------------------------
# SIDEBAR — GROUPED LAYOUT
# ---------------------------------------------------------

st.sidebar.header("Simulation Parameters")

# -------------------------------
# ELECTROCHEMICAL PROTOCOL
# -------------------------------
with st.sidebar.expander("Electrochemical Protocol"):
    E_start = st.number_input("Start Potential (V)", value=-0.2)
    E_vertex = st.number_input("Vertex Potential (V)", value=0.4)
    E_end = st.number_input("End Potential (V)", value=-0.2)

    v_user = st.number_input(
        "Scan Rate (V/s)",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01
    )
    v = v_user * 10.0  # internal ×10 scaling

    t_eq = st.number_input("Equilibration Time (s)", value=1.0)
    E0 = st.number_input("Formal Potential (V)", value=0.1)

# -------------------------------
# PHYSICOCHEMICAL PARAMETERS
# -------------------------------
with st.sidebar.expander("Physicochemical Parameters"):
    D = st.number_input(
        "Diffusion Coefficient (m²/s)",
        value=7.0e-10,
        format="%.1e",
        help="Typical diffusion coefficient for ferrocyanide in aqueous solution at room temperature."
    )

    C_bulk_mM = st.number_input("Bulk Concentration (mM)", value=5.0)
    C_bulk = C_bulk_mM * 1.0  # mM → mol/m³

    T = st.number_input("Temperature (K)", value=298.15)

# -------------------------------
# ELECTRODE / GEOMETRY
# -------------------------------
with st.sidebar.expander("Electrode & Geometry"):
    A_cm2 = st.number_input("Electrode Area (cm²)", value=0.126)
    A = A_cm2 * 1e-4  # cm² → m²

# -------------------------------
# NUMERICAL / SIMULATION
# -------------------------------
with st.sidebar.expander("Numerical Simulation Settings"):
    dt = st.number_input(
        "Time Step (s)",
        value=2e-5,
        format="%.1e",
        help="Simulation time increment. Smaller values improve accuracy but increase computation time. Must satisfy dt < (Δx²)/(2D) for stability."
    )

    x_max_um = st.number_input(
        "Domain Size (µm)",
        value=700.0,
        help="Depth of the simulated solution. Should exceed the diffusion layer thickness to avoid boundary artefacts."
    )
    x_max = x_max_um * 1e-6  # µm → m

    Nx = st.number_input(
        "Grid Points",
        value=400,
        help="Number of spatial points used to discretise the diffusion domain. Higher values improve accuracy but increase computation time."
    )

# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------

if st.button("Run Simulation"):

    progress_bar = st.progress(0)

    def progress_callback(k, n_steps):
        progress_bar.progress((k + 1) / n_steps)

    E, i, t, Cred, Cox, snaps = run_classic_cv(
        E_start=E_start,
        E_vertex=E_vertex,
        E_end=E_end,
        v=v,
        dt=dt,
        t_eq=t_eq,
        D=D,
        C_bulk=C_bulk,
        A=A,
        E0=E0,
        T=T,
        x_max=x_max,
        Nx=Nx,
        snapshot_times=[
            t_eq + 0.25 * (abs(E_vertex - E_start) / v),
            t_eq + 0.50 * (abs(E_vertex - E_start) / v),
            t_eq + abs(E_vertex - E_start) / v + 0.5 * (abs(E_vertex - E_end) / v)
        ],
        progress_callback=progress_callback
    )

    st.success("Simulation complete!")

    # Baseline correction at start of scan
    scan_start_index = np.argmin(np.abs(t - t_eq))
    i_corrected = i - i[scan_start_index]

    # Trim equilibration transient: only plot from start of scan onward
    E_plot = E[scan_start_index:]
    i_plot = i_corrected[scan_start_index:]

    # ---------------------------------------------------------
    # TABS
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["Voltammogram", "Surface Concentrations", "Depletion Profiles"])

    # -------------------------------
    # TAB 1 — CYCLIC VOLTAMMOGRAM
    # -------------------------------
    with tab1:
        fig_cv = px.line(
            x=E_plot,
            y=1e6 * i_plot,
            labels={"x": "E (V)", "y": "i (μA)"},
            title="Cyclic Voltammogram (Model A, v2.3.9, baseline-corrected)"
        )

        fig_cv.update_layout(
            xaxis_range=[min(E_plot), max(E_plot)],
            width=600,
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_cv, use_container_width=True, config={"scrollZoom": True})

    # -------------------------------
    # TAB 2 — SURFACE CONCENTRATIONS
    # -------------------------------
    with tab2:
        df_surf = pd.DataFrame({
            "time": t,
            "C_red(0,t)": Cred,
            "C_ox(0,t)": Cox
        })

        fig_surf = px.line(
            df_surf,
            x="time",
            y=["C_red(0,t)", "C_ox(0,t)"],
            labels={"value": "Concentration (mol/m³)", "time": "Time (s)"},
            title="Surface Concentrations vs Time"
        )

        st.plotly_chart(fig_surf, use_container_width=True)

    # -------------------------------
    # TAB 3 — DEPLETION PROFILES
    # -------------------------------
    with tab3:
        if snaps["x"] is not None:
            df_dep = pd.DataFrame({"x": snaps["x"]})

            for idx, profile in enumerate(snaps["Cred_profiles"]):
                df_dep[f"t = {snaps['times'][idx]:.2f} s"] = profile

            fig_dep = px.line(
                df_dep,
                x="x",
                y=df_dep.columns[1:],
                labels={"value": "C_red(x,t)", "x": "x (m)"},
                title="Depletion Profiles at Selected Times"
            )

            st.plotly_chart(fig_dep, use_container_width=True)
        else:
            st.info("No snapshots available.")