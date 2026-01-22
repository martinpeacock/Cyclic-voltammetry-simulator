# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:01:32 2026

@author: martp
"""

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

from ReversibleCV_ClassicPeak_v2_3_2 import run_classic_cv

st.set_page_config(page_title="Cyclic Voltammetry Simulator", layout="wide")

st.title("🔬 Cyclic Voltammetry Simulator")
st.markdown("### Classic Reversible CV (Model A, v2.3.7)")

# Sidebar
st.sidebar.header("Simulation Parameters")

E_start = st.sidebar.number_input("Start Potential (V)", value=-0.2)
E_vertex = st.sidebar.number_input("Vertex Potential (V)", value=0.4)
E_end = st.sidebar.number_input("End Potential (V)", value=-0.2)

v = st.sidebar.slider("Scan Rate (V/s)", 0.1, 10.0, 2.0)
dt = st.sidebar.number_input("Time Step (s)", value=2e-5, format="%.1e")
t_eq = st.sidebar.number_input("Equilibration Time (s)", value=1.0)

D = st.sidebar.number_input("Diffusion Coefficient (m²/s)", value=4e-11, format="%.1e")
C_bulk = st.sidebar.number_input("Bulk Concentration (mol/m³)", value=1.0)
A = st.sidebar.number_input("Electrode Area (m²)", value=1.96e-6, format="%.2e")

E0 = st.sidebar.number_input("Formal Potential (V)", value=0.1)
T = st.sidebar.number_input("Temperature (K)", value=298.15)

x_max = st.sidebar.number_input("Domain Size (m)", value=7e-4, format="%.1e")
Nx = st.sidebar.number_input("Grid Points", value=400)

# Run simulation
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

    # Precise baseline correction
    start_index = np.where(E == E_start)[0][0]
    i_corrected = i - i[start_index]

    tab1, tab2, tab3 = st.tabs(["Voltammogram", "Surface Concentrations", "Depletion Profiles"])

    # CV
    with tab1:
        fig_cv = px.line(
            x=E,
            y=1e6 * i_corrected,
            labels={"x": "E (V)", "y": "i (μA)"},
            title="Cyclic Voltammogram (Model A, v2.3.7, baseline-corrected)"
        )

        fig_cv.update_layout(
            xaxis_range=[min(E), max(E)],
            width=600,
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig_cv, use_container_width=True, config={"scrollZoom": True})

    # Surface concentrations
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

    # Depletion profiles
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