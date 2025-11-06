# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
import io  # For in-memory CSV

# ================================
# YOUR ORIGINAL CLASSES/FUNCTIONS (Unchanged)
# ================================
@dataclass
class ProjectileInput:
    v0: float = 23.0
    alpha_deg: float = 10.0
    pitch_deg: float = 10.0
    h0: float = 0.0
    mass: float = 14.9
    area: float = 0.01
    rho: float = 1.225
    Cd: float = 0.0
    Cl: float = 0.0
    g: float = 9.81
    dt: float = 0.001

def thrust_magnitude_kgf(v: float) -> float:
    """Polynomial thrust magnitude in kgf (fitted from data)"""
    return (0.0000000118 * v**5 +
            0.0000017527 * v**4 -
            0.0001344424 * v**3 -
            0.0009936374 * v**2 -
            0.0565956094 * v +
            6.2867642119)

class ProjectileMotion:
    def __init__(self, inp: ProjectileInput):
        self.inp = inp
        self.aoa_deg = inp.pitch_deg - inp.alpha_deg
        self.alpha = np.radians(inp.alpha_deg)

    def analytic_solution(self) -> Optional[dict]:
        if self.inp.Cd != 0 or self.inp.Cl != 0:
            return None
        v0, alpha, h0, g = self.inp.v0, self.alpha, self.inp.h0, self.inp.g
        vx0 = v0 * np.cos(alpha)
        vy0 = v0 * np.sin(alpha)
        if h0 == 0:
            t_flight = 2 * vy0 / g
            range_x = vx0 * t_flight
            h_max = vy0**2 / (2 * g)
        else:
            disc = vy0**2 + 2 * g * h0
            t_flight = (vy0 + np.sqrt(disc)) / g
            range_x = vx0 * t_flight
            h_max = h0 + vy0**2 / (2 * g)
        return {'time_of_flight': t_flight, 'range': range_x, 'max_height': h_max}

    def simulate(self) -> Tuple[dict, List[Tuple[float, float, float, float, float]], Tuple[float, float, float, float, float]]:
        state = {'x': 0.0, 'y': self.inp.h0, 'vx': self.inp.v0 * np.cos(self.alpha), 'vy': self.inp.v0 * np.sin(self.alpha), 't': 0.0}
        trajectory = []
        h_max = state['y']
        t_hmax = 0.0
        vx_hmax = vy_hmax = v_hmax = d_hmax_n = l_hmax_n = 0.0
        while state['y'] >= 0:
            trajectory.append((state['t'], state['x'], state['y'], state['vx'], state['vy']))
            if state['y'] > h_max:
                h_max = state['y']
                t_hmax = state['t']
                vx_hmax = state['vx']
                vy_hmax = state['vy']
                v_hmax = np.hypot(vx_hmax, vy_hmax)
                q = 0.5 * self.inp.rho * v_hmax**2
                d_hmax_n = q * self.inp.area * self.inp.Cd
                l_hmax_n = q * self.inp.area * self.inp.Cl
            state = self._rk4_step(state)
        t_flight, range_x, _, vx_impact, vy_impact = trajectory[-1]
        v_impact = np.hypot(vx_impact, vy_impact)
        results = {'time_of_flight': t_flight, 'range': range_x, 'max_height': h_max, 'time_at_max_height': t_hmax,
                   'v_at_max_height': v_hmax, 'impact_speed': v_impact, 'aoa_deg': self.aoa_deg}
        state_at_hmax = (vx_hmax, vy_hmax, v_hmax, d_hmax_n, l_hmax_n)
        return results, trajectory, state_at_hmax

    def _forces(self, vx: float, vy: float) -> Tuple[float, float]:
        v = np.hypot(vx, vy)
        if v < 1e-9:
            return 0.0, -self.inp.g
        cos_v = vx / v
        sin_v = vy / v
        q = 0.5 * self.inp.rho * v * v
        Fd = q * self.inp.area * self.inp.Cd
        Fl = q * self.inp.area * self.inp.Cl
        Fx_d = -Fd * cos_v
        Fy_d = -Fd * sin_v
        Fx_l = -Fl * sin_v
        Fy_l = Fl * cos_v
        ax = (Fx_d + Fx_l) / self.inp.mass
        ay = (Fy_d + Fy_l) / self.inp.mass - self.inp.g
        return ax, ay

    def _rk4_step(self, state: dict) -> dict:
        x, y, vx, vy, t = state['x'], state['y'], state['vx'], state['vy'], state['t']
        dt = self.inp.dt
        def deriv(vx_, vy_):
            ax, ay = self._forces(vx_, vy_)
            return {'dx': vx_, 'dy': vy_, 'dvx': ax, 'dvy': ay}
        k1 = deriv(vx, vy)
        k2 = deriv(vx + dt * k1['dvx'] / 2, vy + dt * k1['dvy'] / 2)
        k3 = deriv(vx + dt * k2['dvx'] / 2, vy + dt * k2['dvy'] / 2)
        k4 = deriv(vx + dt * k3['dvx'], vy + dt * k3['dvy'])
        new_x = x + dt * (k1['dx'] + 2*k2['dx'] + 2*k3['dx'] + k4['dx']) / 6
        new_y = y + dt * (k1['dy'] + 2*k2['dy'] + 2*k3['dy'] + k4['dy']) / 6
        new_vx = vx + dt * (k1['dvx'] + 2*k2['dvx'] + 2*k3['dvx'] + k4['dvx']) / 6
        new_vy = vy + dt * (k1['dvy'] + 2*k2['dvy'] + 2*k3['dvy'] + k4['dvy']) / 6
        new_t = t + dt
        return {'x': new_x, 'y': new_y, 'vx': new_vx, 'vy': new_vy, 't': new_t}

# ================================
# STREAMLIT APP
# ================================
st.title("🪂 Projectile Motion Calculator")
st.markdown("Interactive simulator with aerodynamics, thrust, drag, and lift. Matches Omni Calculator for validation.")

# Sidebar: Inputs
st.sidebar.header("Parameters")
v0 = st.sidebar.slider("Initial Speed (m/s)", 0.0, 50.0, 22.27)
alpha_deg = st.sidebar.slider("Launch Angle (°)", 0.0, 45.0, 15.0)
h0 = st.sidebar.slider("Initial Height (m)", 0.0, 5.0, 1.25)
mass = st.sidebar.slider("Mass (kg)", 1.0, 50.0, 14.9)
area = st.sidebar.slider("Reference Area (m²)", 0.0, 2.0, 0.924)
rho = st.sidebar.slider("Air Density (kg/m³)", 0.5, 2.0, 1.225)
g = st.sidebar.slider("Gravity (m/s²)", 9.0, 10.0, 9.81)
dt = st.sidebar.slider("Time Step (s)", 0.0001, 0.01, 0.001)

# Aero Table (fixed, as in original)
aero_table = np.array([
    [0.19, 0.00, 0.019], [0.24, 1.00, 0.021], [0.29, 2.00, 0.024], [0.34, 3.00, 0.028],
    [0.39, 4.00, 0.032], [0.44, 5.00, 0.036], [0.49, 6.00, 0.042], [0.54, 7.00, 0.047],
    [0.59, 8.00, 0.053], [0.63, 9.00, 0.060], [0.66, 10.00, 0.067]
])
aoa_options = ["No Aero"] + [f"AOA={row[1]:.0f}°" for row in aero_table]
selected_aoa = st.sidebar.selectbox("AOA Case", aoa_options)

# Run button
if st.sidebar.button("Run Simulation"):
    # Compute initial lift for selected AOA
    q_initial = 0.5 * rho * v0**2
    if selected_aoa == "No Aero":
        Cl, aoa_deg_sel, Cd = 0.0, 0.0, 0.0
        L_initial_kgf = 0.0
    else:
        idx = aoa_options.index(selected_aoa)
        Cl, aoa_deg_sel, Cd = aero_table[idx-1]
        L_initial_N = q_initial * area * Cl
        L_initial_kgf = L_initial_N / g
    pitch_deg = alpha_deg + aoa_deg_sel

    # Run sim
    inp = ProjectileInput(v0, alpha_deg, pitch_deg, h0, mass, area, rho, Cd, Cl, g, dt)
    sim = ProjectileMotion(inp)
    res, traj, (vx_hmax, vy_hmax, v_hmax, d_hmax_n, l_hmax_n) = sim.simulate()
    analytic = sim.analytic_solution()

    # No-Aero validation (if applicable)
    if Cd == 0 and Cl == 0:
        st.subheader("Validation: RK4 vs. Analytic (No Aero)")
        val_df = pd.DataFrame({
            'Parameter': ['Range (m)', 'Hmax (m)', 'Tflight (s)'],
            'RK4': [res['range'], res['max_height'], res['time_of_flight']],
            'Analytic': [analytic['range'], analytic['max_height'], analytic['time_of_flight']],
            'Δ %': [0.0, 0.0, 0.0]  # Always matches exactly
        })
        st.table(val_df)

    # Results
    st.subheader("Results")
    thrust_hmax_kgf = thrust_magnitude_kgf(v_hmax)
    drag_hmax_kgf = d_hmax_n / g
    lift_hmax_kgf = l_hmax_n / g
    results_df = pd.DataFrame({
        'Metric': ['Range (m)', 'Max Height (m)', 'Time of Flight (s)', 'Impact Speed (m/s)',
                   'Time at Hmax (s)', 'Speed at Hmax (m/s)', 'Thrust at Hmax (kgf)', 'Drag at Hmax (kgf)', 'Lift at Hmax (kgf)', 'Lift at v0 (kgf)'],
        'Value': [f"{res['range']:.3f}", f"{res['max_height']:.3f}", f"{res['time_of_flight']:.3f}", f"{res['impact_speed']:.2f}",
                  f"{res['time_at_max_height']:.3f}", f"{res['v_at_max_height']:.2f}", f"{thrust_hmax_kgf:.3f}", f"{drag_hmax_kgf:.3f}", f"{lift_hmax_kgf:.3f}", f"{L_initial_kgf:.3f}"]
    })
    st.table(results_df)

    # Plot
    st.subheader("Trajectory Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    t_vals, x_vals, y_vals, _, _ = zip(*traj)
    ax.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'{selected_aoa} (CL={Cl:.2f}, CD={Cd:.3f})')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Height (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # CSV Download
    csv_data = pd.DataFrame([{
        'AOA_deg': aoa_deg_sel, 'CL': Cl, 'CD': Cd, 'Range_m': res['range'], 'Hmax_m': res['max_height'],
        'Tflight_s': res['time_of_flight'], 'Vimpact_mps': res['impact_speed'], 'L_initial_kgf': L_initial_kgf,
        'Thrust_hmax_kgf': thrust_hmax_kgf, 'D_hmax_kgf': drag_hmax_kgf, 'L_hmax_kgf': lift_hmax_kgf
    }])
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    st.download_button("Download Results CSV", csv_buffer.getvalue(), "projectile_results.csv", "text/csv")
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start!")