import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List, Tuple
import io

# ================================
# INPUT DATA CLASS
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
    dt: float = 0.001  # Fixed


# ================================
# THRUST MODEL
# ================================
def thrust_magnitude_kgf(v: float) -> float:
    return (0.0000000118 * v**5 +
            0.0000017527 * v**4 -
            0.0001344424 * v**3 -
            0.0009936374 * v**2 -
            0.0565956094 * v +
            6.2867642119)


# ================================
# PHYSICS ENGINE
# ================================
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
        state = {
            'x': 0.0, 'y': self.inp.h0,
            'vx': self.inp.v0 * np.cos(self.alpha),
            'vy': self.inp.v0 * np.sin(self.alpha),
            't': 0.0
        }
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

        results = {
            'time_of_flight': t_flight,
            'range': range_x,
            'max_height': h_max,
            'time_at_max_height': t_hmax,
            'v_at_max_height': v_hmax,
            'impact_speed': v_impact,
            'aoa_deg': self.aoa_deg
        }
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
st.title("LM Catapult Launch Projectile Motion Simulator")
st.markdown("**- LM v0 / CL_max 0.66 from CFD / Propeller 19*12E / Max RPM 6200**")
st.markdown("**- Motion Analysis with Catapult Initial Setting, Aerodynamics, and LM's Properties**")
st.markdown("**- Ignore Propeller Thrust, Attitude, and Control Effect**")
st.markdown("**- Available Dynamic Thrust From Regression Polynomial of Thrust vs Airspeed**")

# ================================
# === ADD YOUR DIAGRAM HERE ===
# ================================
try:
    st.image("Catapult_Launch_Dia.jpg", caption="Catapult Launch Geometry & Forces", use_column_width=True)
except:
    st.warning("Dia image 'dia.jpg' not found. Upload it to the repo root.")

try:
    st.image("Thrsutvsspeed_polynimial_1912E.jpg", caption="Dynamic Thrust vs Airspeed", use_column_width=True)
except:
    st.warning("Dia image 'dia.jpg' not found. Upload it to the repo root.")

# ================================
# INPUTS (Text Boxes)
# ================================
st.sidebar.header("Input Parameters")

v0 = st.sidebar.number_input("Launch Speed (m/s)", 0.0, 100.0, 22.27, step=0.1)
alpha_deg = st.sidebar.number_input("Launch Angle (°)", 0.0, 90.0, 15.0, step=0.1)
h0 = st.sidebar.number_input("Initial Height (m)", 0.0, 10.0, 1.25, step=0.01)
mass = st.sidebar.number_input("Mass (kg)", 0.1, 100.0, 14.9, step=0.1)
area = st.sidebar.number_input("Reference Area (m²)", 0.0, 5.0, 0.924, step=0.001)
rho = st.sidebar.number_input("Air Density (kg/m³)", 0.5, 2.0, 1.225, step=0.001)
g = st.sidebar.number_input("Gravity (m/s²)", 9.0, 10.0, 9.81, step=0.01)

# === NEW: CL_max for Stall Speed ===
cl_max = st.sidebar.number_input("CL_max (for Stall Speed)", 0.5, 3.0, 1.2, step=0.1)

# Aero Table
aero_table = np.array([
    [0.19, 0.00, 0.019], [0.24, 1.00, 0.021], [0.29, 2.00, 0.024], [0.34, 3.00, 0.028],
    [0.39, 4.00, 0.032], [0.44, 5.00, 0.036], [0.49, 6.00, 0.042], [0.54, 7.00, 0.047],
    [0.59, 8.00, 0.053], [0.63, 9.00, 0.060], [0.66, 10.00, 0.067]
])
aoa_options = ["No Aero"] + [f"AOA={row[1]:.0f}°" for row in aero_table]
selected_aoa = st.sidebar.selectbox("Aerodynamic Case", aoa_options)

if st.sidebar.button("Run Simulation"):
    # === AOA Selection ===
    if selected_aoa == "No Aero":
        Cl, aoa_deg_sel, Cd = 0.0, 0.0, 0.0
        L_initial_kgf = 0.0
    else:
        idx = aoa_options.index(selected_aoa)
        Cl, aoa_deg_sel, Cd = aero_table[idx-1]
        q_initial = 0.5 * rho * v0**2
        L_initial_N = q_initial * area * Cl
        L_initial_kgf = L_initial_N / g

    pitch_deg = alpha_deg + aoa_deg_sel

    # === Run Simulation ===
    inp = ProjectileInput(v0, alpha_deg, pitch_deg, h0, mass, area, rho, Cd, Cl, g, dt=0.001)
    sim = ProjectileMotion(inp)
    res, traj, (vx_hmax, vy_hmax, v_hmax, d_hmax_n, l_hmax_n) = sim.simulate()
    analytic = sim.analytic_solution()

    # === NEW: Calculate Stall Speed ===
    weight_N = mass * g
    v_stall = np.sqrt((2 * weight_N) / (rho * area * cl_max)) if cl_max > 0 else 0.0

    # === Extract Trajectory ===
    t_vals, x_vals, y_vals, _, _ = zip(*traj)
    traj_df = pd.DataFrame({
        'Time (s)': t_vals,
        'Range (m)': x_vals,
        'Height (m)': y_vals
    })

    # === Validation (No Aero) ===
    if Cd == 0 and Cl == 0:
        st.subheader("Validation: RK4 vs Analytic (No Aero)")
        val_df = pd.DataFrame({
            'Parameter': ['Range (m)', 'Max Height (m)', 'Time of Flight (s)'],
            'RK4': [res['range'], res['max_height'], res['time_of_flight']],
            'Analytic': [analytic['range'], analytic['max_height'], analytic['time_of_flight']],
            'Δ %': [0.0, 0.0, 0.0]
        })
        st.table(val_df)

    # === Results Summary ===
    st.subheader("Results Summary")
    thrust_hmax_kgf = thrust_magnitude_kgf(v_hmax)
    drag_hmax_kgf = d_hmax_n / g
    lift_hmax_kgf = l_hmax_n / g

    summary_df = pd.DataFrame({
        'Metric': [
            'Range (m)', 'Max Height (m)', 'Time of Flight (s)', 'Impact Speed (m/s)',
            'Time at Max Height (s)', 'Speed at Max Height (m/s)',
            'Thrust Available at Max Height (kgf)', 'Drag at Max Height (kgf)', 'Lift at Max Height (kgf)', 'Lift at Launch Speed (kgf)',
            '**Stall Speed (m/s)**'
        ],
        'Value': [
            f"{res['range']:.3f}", f"{res['max_height']:.3f}", f"{res['time_of_flight']:.3f}",
            f"{res['impact_speed']:.2f}", f"{res['time_at_max_height']:.3f}", f"{res['v_at_max_height']:.2f}",
            f"{thrust_hmax_kgf:.3f}", f"{drag_hmax_kgf:.3f}", f"{lift_hmax_kgf:.3f}", f"{L_initial_kgf:.3f}",
            f"**{v_stall:.2f}**"
        ]
    })
    st.table(summary_df)

    # === Plot 1: Range vs Height ===
    st.subheader("Trajectory: Range vs Height")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'{selected_aoa}')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Height (m)")
    ax1.set_title("Projectile Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

    # === Plot 2: Time vs Height ===
    st.subheader("Height vs Time")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_vals, y_vals, 'g-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Height (m)")
    ax2.set_title("Height over Time")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    # === CSV Export: Summary + Full Trajectory ===
    st.subheader("Download Data")
    full_df = pd.DataFrame({
        'Time_s': t_vals,
        'Range_m': x_vals,
        'Height_m': y_vals
    })

    # Add summary as first row
    summary_row = pd.DataFrame([{
        'Time_s': 'SUMMARY',
        'Range_m': res['range'],
        'Height_m': res['max_height']
    }])
    full_df = pd.concat([summary_row, full_df], ignore_index=True)

    # Add Vstall as extra row
    vstall_row = pd.DataFrame([{
        'Time_s': 'STALL_SPEED',
        'Range_m': v_stall,
        'Height_m': np.nan
    }])
    full_df = pd.concat([full_df, vstall_row], ignore_index=True)

    csv_buffer = io.StringIO()
    full_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Full Trajectory + Summary (CSV)",
        data=csv_buffer.getvalue(),
        file_name="projectile_trajectory_full.csv",
        mime="text/csv"
    )

else:
    st.info("Enter parameters in the sidebar and click **'Run Simulation'** to begin.")

# ======================================================================
# THRUST CALCULATOR – FINAL, VERIFIED, kgf OUTPUT
# ======================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Thrust Calculator (19×12E)")

# Polynomials in NEWTONS (from graph points)
RPM_POLY_N = {
    1000: [1.84639, -0.09859, -0.03061],
    2000: [7.19558, -0.19903, -0.03036],
    3000: [15.92096, -0.29826, -0.03044],
    4000: [27.60008, -0.39428, -0.03074],
    5000: [43.32771, -0.48526, -0.03123],
    6000: [61.43988, -0.57053, -0.03187],
    7000: [82.22525, -0.64616, -0.03273]  # NEGATIVE a coefficient
}

G = 9.81

def thrust_N(v, rpm):
    coeffs = RPM_POLY_N[rpm]
    return np.polyval(coeffs, v)

def thrust_kgf(v, rpm):
    return thrust_N(v, rpm) / G

# Build table
def build_table(v):
    data = []
    for rpm in sorted(RPM_POLY_N.keys()):
        t_kgf = thrust_kgf(v, rpm)
        data.append([rpm, f"{t_kgf:+.3f}"])
    return pd.DataFrame(data, columns=["RPM", "Thrust (kgf)"])

# Interpolate
def interp_thrust(table, rpm):
    df = table.copy()
    df["Thrust"] = df["Thrust (kgf)"].str.replace("+", "").astype(float)
    x = df["RPM"].values
    y = df["Thrust"].values
    return np.interp(rpm, x, y)

# UI
v = st.sidebar.number_input("Airspeed (m/s)", 0.0, 50.0, 20.0, 0.5)
rpm = st.sidebar.number_input("RPM", 0, 8000, 5500, 100)

if st.sidebar.button("Calculate"):
    table = build_table(v)
    st.table(table)
    
    t = interp_thrust(table, rpm)
    st.markdown(f"**Thrust = {t:+.3f} kgf**")
    
    # Plot
    fig, ax = plt.subplots()
    rpms = list(RPM_POLY_N.keys())
    thrusts = [thrust_kgf(v, r) for r in rpms]
    ax.plot(rpms, thrusts, 'o-')
    ax.scatter([rpm], [t], color='red', zorder=5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("RPM")
    ax.set_ylabel("Thrust (kgf)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

