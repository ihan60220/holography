import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
import sys
import os

# Add paths to imports
sys.path.append(os.path.abspath("HoloML/Vaidya_AdS/src"))
from Vaidya_AdS import integrate_geodesic as holoml_integrate

# Local version of shooting_mass logic (surgical extraction)
def m_vaidya(v): return (1.0 + 1) / 2 * np.tanh(v / 1.0) + (1.0 - 1) / 2
def dm_dv_vaidya(v): return (1.0 + 1) / (2 * 1.0) * (1 / np.cosh(v / 1.0))**2

def shooting_derivatives(x, state, r_star):
    r, v, rp, vp = state
    f_val = r**2 - m_vaidya(v)
    vpp = (r**2 - r**2 * vp**2 + 2 * vp * rp) / r
    df_dr = 2 * r
    df_dv = -dm_dv_vaidya(v)
    num = (4 * (r**3 / r_star**2) * rp - 2 * r * rp 
           + df_dr * rp * vp**2 + df_dv * vp**3 
           + 2 * (f_val * vp - rp) * vpp)
    rpp = num / (2 * vp) if abs(vp) > 1e-10 else f_val * r
    return [rp, vp, rpp, vpp]

# Comparison Parameters
v_turn = 0.5
r_star = 2.0
r_cut = 50.0

# 1. Integrate using HoloML (Affine Lambda)
# n_steps and dt adjusted to ensure it reaches r_cut
traj_h = holoml_integrate(r_star, v_turn, n_steps=10000, dt=0.005)
r_h = np.array(traj_h[:, 1])
x_h = np.array(traj_h[:, 2])
v_h = np.array(traj_h[:, 0])

# Find boundary hit in HoloML
hit_idx = np.where(r_h >= r_cut)[0][0]
h_holoml = x_h[hit_idx]
v_bdy_holoml = v_h[hit_idx]

# 2. Integrate using Local shooting_mass logic (Coordinate x)
x0 = 1e-4
# Initial conditions from your shooting_mass.py
r2 = (r_star**2 - m_vaidya(v_turn)) * r_star
v2 = r_star
state0 = [r_star + 0.5*r2*x0**2, v_turn + 0.5*v2*x0**2, r2*x0, v2*x0]

def reach_boundary(x, state, r_star): return state[0] - r_cut
reach_boundary.terminal = True

sol_local = solve_ivp(
    shooting_derivatives, [x0, 2.0], state0, 
    args=(r_star,), events=reach_boundary, 
    rtol=1e-10, atol=1e-10
)

h_local = sol_local.t_events[0][0]
v_bdy_local = sol_local.y_events[0][0][1]

print(f"Comparison for Turning Point: v_turn={v_turn}, r_star={r_star}")
print(f"{'Metric':<15} | {'HoloML (Affine)':<18} | {'Local (Coord x)':<18} | {'Abs Diff':<10}")
print("-" * 75)
print(f"{'Boundary h':<15} | {h_holoml:<18.8f} | {h_local:<18.8f} | {abs(h_holoml-h_local):.2e}")
print(f"{'Boundary v':<15} | {v_bdy_holoml:<18.8f} | {v_bdy_local:<18.8f} | {abs(v_bdy_holoml-v_bdy_local):.2e}")

