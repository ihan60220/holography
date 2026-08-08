
import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
import plotly.graph_objects as go
import sys
import os


# Ensure we can import HoloML code
sys.path.append(os.path.abspath("HoloML/Vaidya_AdS/src"))
from Vaidya_AdS import integrate_geodesic as holoml_integrate, ds_dlambda

# --- SHARED PHYSICS ---
m0, vs = 1.0, 1.0
def m_vaidya(v): return (m0 + 1) / 2 * np.tanh(v / vs) + (m0 - 1) / 2
def dm_dv_vaidya(v): return (m0 + 1) / (2 * vs) * (1 / np.cosh(v / vs))**2

# --- LOCAL COORDINATE X DERIVATIVES ---
def local_derivatives(x, state, r_star):
    r, v, rp, vp = state
    f_val = r**2 - m_vaidya(v)
    vpp = (r**2 - r**2 * vp**2 + 2 * vp * rp) / r
    df_dr, df_dv = 2 * r, -dm_dv_vaidya(v)
    num = (4 * (r**3 / r_star**2) * rp - 2 * r * rp 
           + df_dr * rp * vp**2 + df_dv * vp**3 + 2 * (f_val * vp - rp) * vpp)
    rpp = num / (2 * vp) if abs(vp) > 1e-10 else f_val * r
    return [rp, vp, rpp, vpp]

def local_proper_length(x_vals, r_vals, v_vals, rp_vals, vp_vals):
    # Standard Vaidya-AdS metric in EF coordinates
    # ds^2 = -f dv^2 + 2 dv dr + r^2 dx^2
    f_vals = r_vals**2 - np.array([m_vaidya(v) for v in v_vals])
    inside = -f_vals * vp_vals**2 + 2.0 * vp_vals * rp_vals + r_vals**2
    ds_dx = np.sqrt(np.maximum(inside, 0.0))
    # Proper length of the full geodesic (both sides)
    return 2.0 * np.trapezoid(ds_dx, x_vals)

# --- RUN COMPARISON ---
v_turn, r_star, r_cut = 0.5, 2.0, 50.0

# 1. HoloML Affine Integration
traj_h = holoml_integrate(r_star, v_turn, n_steps=20000, dt=0.002)
r_h, x_h, v_h = np.array(traj_h[:, 1]), np.array(traj_h[:, 2]), np.array(traj_h[:, 0])
hit_h = np.where(r_h >= r_cut)[0][0]
traj_h_cut = traj_h[:hit_h+1]

# HoloML Length
sdot_h = jax.vmap(ds_dlambda)(traj_h_cut)
L_holoml = 2.0 * 0.002 * (0.5 * sdot_h[0] + np.sum(sdot_h[1:-1]) + 0.5 * sdot_h[-1])

# 2. Local Coordinate Integration
x0 = 1e-4
r2_init = (r_star**2 - m_vaidya(v_turn)) * r_star
v2_init = r_star
state0 = [r_star + 0.5*r2_init*x0**2, v_turn + 0.5*v2_init*x0**2, r2_init*x0, v2_init*x0]

def hit_boundary(x, state, r_star):
    return state[0] - r_cut
hit_boundary.terminal = True

sol_l = solve_ivp(local_derivatives, [x0, 2.0], state0, args=(r_star,), 
                  events=hit_boundary, rtol=1e-10, atol=1e-10)
x_l, r_l, v_l = sol_l.t, sol_l.y[0], sol_l.y[1]
rp_l, vp_l = sol_l.y[2], sol_l.y[3]

L_local = local_proper_length(x_l, r_l, v_l, rp_l, vp_l)

# --- OUTPUT STATS ---
print(f"Comparison: v_turn={v_turn}, r_star={r_star}")
print(f"{'Metric':<15} | {'HoloML (Affine)':<18} | {'Local (Coord x)':<18} | {'Diff'}")
print("-" * 75)
print(f"{'Total Length L':<15} | {L_holoml:<18.8f} | {L_local:<18.8f} | {abs(L_holoml-L_local):.2e}")
print(f"{'Boundary h':<15} | {x_h[hit_h]:<18.8f} | {x_l[-1]:<18.8f} | {abs(x_h[hit_h]-x_l[-1]):.2e}")

# --- PLOTLY VISUALIZATION ---
def get_plot_coords(x, r, v):
    R = np.arctan(r)
    # Full geodesic (both sides)
    xf = np.concatenate([-x[::-1], x])
    rf = np.concatenate([R[::-1], R])
    vf = np.concatenate([v[::-1], v])
    return rf * np.cos(xf), rf * np.sin(xf), vf

Xh, Yh, Zh = get_plot_coords(x_h[:hit_h+1], r_h[:hit_h+1], v_h[:hit_h+1])
Xl, Yl, Zl = get_plot_coords(x_l, r_l, v_l)

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=Xh, y=Yh, z=Zh, mode='lines', name='HoloML (Affine)', line=dict(color='red', width=8)))
fig.add_trace(go.Scatter3d(x=Xl, y=Yl, z=Zl, mode='lines', name='Local (Coord x)', line=dict(color='cyan', width=4)))

v_range = np.linspace(-1, 2, 50)
theta_range = np.linspace(0, 2*np.pi, 50)
V_mesh, TH_mesh = np.meshgrid(v_range, theta_range)
Rh_mesh = np.arctan(np.sqrt(np.maximum(m_vaidya(V_mesh), 0)))
fig.add_trace(go.Surface(x=Rh_mesh*np.cos(TH_mesh), y=Rh_mesh*np.sin(TH_mesh), z=V_mesh, colorscale='Greys', opacity=0.2, showscale=False, name='Horizon'))

fig.update_layout(title="Geodesic Comparison: HoloML vs Local", scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='v'))
fig.write_html("interactive_plots/compare_parameterizations.html")
print("\nSaved interactive plot to 'interactive_plots/compare_parameterizations.html'")
