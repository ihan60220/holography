
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

def get_plot_coords(x, r, v):
    R = np.arctan(r)
    xf = np.concatenate([-x[::-1], x])
    rf = np.concatenate([R[::-1], R])
    vf = np.concatenate([v[::-1], v])
    return rf * np.cos(xf), rf * np.sin(xf), vf

# --- PRECOMPUTE GEODESICS ---
v_turn = 0.5
r_cut = 50.0
r_star_values = np.linspace(1.1, 15.0, 100)

print(f"Precomputing {len(r_star_values)} geodesic pairs for slider...")

frames = []
# We'll store the data for the initial plot
initial_data = None

for i, r_star in enumerate(r_star_values):
    if (i+1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(r_star_values)}...")

    # 1. HoloML
    traj_h = holoml_integrate(r_star, v_turn, n_steps=10000, dt=0.005)
    r_h, x_h, v_h = np.array(traj_h[:, 1]), np.array(traj_h[:, 2]), np.array(traj_h[:, 0])
    hit_h = np.where(r_h >= r_cut)[0]
    if len(hit_h) > 0:
        idx = hit_h[0]
        Xh, Yh, Zh = get_plot_coords(x_h[:idx+1], r_h[:idx+1], v_h[:idx+1])
    else:
        Xh, Yh, Zh = get_plot_coords(x_h, r_h, v_h)

    # 2. Local
    x0 = 1e-4
    r2_init = (r_star**2 - m_vaidya(v_turn)) * r_star
    v2_init = r_star
    state0 = [r_star + 0.5*r2_init*x0**2, v_turn + 0.5*v2_init*x0**2, r2_init*x0, v2_init*x0]
    
    sol_l = solve_ivp(local_derivatives, [x0, 2.0], state0, args=(r_star,), 
                      events=lambda x, s, rs: s[0] - r_cut, rtol=1e-8, atol=1e-8)
    Xl, Yl, Zl = get_plot_coords(sol_l.t, sol_l.y[0], sol_l.y[1])

    # Create Frame
    frame = go.Frame(
        data=[
            go.Scatter3d(x=Xh, y=Yh, z=Zh, mode='lines'),
            go.Scatter3d(x=Xl, y=Yl, z=Zl, mode='lines')
        ],
        name=f"r{r_star:.2f}",
        traces=[0, 1] # Explicitly target the first two traces
    )
    frames.append(frame)
    
    if i == 0:
        initial_data = [
            go.Scatter3d(x=Xh, y=Yh, z=Zh, mode='lines', name='HoloML (Affine)', line=dict(color='red', width=8)),
            go.Scatter3d(x=Xl, y=Yl, z=Zl, mode='lines', name='Local (Coord x)', line=dict(color='cyan', width=4))
        ]

# --- BUILD FIGURE ---
# We must include the Horizon and Boundary in the initial data to keep trace indices consistent
v_range = np.linspace(-0.5, 2.5, 50)
theta_range = np.linspace(0, 2*np.pi, 50)
V_mesh, TH_mesh = np.meshgrid(v_range, theta_range)

# Horizon
Rh_mesh = np.arctan(np.sqrt(np.maximum(m_vaidya(V_mesh), 0)))
horizon_trace = go.Surface(x=Rh_mesh*np.cos(TH_mesh), y=Rh_mesh*np.sin(TH_mesh), z=V_mesh, 
                           colorscale='Reds', opacity=0.15, showscale=False, name='Horizon')

# AdS Boundary (Cylinder at R = pi/2)
Rb = np.pi/2
boundary_trace = go.Surface(x=Rb*np.cos(TH_mesh), y=Rb*np.sin(TH_mesh), z=V_mesh,
                            colorscale=[[0, 'black'], [1, 'black']], opacity=0.05, 
                            showscale=False, name='AdS Boundary', hoverinfo='none')

fig = go.Figure(data=initial_data + [horizon_trace, boundary_trace], frames=frames)

# Slider setup
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Turning Point r*: "},
    pad={"t": 50},
    steps=[dict(
        method="animate",
        args=[[f"r{r_val:.2f}"], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))],
        label=f"{r_val:.2f}"
    ) for r_val in r_star_values]
)]

fig.update_layout(
    title=f"Geodesic Comparison with r* Slider (v_turn={v_turn})",
    scene=dict(
        xaxis=dict(title='X', range=[-1.7, 1.7]),
        yaxis=dict(title='Y', range=[-1.7, 1.7]),
        zaxis=dict(title='v', range=[-0.5, 2.5]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.8)
    ),
    sliders=sliders,
    width=1000,
    height=800
)

fig.write_html("interactive_plots/slider_compare_parameterizations.html")
print("\nSuccess! Saved slider plot to 'interactive_plots/slider_compare_parameterizations.html'")
