import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. PARAMETERS & MASS FUNCTIONS
# ==========================================
m0 = 1.0  
vs = 1.0  

def m_vaidya(v): return (m0 + 1) / 2 * np.tanh(v / vs) + (m0 - 1) / 2
def dm_dv_vaidya(v): return (m0 + 1) / (2 * vs) * (1 / np.cosh(v / vs))**2
def m_btz(v): return m0
def dm_dv_btz(v): return 0.0

# ==========================================
# 2. GEODESIC SOLVER & TRANSFORMATION
# ==========================================
def geodesic_derivatives(x, state, r_star, mass_func, dm_dv_func):
    r, v, rp, vp = state
    f_val = r**2 - mass_func(v)
    
    vpp = (r**2 - r**2 * vp**2 + 2 * vp * rp) / r
    df_dr = 2 * r
    df_dv = -dm_dv_func(v)
    
    num = (4 * (r**3 / r_star**2) * rp - 2 * r * rp 
           + df_dr * rp * vp**2 + df_dv * vp**3 
           + 2 * (f_val * vp - rp) * vpp)
    
    if abs(vp) < 1e-10:
        rpp = f_val * r 
    else:
        rpp = num / (2 * vp)
        
    return [rp, vp, rpp, vpp]

def r_star_btz(r):
    sqm = np.sqrt(m0)
    r_safe = np.maximum(r, sqm + 1e-5)
    return (1.0 / (2 * sqm)) * np.log((r_safe - sqm) / (r_safe + sqm))

def map_to_normal_time(v_array, r_array, m_func):
    t_array = np.zeros_like(v_array)
    for i in range(len(v_array)):
        v, r, m_val = v_array[i], r_array[i], m_func(v_array[i])
        
        if m_val < 1e-4:
            r_star = -1.0 / r
        else:
            sqm = np.sqrt(m_val)
            if r <= sqm + 1e-5: r_star = -10.0 
            else: r_star = (1.0 / (2 * sqm)) * np.log((r - sqm) / (r + sqm))
        t_array[i] = v - r_star
    return t_array

# ==========================================
# 3. BUILD THE PLOT
# ==========================================
def create_perfect_slice_comparison():
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('Finkelstein (v)', 'Normal Time (t)')
    )

    # Calculate Geodesics first so we know exactly which slice to draw
    v0_turn = 1.0 
    r_star = 1.5 
    
    def reach_boundary(x, state, rs, m_func, dm_func): return state[0] - 100.0 
    reach_boundary.terminal = True

    # BTZ Geodesic
    r2_b = (r_star**2 - m_btz(v0_turn)) * r_star
    state0_b = [r_star + 0.5*r2_b*(1e-4)**2, v0_turn + 0.5*r_star*(1e-4)**2, r2_b*1e-4, r_star*1e-4]
    sol_btz = solve_ivp(geodesic_derivatives, [1e-4, 4.0], state0_b, args=(r_star, m_btz, dm_dv_btz), events=reach_boundary, rtol=1e-8, atol=1e-8)
    
    # Vaidya Geodesic
    r2_v = (r_star**2 - m_vaidya(v0_turn)) * r_star
    state0_v = [r_star + 0.5*r2_v*(1e-4)**2, v0_turn + 0.5*r_star*(1e-4)**2, r2_v*1e-4, r_star*1e-4]
    sol_vaidya = solve_ivp(geodesic_derivatives, [1e-4, 4.0], state0_v, args=(r_star, m_vaidya, dm_dv_vaidya), events=reach_boundary, rtol=1e-8, atol=1e-8)
    
    def process_sol(sol, m_func):
        x_full = np.concatenate((-sol.t[::-1], sol.t))
        r_full = np.concatenate((sol.y[0][::-1], sol.y[0]))
        v_full = np.concatenate((sol.y[1][::-1], sol.y[1]))
        t_full = map_to_normal_time(v_full, r_full, m_func)
        R_comp = np.arctan(r_full)
        return R_comp * np.cos(x_full), R_comp * np.sin(x_full), v_full, t_full

    X_b, Y_b, v_b, t_b = process_sol(sol_btz, m_btz)
    X_v, Y_v, v_v, t_v = process_sol(sol_vaidya, m_vaidya)

    # ------------------------------------------------
    # Draw the PERFECT Time Slice
    # ------------------------------------------------
    # The exact time value the BTZ geodesic lives on
    t_exact_slice = v0_turn - r_star_btz(r_star)
    
    theta_mesh = np.linspace(0, 2*np.pi, 50)
    R_comp_vals = np.linspace(np.arctan(np.sqrt(m0) + 0.05), np.pi/2 - 0.01, 50)
    R_comp_grid, Theta_grid = np.meshgrid(R_comp_vals, theta_mesh)
    r_grid = np.tan(R_comp_grid)
    
    X_slice = R_comp_grid * np.cos(Theta_grid)
    Y_slice = R_comp_grid * np.sin(Theta_grid)
    
    # Left: Finkelstein Time Slice Funnel
    Z_slice_v = t_exact_slice + r_star_btz(r_grid)
    fig.add_trace(go.Surface(x=X_slice, y=Y_slice, z=Z_slice_v, colorscale='Purples', opacity=0.5, showscale=False, name='Exact BTZ Time Slice'), row=1, col=1)

    # Right: Normal Time Slice Plane
    Z_slice_t = np.full_like(X_slice, t_exact_slice)
    fig.add_trace(go.Surface(x=X_slice, y=Y_slice, z=Z_slice_t, colorscale='Purples', opacity=0.5, showscale=False, name='Exact BTZ Time Slice'), row=1, col=2)

    # ------------------------------------------------
    # Plot the Geodesics
    # ------------------------------------------------
    # Left
    fig.add_trace(go.Scatter3d(x=X_b, y=Y_b, z=v_b, mode='lines', line=dict(color='orange', width=8), name='BTZ Geodesic'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=X_v, y=Y_v, z=v_v, mode='lines', line=dict(color='blue', width=8), name='Vaidya Geodesic'), row=1, col=1)

    # Right
    fig.add_trace(go.Scatter3d(x=X_b, y=Y_b, z=t_b, mode='lines', line=dict(color='orange', width=8), name='BTZ Geodesic'), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=X_v, y=Y_v, z=t_v, mode='lines', line=dict(color='blue', width=8), name='Vaidya Geodesic'), row=1, col=2)

    # ------------------------------------------------
    # Formatting
    # ------------------------------------------------
    scene_config = dict(xaxis_title='X', yaxis_title='Y', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6))
    
    fig.update_layout(
        title_text="Geodesics Intersecting the Exact Foliation",
        scene1=dict(**scene_config, zaxis_title='Advanced Time (v)'),
        scene2=dict(**scene_config, zaxis_title='Normal Time (t)'),
        width=1300, height=700, margin=dict(l=0, r=0, b=0, t=60)
    )
    fig.write_html("perfect_slice.html")
    print("Saved as 'perfect_slice.html'. Open via: explorer.exe perfect_slice.html")

if __name__ == "__main__":
    create_perfect_slice_comparison()