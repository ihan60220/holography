import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
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
# 2. GEODESIC SOLVER 
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
    
    if abs(vp) < 1e-10: rpp = f_val * r 
    else: rpp = num / (2 * vp)
        
    return [rp, vp, rpp, vpp]

def map_to_normal_time(v_array, r_array, m_func):
    t_array = np.zeros_like(v_array)
    for i in range(len(v_array)):
        v, r, m_val = v_array[i], r_array[i], m_func(v_array[i])
        if m_val < 1e-4: r_star = -1.0 / r
        else:
            sqm = np.sqrt(m_val)
            if r <= sqm + 1e-5: r_star = -10.0 
            else: r_star = (1.0 / (2 * sqm)) * np.log((r - sqm) / (r + sqm))
        t_array[i] = v - r_star
    return t_array

# ==========================================
# 3. THE SHOOTING METHOD
# ==========================================
def find_geodesic(h_target, v_target, m_func, dm_func, guess_r, guess_v):
    r_cutoff = 50.0 
    
    def objective(params):
        r_star, v_turn = params
        x0 = 1e-4
        r2 = (r_star**2 - m_func(v_turn)) * r_star
        v2 = r_star
        
        state0 = [r_star + 0.5*r2*x0**2, v_turn + 0.5*v2*x0**2, r2*x0, v2*x0]
        
        def reach_boundary(x, state, r_arg, m_arg, dm_arg): 
            return state[0] - r_cutoff
        reach_boundary.terminal = True
        
        sol = solve_ivp(
            geodesic_derivatives, [x0, 5.0], state0, 
            args=(r_star, m_func, dm_func), events=reach_boundary, 
            rtol=1e-6, atol=1e-6
        )
        
        if sol.status == 1 and len(sol.t_events[0]) > 0:
            return [sol.t_events[0][0] - h_target, sol.y_events[0][0][1] - v_target]
        else:
            return [10.0, 10.0] 

    print(f"Shooting for targets: h={h_target}, v_bdy={v_target}...")
    
    # Notice the lower bound for r_star is now 0.05 so it can dive extremely deep!
    res = least_squares(
        objective, x0=[guess_r, guess_v], 
        bounds=([0.05, -2.0], [5.0, 3.0]), 
        ftol=1e-6
    )
    
    best_r_star, best_v_turn = res.x
    print(f"  -> Bullseye! Found params: r_*={best_r_star:.4f}, v_turn={best_v_turn:.4f}\n")
    
    x0 = 1e-4
    r2 = (best_r_star**2 - m_func(best_v_turn)) * best_r_star
    v2 = best_r_star
    state0 = [best_r_star + 0.5*r2*x0**2, best_v_turn + 0.5*v2*x0**2, r2*x0, v2*x0]
    
    def reach_boundary_final(x, state, r_arg, m_arg, dm_arg): 
        return state[0] - r_cutoff
    reach_boundary_final.terminal = True
    
    sol = solve_ivp(geodesic_derivatives, [x0, 5.0], state0, args=(best_r_star, m_func, dm_func), events=reach_boundary_final, rtol=1e-8, atol=1e-8)
    return sol, best_r_star, best_v_turn

# ==========================================
# 4. BUILD THE PLOT
# ==========================================
def create_centered_plot():
    # Widened Target to force the geodesic to the center
    h_target = 1.45     
    v_bdy_target = 1.5 
    
    print("--- Solving BTZ Geodesic ---")
    sol_btz, r_btz, v_btz = find_geodesic(h_target, v_bdy_target, m_btz, dm_dv_btz, guess_r=0.2, guess_v=1.0)
    
    print("--- Solving Vaidya Geodesic ---")
    sol_vai, r_vai, v_vai = find_geodesic(h_target, v_bdy_target, m_vaidya, dm_dv_vaidya, guess_r=0.2, guess_v=0.5)

    def process_sol(sol, m_func):
        x_full = np.concatenate((-sol.t[::-1], sol.t))
        r_full = np.concatenate((sol.y[0][::-1], sol.y[0]))
        v_full = np.concatenate((sol.y[1][::-1], sol.y[1]))
        t_full = map_to_normal_time(v_full, r_full, m_func)
        R_comp = np.arctan(r_full)
        return R_comp * np.cos(x_full), R_comp * np.sin(x_full), v_full, t_full

    X_b, Y_b, v_b, t_b = process_sol(sol_btz, m_btz)
    X_v, Y_v, v_v, t_v = process_sol(sol_vai, m_vaidya)

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]], subplot_titles=('Finkelstein (v)', 'Normal Time (t)'))

    # ------------------------------------------------
    # Draw The Boundary, Center Line, and Horizons
    # ------------------------------------------------
    R_bdy = np.arctan(50.0)
    v_mesh = np.linspace(-1.5, 3.0, 50)
    theta_mesh = np.linspace(0, 2*np.pi, 50)
    V_grid, Theta_grid = np.meshgrid(v_mesh, theta_mesh)
    
    # The Outer AdS Boundary Cylinder
    X_bound = R_bdy * np.cos(Theta_grid)
    Y_bound = R_bdy * np.sin(Theta_grid)
    for col in [1, 2]:
        fig.add_trace(go.Surface(x=X_bound, y=Y_bound, z=V_grid, colorscale='Greys', opacity=0.1, showscale=False, name='AdS Boundary', hoverinfo='skip'), row=1, col=col)
        # The exact center line (r=0)
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-1.5, 3.0], mode='lines', line=dict(color='gray', width=3, dash='dash'), name='Spacetime Center (r=0)'), row=1, col=col)

    # Apparent Horizons
    Mass_vai = np.maximum(m_vaidya(V_grid), 0)
    R_vai = np.arctan(np.sqrt(Mass_vai))
    fig.add_trace(go.Surface(x=R_vai * np.cos(Theta_grid), y=R_vai * np.sin(Theta_grid), z=V_grid, colorscale='Blues', opacity=0.2, showscale=False, name='Vaidya Horizon'), row=1, col=1)
    
    R_btz_surf = np.arctan(np.sqrt(m0))
    fig.add_trace(go.Surface(x=R_btz_surf * np.cos(Theta_grid), y=R_btz_surf * np.sin(Theta_grid), z=V_grid, colorscale='Oranges', opacity=0.2, showscale=False, name='BTZ Horizon'), row=1, col=1)

    # ------------------------------------------------
    # Plot the Geodesics
    # ------------------------------------------------
    # Finkelstein (Left)
    fig.add_trace(go.Scatter3d(x=X_b, y=Y_b, z=v_b, mode='lines', line=dict(color='orange', width=6, dash='dash'), name='BTZ Geodesic'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=X_v, y=Y_v, z=v_v, mode='lines', line=dict(color='blue', width=6), name='Vaidya Geodesic'), row=1, col=1)
    
    # Normal Time (Right)
    fig.add_trace(go.Scatter3d(x=X_b, y=Y_b, z=t_b, mode='lines', line=dict(color='orange', width=6, dash='dash'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=X_v, y=Y_v, z=t_v, mode='lines', line=dict(color='blue', width=6), showlegend=False), row=1, col=2)

    # Boundary Targets (Bullseye)
    target_x = [R_bdy * np.cos(h_target), R_bdy * np.cos(-h_target)]
    target_y = [R_bdy * np.sin(h_target), R_bdy * np.sin(-h_target)]
    t_bdy_btz = v_bdy_target - (1.0/(2*np.sqrt(m0)))*np.log((50.0-np.sqrt(m0))/(50.0+np.sqrt(m0)))

    fig.add_trace(go.Scatter3d(x=target_x, y=target_y, z=[v_bdy_target, v_bdy_target], mode='markers', marker=dict(color='red', size=6, symbol='diamond'), name='Boundary Conditions'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=target_x, y=target_y, z=[t_bdy_btz, t_bdy_btz], mode='markers', marker=dict(color='red', size=6, symbol='diamond'), showlegend=False), row=1, col=2)

    # ------------------------------------------------
    # Formatting
    # ------------------------------------------------
    scene_config = dict(xaxis_title='X', yaxis_title='Y', aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.7),
                        xaxis=dict(range=[-1.6, 1.6]), yaxis=dict(range=[-1.6, 1.6]))
    fig.update_layout(
        title_text="Wide Boundary Conditions (Sweeping through the Bulk Center)",
        scene1=dict(**scene_config, zaxis_title='Advanced Time (v)'),
        scene2=dict(**scene_config, zaxis_title='Normal Time (t)'),
        width=1300, height=700, margin=dict(l=0, r=0, b=0, t=60)
    )
    fig.write_html("centered_boundary.html")
    print("Saved as 'centered_boundary.html'. Open via: explorer.exe centered_boundary.html")

if __name__ == "__main__":
    create_centered_plot()