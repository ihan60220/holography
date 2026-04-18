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

def process_sol(sol, m_func):
    x_full = np.concatenate((-sol.t[::-1], sol.t))
    r_full = np.concatenate((sol.y[0][::-1], sol.y[0]))
    v_full = np.concatenate((sol.y[1][::-1], sol.y[1]))
    t_full = map_to_normal_time(v_full, r_full, m_func)
    R_comp = np.arctan(r_full)
    return R_comp * np.cos(x_full), R_comp * np.sin(x_full), v_full, t_full

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
        
        def reach_boundary(x, state, r_arg, m_arg, dm_arg): return state[0] - r_cutoff
        reach_boundary.terminal = True
        
        sol = solve_ivp(geodesic_derivatives, [x0, 5.0], state0, args=(r_star, m_func, dm_func), events=reach_boundary, rtol=1e-6, atol=1e-6)
        if sol.status == 1 and len(sol.t_events[0]) > 0:
            return [sol.t_events[0][0] - h_target, sol.y_events[0][0][1] - v_target]
        return [10.0, 10.0] 

    res = least_squares(objective, x0=[guess_r, guess_v], bounds=([1.001, -2.0], [5.0, 3.0]), ftol=1e-6)
    best_r_star, best_v_turn = res.x
    
    x0 = 1e-4
    r2 = (best_r_star**2 - m_func(best_v_turn)) * best_r_star
    v2 = best_r_star
    state0 = [best_r_star + 0.5*r2*x0**2, best_v_turn + 0.5*v2*x0**2, r2*x0, v2*x0]
    
    def reach_boundary_final(x, state, r_arg, m_arg, dm_arg): return state[0] - r_cutoff
    reach_boundary_final.terminal = True
    sol = solve_ivp(geodesic_derivatives, [x0, 5.0], state0, args=(best_r_star, m_func, dm_func), events=reach_boundary_final, rtol=1e-8, atol=1e-8)
    
    return sol, best_r_star, best_v_turn

# ==========================================
# 4. BUILD THE INTERACTIVE PLOT
# ==========================================
def create_interactive_plot():
    v_bdy_target = 1.5 
    
    # 10 frames for smooth slider animation
    h_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    print(f"Precomputing {len(h_values)} frames. This will take about 20-30 seconds...\n")
    
    frames = []
    slider_steps = []
    init_traces = []

    for i, h in enumerate(h_values):
        print(f"--- Solving for Width h = {h} ---")
        # Dynamic guess: smaller h means it doesn't dive as deep (r_* is larger)
        g_r = 1.05 + (1.3 - h) * 1.5 
        
        sol_btz, _, _ = find_geodesic(h, v_bdy_target, m_btz, dm_dv_btz, guess_r=g_r, guess_v=1.0)
        sol_vai, _, _ = find_geodesic(h, v_bdy_target, m_vaidya, dm_dv_vaidya, guess_r=g_r, guess_v=0.5)

        X_b, Y_b, v_b, t_b = process_sol(sol_btz, m_btz)
        X_v, Y_v, v_v, t_v = process_sol(sol_vai, m_vaidya)

        R_bdy = np.arctan(50.0)
        target_x = [R_bdy * np.cos(h), R_bdy * np.cos(-h)]
        target_y = [R_bdy * np.sin(h), R_bdy * np.sin(-h)]
        t_bdy_btz = v_bdy_target - (1.0/(2*np.sqrt(m0)))*np.log((50.0-np.sqrt(m0))/(50.0+np.sqrt(m0)))

        # Define the dynamic traces for this frame
        frame_data = [
            go.Scatter3d(x=X_b, y=Y_b, z=v_b, mode='lines', line=dict(color='orange', width=6, dash='dash'), name='BTZ'),
            go.Scatter3d(x=X_v, y=Y_v, z=v_v, mode='lines', line=dict(color='blue', width=6), name='Vaidya'),
            go.Scatter3d(x=X_b, y=Y_b, z=t_b, mode='lines', line=dict(color='orange', width=6, dash='dash'), showlegend=False),
            go.Scatter3d(x=X_v, y=Y_v, z=t_v, mode='lines', line=dict(color='blue', width=6), showlegend=False),
            go.Scatter3d(x=target_x, y=target_y, z=[v_bdy_target, v_bdy_target], mode='markers', marker=dict(color='red', size=6, symbol='diamond'), name='Targets'),
            go.Scatter3d(x=target_x, y=target_y, z=[t_bdy_btz, t_bdy_btz], mode='markers', marker=dict(color='red', size=6, symbol='diamond'), showlegend=False)
        ]
        
        # Save the first run to initialize the plot
        if i == 0: init_traces = frame_data
            
        # Plotly frames update specific traces by index. 
        # Since we will add 6 static traces to the main figure first, our dynamic traces will be at indices 6 through 11.
        frames.append(go.Frame(data=frame_data, name=str(h), traces=[6, 7, 8, 9, 10, 11]))
        
        slider_steps.append(dict(
            method="animate",
            args=[[str(h)], dict(mode="immediate", frame=dict(duration=200, redraw=True), transition=dict(duration=200))],
            label=str(h)
        ))

    print("\nAssembling Plot...")
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]], subplot_titles=('Finkelstein (v)', 'Normal Time (t)'))

    # --- 1. Add Static Traces (Indices 0 to 5) ---
    v_mesh = np.linspace(-1.5, 3.0, 50)
    theta_mesh = np.linspace(0, 2*np.pi, 50)
    V_grid, Theta_grid = np.meshgrid(v_mesh, theta_mesh)
    
    # Boundaries and Centerlines
    X_bound, Y_bound = R_bdy * np.cos(Theta_grid), R_bdy * np.sin(Theta_grid)
    for col in [1, 2]:
        fig.add_trace(go.Surface(x=X_bound, y=Y_bound, z=V_grid, colorscale='Greys', opacity=0.1, showscale=False, hoverinfo='skip'), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-1.5, 3.0], mode='lines', line=dict(color='gray', width=3, dash='dash'), name='Center'), row=1, col=col)

    # Horizons
    R_vai = np.arctan(np.sqrt(np.maximum(m_vaidya(V_grid), 0)))
    R_btz_surf = np.arctan(np.sqrt(m0))
    fig.add_trace(go.Surface(x=R_vai*np.cos(Theta_grid), y=R_vai*np.sin(Theta_grid), z=V_grid, colorscale='Blues', opacity=0.2, showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(x=R_btz_surf*np.cos(Theta_grid), y=R_btz_surf*np.sin(Theta_grid), z=V_grid, colorscale='Oranges', opacity=0.2, showscale=False), row=1, col=1)

    # --- 2. Add Initial Dynamic Traces (Indices 6 to 11) ---
    fig.add_trace(init_traces[0], row=1, col=1) # BTZ v
    fig.add_trace(init_traces[1], row=1, col=1) # Vaidya v
    fig.add_trace(init_traces[2], row=1, col=2) # BTZ t
    fig.add_trace(init_traces[3], row=1, col=2) # Vaidya t
    fig.add_trace(init_traces[4], row=1, col=1) # Target v
    fig.add_trace(init_traces[5], row=1, col=2) # Target t

    # --- 3. Attach Frames and Slider ---
    fig.frames = frames
    
    fig.update_layout(
        sliders=[dict(
            active=0, yanchor="top", xanchor="left", currentvalue=dict(font=dict(size=16), prefix="Subsystem Width (h): "),
            transition=dict(duration=200, easing="cubic-in-out"), pad=dict(b=10, t=50), len=0.9, x=0.1, y=0, steps=slider_steps
        )],
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="right", yanchor="top", pad=dict(t=50, r=10),
            buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)])]
        )],
        title_text="Dynamic RG Flow Visualizer (10 Frames)",
        scene1=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='v', aspectmode='manual', aspectratio=dict(x=1,y=1,z=0.7), xaxis=dict(range=[-1.6,1.6]), yaxis=dict(range=[-1.6,1.6])),
        scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='t', aspectmode='manual', aspectratio=dict(x=1,y=1,z=0.7), xaxis=dict(range=[-1.6,1.6]), yaxis=dict(range=[-1.6,1.6])),
        width=1300, height=750, margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.write_html("slider_boundary_10frames.html")
    print("\nSaved as 'slider_boundary_10frames.html'. Open via WSL: explorer.exe slider_boundary_10frames.html")

if __name__ == "__main__":
    create_interactive_plot()