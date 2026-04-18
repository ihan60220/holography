import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

# ==========================================
# 1. PARAMETERS & BACKGROUND FUNCTIONS
# ==========================================
m0 = 1.0  # Final BTZ mass
vs = 1.0  # Shell thickness / characteristic collapse time

def m(v):
    return (m0 + 1) / 2 * np.tanh(v / vs) + (m0 - 1) / 2

def dm_dv(v):
    return (m0 + 1) / (2 * vs) * (1 / np.cosh(v / vs))**2

def f(r, v):
    return r**2 - m(v)

# ==========================================
# 2. GEODESIC EQUATIONS 
# ==========================================
def geodesic_derivatives(x, state, r_star):
    r, v, rp, vp = state
    
    vpp = (r**2 - r**2 * vp**2 + 2 * vp * rp) / r
    
    df_dr = 2 * r
    df_dv = -dm_dv(v)
    
    num = (4 * (r**3 / r_star**2) * rp - 2 * r * rp 
           + df_dr * rp * vp**2 + df_dv * vp**3 
           + 2 * (f(r, v) * vp - rp) * vpp)
    
    if abs(vp) < 1e-10:
        rpp = f(r, v) * r 
    else:
        rpp = num / (2 * vp)
        
    return [rp, vp, rpp, vpp]

# ==========================================
# 3. CALCULATIONS & PLOTLY RENDERING
# ==========================================
def create_interactive_3d_geodesic():
    fig = go.Figure()

    # ------------------------------------------------
    # A. Calculate and Plot the Apparent Horizon
    # ------------------------------------------------
    v_vals = np.linspace(-2, 3, 50)
    theta_vals = np.linspace(0, 2*np.pi, 50)
    V_mesh, Theta_mesh = np.meshgrid(v_vals, theta_vals)
    
    Mass_mesh = np.maximum((m0 + 1) / 2 * np.tanh(V_mesh / vs) + (m0 - 1) / 2, 0)
    Rh_mesh = np.sqrt(Mass_mesh)
    
    R_plot = np.arctan(Rh_mesh)
    X_h = R_plot * np.cos(Theta_mesh)
    Y_h = R_plot * np.sin(Theta_mesh)
    Z_h = V_mesh
    
    # Add the horizon as a 3D surface
    fig.add_trace(go.Surface(
        x=X_h, y=Y_h, z=Z_h, 
        colorscale='Reds', 
        opacity=0.5, 
        showscale=False,
        name='Apparent Horizon'
    ))

    # ------------------------------------------------
    # B. Calculate and Plot the Geodesics
    # ------------------------------------------------
    v0 = 1.0 # The time the geodesic is anchored on the boundary
    r_star_values = [0.8, 1.2, 2.0] # Different diving depths
    colors = ['blue', 'cyan', 'lime']
    
    for idx, r_star in enumerate(r_star_values):
        x0 = 1e-4
        r2 = f(r_star, v0) * r_star
        v2 = r_star
        
        state0 = [
            r_star + 0.5 * r2 * x0**2, 
            v0 + 0.5 * v2 * x0**2, 
            r2 * x0, 
            v2 * x0
        ]
        
        def reach_boundary(x, state, r_star):
            return state[0] - 100.0 
        reach_boundary.terminal = True
        
        sol = solve_ivp(
            geodesic_derivatives, [x0, 4.0], state0, args=(r_star,),
            events=reach_boundary, method='RK45', rtol=1e-8, atol=1e-8
        )
        
        r_sol = sol.y[0]
        v_sol = sol.y[1]
        x_sol = sol.t
        
        x_full = np.concatenate((-x_sol[::-1], x_sol))
        r_full = np.concatenate((r_sol[::-1], r_sol))
        v_full = np.concatenate((v_sol[::-1], v_sol))
        
        R_comp = np.arctan(r_full)
        X_geo = R_comp * np.cos(x_full)
        Y_geo = R_comp * np.sin(x_full)
        Z_geo = v_full
        
        # Add the geodesic as a 3D line
        fig.add_trace(go.Scatter3d(
            x=X_geo, y=Y_geo, z=Z_geo,
            mode='lines',
            line=dict(color=colors[idx], width=6),
            name=f'Geodesic (r_* = {r_star})'
        ))

    # ------------------------------------------------
    # C. Layout and Formatting
    # ------------------------------------------------
    fig.update_layout(
        title="Interactive 3D Vaidya-AdS Spacelike Geodesics",
        scene=dict(
            xaxis_title='Boundary X',
            yaxis_title='Boundary Y',
            zaxis_title='Advanced Time (v)',
            zaxis=dict(range=[-2, 3]),
            aspectmode='data' # Keeps the proportions accurate
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Saves the interactive plot as a standalone webpage
    fig.write_html("interactive_geodesics.html")
    print("Success! The interactive plot has been saved as 'interactive_geodesics.html'.")
    print("Double-click this file in your file explorer to open it in your web browser.")

if __name__ == "__main__":
    create_interactive_3d_geodesic()