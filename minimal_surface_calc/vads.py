import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMETERS & MASS FUNCTION
# ==========================================
m0 = 1.0  # Final BTZ mass
vs = 1.0  # Shell thickness / characteristic time

def m(v):
    return (m0 + 1) / 2 * np.tanh(v / vs) + (m0 - 1) / 2

def dm_dv(v):
    return (m0 + 1) / (2 * vs) * (1 / np.cosh(v / vs))**2

# ==========================================
# 2. TENSOR MATH ENGINE
# ==========================================
# Coordinates are indexed as: 0 = v, 1 = r, 2 = x

def g_tensor(v, r):
    g = np.zeros((3, 3))
    g[0, 0] = -(r**2 - m(v))  # g_vv
    g[0, 1] = 1.0             # g_vr
    g[1, 0] = 1.0             # g_rv
    g[2, 2] = r**2            # g_xx
    return g

def dg_tensor(v, r):
    dg = np.zeros((3, 3, 3))
    dg[0, 0, 0] = dm_dv(v)    # d/dv g_vv
    dg[1, 0, 0] = -2 * r      # d/dr g_vv
    dg[1, 2, 2] = 2 * r       # d/dr g_xx
    return dg

def christoffel(v, r):
    g_inv = np.linalg.inv(g_tensor(v, r))
    dg = dg_tensor(v, r)
    
    A = np.zeros((3, 3, 3))
    for sigma in range(3):
        for mu in range(3):
            for nu in range(3):
                A[sigma, mu, nu] = dg[mu, sigma, nu] + dg[nu, sigma, mu] - dg[sigma, mu, nu]
                
    Gamma = 0.5 * np.einsum('rs,smn->rmn', g_inv, A)
    return Gamma

# ==========================================
# 3. DIRECT IVP GEODESIC EQUATION
# ==========================================
def geodesic_ivp(tau, state):
    v, r, x, v_dot, r_dot, x_dot = state
    vel = np.array([v_dot, r_dot, x_dot])
    
    Gamma = christoffel(v, r)
    accel = -np.einsum('rmn,m,n->r', Gamma, vel, vel)
    
    return [v_dot, r_dot, x_dot, accel[0], accel[1], accel[2]]

# ==========================================
# 4. PLOTTING ROUTINE
# ==========================================
def plot_vaidya_geodesics_numerical():
    # expanded v0 values
    v0_values = [-2, -1, -0.5, 0, 0.1, 0.5, 1, 1.5, 2]
    
    # Updated to 3x3 plot
    fig, axes = plt.subplots(3, 3, subplot_kw={'projection': 'polar'}, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, v0 in enumerate(v0_values):
        ax = axes[idx]
        
        mass_v0 = m(v0)
        rh = np.sqrt(mass_v0) if mass_v0 > 0 else 0.0
        
        if rh > 0:
            theta_h = np.linspace(0, 2*np.pi, 100)
            ax.plot(theta_h, [np.arctan(rh)]*100, color='firebrick', linewidth=2)
            
        theta_bdy = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta_bdy, [np.pi/2]*100, color='black', linewidth=2)
        
        r_star_values = [rh + 0.1, rh + 0.5, rh + 1.0, rh + 2.0, rh + 4.0]
        if rh == 0:
            r_star_values = [0.1, 0.5, 1.0, 2.0, 4.0]
            
        for r_star in r_star_values:
            x_dot_initial = 1.0 / r_star
            state0 = [v0, r_star, 0.0, 0.0, 0.0, x_dot_initial]
            
            def reach_boundary(tau, state):
                return state[1] - 100.0 
            reach_boundary.terminal = True
            
            sol = solve_ivp(
                geodesic_ivp, 
                [0, 200],
                state0, 
                events=reach_boundary,
                method='RK45', 
                rtol=1e-8, 
                atol=1e-8
            )
            
            r_sol = sol.y[1]
            x_sol = sol.y[2]
            r_compact = np.arctan(r_sol)
            
            ax.plot(x_sol, r_compact, color='blue', alpha=0.6, linewidth=1.2)
            ax.plot(-x_sol, r_compact, color='blue', alpha=0.6, linewidth=1.2)
            
        ax.set_title(f"v0 = {v0}", va='bottom')
        ax.set_rmax(np.pi/2 * 1.05)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_vaidya_geodesics_numerical()