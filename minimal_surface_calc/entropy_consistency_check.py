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
r_cutoff = 50.0 # Our UV boundary cutoff (1 / epsilon)

def m_vaidya(v): return (m0 + 1) / 2 * np.tanh(v / vs) + (m0 - 1) / 2
def dm_dv_vaidya(v): return (m0 + 1) / (2 * vs) * (1 / np.cosh(v / vs))**2
def m_btz(v): return m0
def dm_dv_btz(v): return 0.0

# ==========================================
# 2. GEODESIC SOLVER (WITH PROPER LENGTH)
# ==========================================
def geodesic_derivatives(x, state, r_star, mass_func, dm_dv_func):
    # Added L (proper length) as the 5th variable in our state vector
    r, v, rp, vp, L = state
    f_val = r**2 - mass_func(v)
    
    vpp = (r**2 - r**2 * vp**2 + 2 * vp * rp) / r
    df_dr = 2 * r
    df_dv = -dm_dv_func(v)
    
    num = (4 * (r**3 / r_star**2) * rp - 2 * r * rp 
           + df_dr * rp * vp**2 + df_dv * vp**3 
           + 2 * (f_val * vp - rp) * vpp)
    
    if abs(vp) < 1e-10: rpp = f_val * r 
    else: rpp = num / (2 * vp)
    
    # Calculate the proper length integrand: ds = sqrt(g_uv dx^u dx^v)
    # The max(..., 0) protects against tiny negative floating point errors near zero
    Lp = np.sqrt(max(r**2 - f_val * vp**2 + 2 * vp * rp, 0))
        
    return [rp, vp, rpp, vpp, Lp]

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
# 3. THE SHOOTING METHOD (ROOT FINDER)
# ==========================================
def find_geodesic(h_target, v_target, m_func, dm_func, guess_r, guess_v):
    
    def objective(params):
        r_star, v_turn = params
        x0 = 1e-4
        r2 = (r_star**2 - m_func(v_turn)) * r_star
        v2 = r_star
        
        # Initialize L=0 at the turning point
        state0 = [r_star + 0.5*r2*x0**2, v_turn + 0.5*v2*x0**2, r2*x0, v2*x0, 0.0]
        
        def reach_boundary(x, state, r_arg, m_arg, dm_arg): 
            return state[0] - r_cutoff
        reach_boundary.terminal = True
        
        sol = solve_ivp(
            geodesic_derivatives, [x0, 5.0], state0, 
            args=(r_star, m_func, dm_func), events=reach_boundary, 
            rtol=1e-6, atol=1e-6
        )
        
        if sol.status == 1 and len(sol.t_events[0]) > 0:
            x_end = sol.t_events[0][0]
            v_end = sol.y_events[0][0][1]
            return [x_end - h_target, v_end - v_target]
        else:
            return [10.0, 10.0] 

    print(f"Shooting for targets: h={h_target}, v_bdy={v_target}...")
    res = least_squares(objective, x0=[guess_r, guess_v], bounds=([1.05, -2.0], [5.0, 3.0]), ftol=1e-8)
    
    best_r_star, best_v_turn = res.x
    
    x0 = 1e-4
    r2 = (best_r_star**2 - m_func(best_v_turn)) * best_r_star
    v2 = best_r_star
    state0 = [best_r_star + 0.5*r2*x0**2, best_v_turn + 0.5*v2*x0**2, r2*x0, v2*x0, 0.0]
    
    def reach_boundary_final(x, state, r_arg, m_arg, dm_arg): 
        return state[0] - r_cutoff
    reach_boundary_final.terminal = True
    
    sol = solve_ivp(
        geodesic_derivatives, [x0, 5.0], state0, 
        args=(best_r_star, m_func, dm_func), events=reach_boundary_final, 
        rtol=1e-8, atol=1e-8
    )
    
    # We integrated from x=0 to h. Total length is 2x this value.
    half_length = sol.y_events[0][0][4]
    total_length = 2.0 * half_length
    
    return sol, total_length

# ==========================================
# 4. EXECUTE AND COMPARE
# ==========================================
def calculate_and_compare():
    h_target = 1.0     
    v_bdy_target = 1.5 
    
    # 1. Run the Bulk Calculations
    print("\n--- Solving BTZ Geodesic ---")
    sol_btz, L_btz_num = find_geodesic(h_target, v_bdy_target, m_btz, dm_dv_btz, guess_r=1.5, guess_v=1.0)
    
    print("\n--- Solving Vaidya Geodesic ---")
    sol_vai, L_vai_num = find_geodesic(h_target, v_bdy_target, m_vaidya, dm_dv_vaidya, guess_r=1.5, guess_v=0.5)

    # 2. Run the CFT Analytical Calculation
    # Formula for Entanglement Entropy of a 2D CFT in a Thermal State
    L_cft_theo = 2.0 * np.log((2.0 * r_cutoff / np.sqrt(m0)) * np.sinh(h_target * np.sqrt(m0)))

    # 3. Print the Consistency Check
    print("\n==================================================")
    print("      HOLOGRAPHIC ENTANGLEMENT ENTROPY CHECK      ")
    print("==================================================")
    
    print("\n[ STATIC BTZ BLACK HOLE (Thermal State) ]")
    print(f"CFT Analytical Formula:   {L_cft_theo:.8f}")
    print(f"Bulk Bulk Numerical:      {L_btz_num:.8f}")
    print(f"Consistency Error:        {abs(L_cft_theo - L_btz_num):.8e}")
    
    print("\n[ DYNAMIC VAIDYA-ADS (Quantum Quench) ]")
    print("CFT Analytical Formula:   [No closed-form expression exists!]")
    print(f"Bulk Bulk Numerical:      {L_vai_num:.8f}")
    
    if L_vai_num < L_btz_num:
        print("\nPhysics Insight: The Vaidya length is SMALLER than the BTZ length.")
        print("This proves the CFT boundary subsystem has not yet fully thermalized.")
        print("The entanglement entropy is actively growing as the quench progresses!")
    print("==================================================\n")

if __name__ == "__main__":
    calculate_and_compare()