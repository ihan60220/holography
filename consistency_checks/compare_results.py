
import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp
from functools import partial

# --- USER VERSION (from vads.py) ---
m0 = 1.0
vs = 1.0

def m_user(v):
    return (m0 + 1) / 2 * np.tanh(v / vs) + (m0 - 1) / 2

def dm_dv_user(v):
    return (m0 + 1) / (2 * vs) * (1 / np.cosh(v / vs))**2

def g_tensor(v, r):
    g = np.zeros((3, 3))
    g[0, 0] = -(r**2 - m_user(v))
    g[0, 1] = 1.0
    g[1, 0] = 1.0
    g[2, 2] = r**2
    return g

def dg_tensor(v, r):
    dg = np.zeros((3, 3, 3))
    dg[0, 0, 0] = dm_dv_user(v)
    dg[1, 0, 0] = -2 * r
    dg[1, 2, 2] = 2 * r
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

def geodesic_ivp_user(tau, state):
    v, r, x, v_dot, r_dot, x_dot = state
    vel = np.array([v_dot, r_dot, x_dot])
    Gamma = christoffel(v, r)
    accel = -np.einsum('rmn,m,n->r', Gamma, vel, vel)
    return [v_dot, r_dot, x_dot, accel[0], accel[1], accel[2]]

# --- MARCO VERSION (from Vaidya_AdS.py) ---
def get_mass_and_dmdv_marco(v, m_0=1.0, v_s=1.0):
    z = v / v_s
    t = jnp.tanh(z)
    m = (m_0 + 1.0) / 2.0 * t + (m_0 - 1.0) / 2.0
    sech2 = 1.0 - t**2
    dm_dv = (m_0 + 1.0) / 2.0 * (1.0 / v_s) * sech2
    return m, dm_dv

def get_derivs_marco(state, lam, m_0=1.0, v_s=1.0):
    v, r, x, dv, dr, dx = state
    m, dm_dv = get_mass_and_dmdv_marco(v, m_0=m_0, v_s=v_s)
    f = r**2 - m
    df_dr = 2 * r
    df_dv = -dm_dv
    d_dv = r * dx**2 - r * dv**2
    d_dr = f * d_dv + df_dr * dr * dv + 0.5 * df_dv * dv**2
    d_dx = -2.0 / r * dr * dx
    return jnp.array([dv, dr, dx, d_dv, d_dr, d_dx])

def rk4_step(state, dt):
    k1 = get_derivs_marco(state, 0.0)
    k2 = get_derivs_marco(state + 0.5 * dt * k1, 0.0)
    k3 = get_derivs_marco(state + 0.5 * dt * k2, 0.0)
    k4 = get_derivs_marco(state + dt * k3, 0.0)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# --- COMPARISON ---
v0 = 0.5
r_star = 2.0
r_cut = 100.0

# User Solve
state0_user = [v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star]
def reach_boundary(tau, state): return state[1] - r_cut
reach_boundary.terminal = True
sol_user = solve_ivp(geodesic_ivp_user, [0, 200], state0_user, events=reach_boundary, rtol=1e-10, atol=1e-10)
v_inf_user = sol_user.y[0, -1]
x_inf_user = sol_user.y[2, -1]

# Marco Solve
state_marco = jnp.array([v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star])
dt = 0.001
for _ in range(100000):
    state_marco = rk4_step(state_marco, dt)
    if state_marco[1] >= r_cut:
        break
v_inf_marco = state_marco[0]
x_inf_marco = state_marco[2]

print(f"User: v_inf={v_inf_user:.6f}, x_inf={x_inf_user:.6f}")
print(f"Marco: v_inf={v_inf_marco:.6f}, x_inf={x_inf_marco:.6f}")
print(f"Difference: v_inf={abs(v_inf_user-v_inf_marco):.6e}, x_inf={abs(x_inf_user-x_inf_marco):.6e}")
