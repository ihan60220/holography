import argparse
import json
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def parse_metric_and_derive_equations(config):
    """
    Parses the metric from config, calculates Christoffel symbols,
    and returns a numeric function for the ODE solver.
    """
    print("--- 1. Processing Metric Geometry ---")
    
    # 1. Define Coordinates
    coord_names = config['coords']
    dim = len(coord_names)
    coords = sp.symbols(coord_names)
    print(f"Coordinates: {coords}")

    # 2. Parse Metric Matrix
    # We expect a list of lists of strings, e.g., [["1", "0"], ["0", "r**2"]]
    g_matrix_str = config['metric']
    g_inv_matrix = sp.zeros(dim, dim)
    g_matrix = sp.zeros(dim, dim)

    for i in range(dim):
        for j in range(dim):
            g_matrix[i, j] = sp.sympify(g_matrix_str[i][j])
    
    print("Metric Tensor (g_uv):")
    sp.pprint(g_matrix)

    # 3. Calculate Inverse Metric
    g_inv = g_matrix.inv()

    # 4. Calculate Christoffel Symbols: Gamma^k_ij
    # Gamma^k_ij = 1/2 * g^kl * (d_j g_il + d_i g_jl - d_l g_ij)
    print("\nCalculating Christoffel Symbols...")
    Gamma = [sp.zeros(dim, dim) for _ in range(dim)] # List of matrices, index k is list index

    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                total = 0
                for l in range(dim):
                    term = 0.5 * g_inv[k, l] * (
                        sp.diff(g_matrix[i, l], coords[j]) +
                        sp.diff(g_matrix[j, l], coords[i]) -
                        sp.diff(g_matrix[i, j], coords[l])
                    )
                    total += term
                Gamma[k][i, j] = sp.simplify(total)
    
    print("Christoffel Symbols derived.")

    # 5. Generate Geodesic Equations for Numerical Solver
    # Geodesic Equation: x''_k = -Gamma^k_ij * x'_i * x'_j
    
    # We need a function f(t, y) where y = [x^0...x^n, v^0...v^n]
    # returns [v^0...v^n, a^0...a^n]
    
    # Create symbolic velocity variables
    vels = sp.symbols(f"v_{0}:{dim}")
    
    accelerations = []
    for k in range(dim):
        acc_k = 0
        for i in range(dim):
            for j in range(dim):
                acc_k -= Gamma[k][i, j] * vels[i] * vels[j]
        accelerations.append(acc_k)

    # Lambdify turns symbolic math into fast numpy functions
    # Input args: (t, x_0, ... x_n, v_0, ... v_n)
    #combined_args = (sp.symbols('t'),) + tuple(coords) + tuple(vels)
    # Use 'tau' for the affine parameter to avoid conflict with coordinate 't'
    combined_args = (sp.symbols('tau'),) + tuple(coords) + tuple(vels)
    
    # We create a list of functions: one for each acceleration component
    numeric_acc_funcs = [sp.lambdify(combined_args, acc, 'numpy') for acc in accelerations]

    def ode_system(t, state):
        # State vector is [positions..., velocities...]
        positions = state[:dim]
        velocities = state[dim:]
        
        # Calculate accelerations
        # Unpack all args: t, *positions, *velocities
        args = [t] + list(positions) + list(velocities)
        
        acc_vals = [f(*args) for f in numeric_acc_funcs]
        
        # Return [velocities, accelerations] flattened
        return np.concatenate((velocities, acc_vals))

    return ode_system

def main():
    parser = argparse.ArgumentParser(description="Calculate Geodesics from a Metric Tensor.")
    parser.add_argument('config_file', type=str, help="Path to the JSON configuration file.")
    parser.add_argument('--plot', action='store_true', help="Attempt to plot the result (2D/3D only).")
    args = parser.parse_args()

    # Load Configuration
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    dim = len(config['coords'])
    
    # Derive Physics
    ode_system = parse_metric_and_derive_equations(config)

    # Initial Conditions
    y0 = config['start_pos'] + config['start_vel'] # Concatenate lists
    t_span = (0, config['duration'])
    
    print(f"\n--- 2. Integrating Geodesic ---")
    print(f"Initial State: {y0}")
    print(f"Duration: {config['duration']}")

    # Solve
    sol = solve_ivp(ode_system, t_span, y0, rtol=1e-9, atol=1e-9, t_eval=np.linspace(0, config['duration'], 1000))

    if sol.success:
        print("Integration successful!")
        print(f"Final Position: {sol.y[:dim, -1]}")
    else:
        print("Integration failed.")
        return

    # Visualization
    if args.plot:
        coords = sol.y[:dim]
        if dim == 2:
            plt.figure(figsize=(8, 8))
            plt.plot(coords[0], coords[1], label="Geodesic Path")
            # --- ADD THESE LINES ---
            # If plotting Theta (y) vs Phi (x), fix the Y-axis to see the truth
            if config['coords'][0] == 'theta':
                plt.ylim(0, 3.14159) # Show full range from North to South pole
            # -----------------------
            plt.scatter(coords[0][0], coords[1][0], color='green', label='Start')
            plt.scatter(coords[0][-1], coords[1][-1], color='red', label='End')
            plt.xlabel(config['coords'][0])
            plt.ylabel(config['coords'][1])
            plt.title("Geodesic Trajectory (2D)")
            plt.legend()
            plt.grid(True)
            plt.show()
        elif dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(coords[0], coords[1], coords[2], label="Geodesic Path")
            ax.scatter(coords[0][0], coords[1][0], coords[2][0], color='green', label='Start')
            ax.set_xlabel(config['coords'][0])
            ax.set_ylabel(config['coords'][1])
            ax.set_zlabel(config['coords'][2])
            ax.set_title("Geodesic Trajectory (3D)")
            plt.show()
        else:
            print("Dimension > 3, cannot visualize directly. Saving data to 'geodesic_out.csv'.")
            np.savetxt("geodesic_out.csv", sol.y.T, delimiter=",", header=",".join(config['coords']))

if __name__ == "__main__":
    main()