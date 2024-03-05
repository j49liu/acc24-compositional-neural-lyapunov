from dynamical_system import *
from local_verifier import *
from quadratic_verifier import *
from neural_learner import *
from neural_verifier import *
from generate_data import *
from local_gain_verifier import *
from quadratic_gain_verifier import *
from sos_gain_verifier import *
from neural_gain_verifier import *
from plot import *
from utils import *
import time
import numpy as np
from sos_V_func import *

import csv
import os
import uuid
import argparse

parser = argparse.ArgumentParser(description='Provide a unique id to filename.')
parser.add_argument('--job_id', type=str, default=str(uuid.uuid4()), help='Job ID')
args = parser.parse_args()

job_id = args.job_id

np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
np.random.seed(42)  

def generate_network(n, l, m): 
    mu_s = [round(np.random.uniform(0.5, 2.5), 2) for _ in range(n)]
    # mu_s = [1.25]
    # mu_s = [1.25, 2.4]
    # mu_s = [1.25, 2.4, 1.96]
    # mu_s = [1.25, 2.4, 1.96, 1.7]
    # mu_s = [1.25, 2.4, 1.96, 1.7, 0.81]
    # mu_s = [1.25, 2.4, 1.96, 1.7, 0.81, 0.81]
    # mu_s = [1.25, 2.4, 1.96, 1.7, 0.81, 0.81, 0.62]

    print("System parameters: \n", mu_s)
    # [1.25, 2.4, 1.96, 1.7, 0.81, 0.81, 0.62, 2.23, 1.7, 1.92]

    def generate_matrix():
        matrix = np.zeros((n, n))

        for i in range(n):
            num_non_zero = np.random.randint(2, m)
            indices = np.random.choice([j for j in range(n) if j != i], num_non_zero, replace=False)

            for idx in indices:
                matrix[i][idx] = round(np.random.uniform(-l, l), 2)

        return matrix

    interconnection_matrix = generate_matrix()
    print("Interconnection matrix: \n", interconnection_matrix)

    # [[ 0.   -0.03  0.    0.01  0.    0.    0.    0.    0.   -0.08]
    #  [ 0.    0.    0.07  0.    0.    0.   -0.02  0.    0.    0.  ]
    #  [ 0.    0.    0.   -0.16  0.    0.    0.    0.07 -0.02  0.  ]
    #  [ 0.   -0.12  0.    0.    0.03  0.1   0.    0.    0.   -0.03]
    #  [-0.16  0.    0.17  0.    0.    0.    0.    0.    0.    0.  ]
    #  [ 0.02  0.   -0.14 -0.06  0.    0.    0.    0.   -0.09  0.  ]
    #  [ 0.   -0.12  0.12  0.04  0.08  0.    0.    0.    0.    0.  ]
    #  [ 0.    0.   -0.17  0.   -0.08  0.    0.    0.   -0.07  0.  ]
    #  [ 0.    0.09  0.    0.1   0.02  0.    0.    0.    0.    0.11]
    #  [-0.07  0.    0.05  0.   -0.19  0.    0.    0.    0.    0.  ]]

    # Initialize lists to store subsystems and results
    systems = []

    for idx, mu in enumerate(mu_s):
        sys_name = f"van_der_pol_system_{idx+1}_mu_{mu}"
        f_vdp = [
            lambda x1, x2, mu=mu: -x2,
            lambda x1, x2, mu=mu: x1 - mu * (1 - x1**2) * x2
        ]
        domain_vdp = [[-2.5, 2.5], [-5.5, 5.5]]
        subsystem = DynamicalSystem(f_vdp, domain_vdp, sys_name)
        systems.append(subsystem)

    return systems, interconnection_matrix, mu_s

def verify_subsys_quadratic_lyap(systems):
    print("-" * 50)
    print(f"Verifying stability and ROA using quadratic Lyapunov functions...")
    c_P = []

    start_time = time.time()

    for idx, subsystem in enumerate(systems):
        print(f"Processing subsystem {idx+1}...")
        c1 = local_verifier(subsystem)
        c2 = quadratic_verifier(subsystem, c1)
        c_P.append((c1, c2))

    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Time taken for verifying stability and ROA using quadratic Lyapunov functions: {exec_time} seconds.\n")

    return c_P, exec_time

def learn_subsys_neural_lyap(systems, N=None, max_epoch=None, lr=None, 
                            layer=None, width=None, n_samples=None, batch_size=None):
    print("-" * 50)
    print(f"Training neural Lyapunov functions for subsystems...")
    neural_lyap_funcs = []
    model_paths = []
    start_time = time.time()
    for idx, subsystem in enumerate(systems):
        print(f"Processing subsystem {idx+1}...")        
        data = generate_data(subsystem, n_samples=n_samples)
        net, model_path = neural_learner(subsystem, data=data, N=N, max_epoch=max_epoch, batch_size=batch_size, 
                                            lr=lr, layer=layer, width=width)
        print(f"Model loaded: {model_path}")
        neural_lyap_funcs.append(net)
        model_paths.append(model_path)
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Time taken for training all neural Lyapunov functions: {exec_time} seconds.\n")

    return neural_lyap_funcs, model_paths, exec_time

def verify_subsys_neural_lyap(systems, neural_lyap_funcs, c_P): 
    print("-" * 50)
    print(f"Verifying neural Lyapunov functions for subsystems...")
    c_V = []
    start_time = time.time()
    for idx, subsystem in enumerate(systems): 
        print(f"Processing subsystem {idx+1}...")        
        c1_V, c2_V = neural_verifier(subsystem, neural_lyap_funcs[idx], c_P[idx][1])
        c_V.append((c1_V, c2_V))
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Time taken for verifying all neural Lyapunov functions: {exec_time} seconds.\n")
    return c_V, exec_time

def verify_network_local_stability(systems, interconnection_matrix, c_P, bisection = True, trial_scale = 0.04):
    print("-" * 50)
    print(f"Computing local stability of entire network...")
    n = len(systems)
    R = np.zeros((n, n))
    c1_P = np.array([item[0] for item in c_P])

    start_time = time.time()

    def target_stable_for_scale(scale):
        print("Testing scaling factor for local stability: ",scale)
        for i, system in enumerate(systems):
            eig_min_P = np.min(np.linalg.eigvals(systems[i].P))
            eig_max_P = np.max(np.linalg.eigvals(systems[i].P))

            R[i, i] = 2 * self_local_gain_verifier(system, c1_P[i]*scale)/eig_min_P
            R[i, i] = R[i, i] - 1/eig_max_P

            for j in range(n):
                if interconnection_matrix[i, j] != 0:
                    eig_min_Pj = np.min(np.linalg.eigvals(systems[j].P))
                    rij = local_gain_verifier(system, c1_P[i]*scale, systems[j], c1_P[j]*scale, 
                                                interconnection_matrix[i, j])
                    R[i, j] = rij/eig_min_Pj
                    R[i, i] += 2 * rij/eig_min_P

        print("Local gain matrix R:\n ", R)
        eigenvalues = np.linalg.eigvals(R)
        stability = all(eig.real < 0 for eig in eigenvalues)
        print("All eigenvalues of R have negative real parts?", stability)

        verified_c1_P = c1_P * scale
        x = R @ verified_c1_P
        invariance = all(element < 0 for element in x)
        print("All entries of R * c1_P are negative?", invariance)

        if stability and invariance:
            print("Verification successful! Local stability levels verified: ", verified_c1_P)
        return stability and invariance

    if bisection is True:  
        low, high = 0, 1
        best_scale = 0
        accuracy = 1e-2

        while high - low > accuracy:
            mid = (low + high) / 2
            result = target_stable_for_scale(mid)
            if result:
                low = mid
                best_scale = mid
            else: 
                high = mid
    else: 
        best_scale = trial_scale
        result = target_stable_for_scale(best_scale)

    end_time = time.time()
    exec_time = end_time - start_time

    print(f"Scaling factor for which local stability is verified: ", best_scale)    
    print(f"Level sets verified for local stability: ", c1_P * best_scale)    
    print(f"Time taken for verifying local stability using quadratic Lyapunov functions: {exec_time} seconds.\n")

    return c1_P * best_scale, exec_time

def verify_reach(systems, interconnection_matrix, target_levels, initial_levels, 
                            neural_lyap_funcs = None, sos_V = None, MAX_ITERATIONS=20): 
    n = len(systems)
    R = [[0.0 for _ in range(n)] for _ in range(n)]
    total_gains = [0.0 for _ in range(len(systems))]
    c2_V = initial_levels
    c2_V_updated = [True for _ in range(n)]

    start_time = time.time()
    for iteration in range(MAX_ITERATIONS):
        all_below_target = True
        for i, system in enumerate(systems):
            total_gain = 0.0
            for j in range(n):
                if interconnection_matrix[i, j] != 0:
                    if c2_V_updated[i] or c2_V_updated[j]:
                        if neural_lyap_funcs: 
                            R[i][j] = neural_gain_verifier(system, neural_lyap_funcs[i], c2_V[i], 
                                                        systems[j], neural_lyap_funcs[j], c2_V[j])
                        elif sos_V: 
                            R[i][j] = sos_gain_verifier(system, sos_V[i], c2_V[i], systems[j], sos_V[j], c2_V[j])
                        else: 
                            R[i][j] = quadratic_gain_verifier(system, c2_V[i], systems[j], c2_V[j])
                    total_gain += R[i][j] * np.abs(interconnection_matrix[i, j])
            if (total_gain == total_gains[i]) and iteration != 0:
                print(f"Total gain for subsystem {i+1} remained the same. Skipped self gain verification.", flush=True)
                c2_V_updated[i] = False
            else:
                print(f"Total gain for subsystem {i+1}: ", total_gain, flush=True)
                total_gains[i] = total_gain
                if neural_lyap_funcs:
                    new_c2_V = self_neural_gain_verifier(system, neural_lyap_funcs[i], c2_V[i], total_gain)
                elif sos_V:
                    new_c2_V = self_sos_gain_verifier(system, sos_V[i], c2_V[i], total_gain)                   
                else:
                    new_c2_V = self_quadratic_gain_verifier(system, c2_V[i], total_gain)   
                if new_c2_V is not None:
                    c2_V[i] = new_c2_V
                    print(f"Invariance of V<={c2_V[i]} verified for subsystem {i+1}.", flush=True)
                    c2_V_updated[i] = True
                elif iteration == 0:
                    print(f"!!!Invariance cannot be verified for subsystem {i+1} at initial step!!!\n ")
                    return False
                else:
                    c2_V_updated[i] = False

            # Check conditions based on c2_V values
            if c2_V[i] >= target_levels[i]:
                all_below_target = False  # If any value is above target, set flag to false

        print(f"Updated level sets for Lyapunov functions after iteration {iteration+1}: \n", c2_V, flush=True)
        print("Subsystems updated: ", c2_V_updated)
        if all_below_target:  # If all values are below their target levels, break
            print(f"Successful! Verified reachable level sets contained in target set after iteration {iteration+1}. \n", flush=True)
            end_time = time.time()
            print(f"Time taken for successfully verifying reachability: {end_time - start_time} seconds.\n")
            return True
        elif not any(c2_V_updated):
            print(f"Verified reachable level sets cannot be improved after iteration {iteration+1}. \n", flush=True)
            return False
    return False

def verify_quadratic_network_reach(systems, interconnection_matrix, c1_P, c_P, bisection = True, trial_scale = 0.36):  
    print("-" * 50)
    print(f"Verifying reachability using quadratic Lyapunov functions...")
    n = len(systems)
    start_time = time.time()
    c2_P = np.array([item[1] for item in c_P])
    print("Verified level sets of quadratic Lyapunov functions for subsystems: \n", c2_P)
    target_levels = c1_P
    print("Target level set: \n", target_levels)

    def target_reached_for_scale(scale):
        print(f"Testing scale factor for reachability: ", scale)
        initial_levels = c2_P * scale

        print("Initial level sets for quadratic Lyapunov functions:, \n", initial_levels)
        print("Target level set: \n", target_levels)
        print(f"Verifying reachability using quadratic Lyapunov functions...")

        result = verify_reach(systems, interconnection_matrix, target_levels, initial_levels, MAX_ITERATIONS=20)
        return result

    if bisection is True:  
        low, high = 0, 1
        best_scale = 0
        accuracy = 1e-2

        while high - low > accuracy:
            mid = (low + high) / 2
            result = target_reached_for_scale(mid)
            if result:
                low = mid
                best_scale = mid
            else: 
                high = mid
    else: 
        best_scale = trial_scale
        result = target_reached_for_scale(best_scale)
    
    end_time = time.time()
    exec_time = end_time - start_time

    print(f"Scaling factor for which reachability is verified: ", best_scale)    
    print(f"Level sets verified for reachability: ", c2_P * best_scale)    
    print(f"Time taken for verifying reachability using quadratic Lyapunov functions: {exec_time} seconds.\n")
    return c2_P * best_scale, exec_time

def verify_sos_network_reach(systems, interconnection_matrix, c2_P, bisection = True, trial_level = 0.7):  
    print("-" * 50)
    print(f"Verifying reachability using SOS Lyapunov functions...")
    n = len(systems)
    start_time = time.time()
    sos_V = get_sos_V()

    c1_sos_V = []
    for i in range(len(systems)):
        included_level = compute_sos_V_inclusion(systems[i], sos_V[i], c2_P[i], 1.0)
        c1_sos_V.append(included_level)

    target_levels = c1_sos_V
    print("Target level set: \n", target_levels)

    start_time = time.time()
    
    def target_reached_for_scale(scale):
        print(f"Testing scale factor for reachability: ", scale)
        initial_levels = np.ones(n) * scale

        print("Initial level sets for SOS Lyapunov functions:, \n", initial_levels)
        print("Target level set: \n", target_levels)
        print(f"Verifying reachability using SOS Lyapunov functions...")

        result = verify_reach(systems, interconnection_matrix, target_levels, initial_levels, 
                                sos_V = sos_V, MAX_ITERATIONS=20)
        return result

    if bisection is True:  
        low, high = 0, 1
        best_scale = 0
        accuracy = 1e-2

        while high - low > accuracy:
            mid = (low + high) / 2
            result = target_reached_for_scale(mid)
            if result:
                low = mid
                best_scale = mid
            else: 
                high = mid
    else: 
        best_scale = trial_scale
        result = target_reached_for_scale(best_scale)
    
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Scaling factor for which reachability is verified: ", best_scale)    
    print(f"Level sets verified for reachability: ", np.ones(n) * best_scale)    
    print(f"Time taken for verifying reachability using SOS Lyapunov functions: {exec_time} seconds.\n")
    return np.ones(n) * best_scale, exec_time

def compute_lower_bound(systems, neural_lyapunov_functions, c_V): 
    lb = []
    for i in range(len(systems)):
        lower_bound = compute_lower_bound_V(systems[i], neural_lyapunov_functions[i], 
                                c_V[i])
        lb.append(lower_bound)
    print("Lower bounds of neural Lyapunov functions: \n", np.array(lb), flush=True)    

def verify_neural_network_reach(systems, interconnection_matrix, neural_lyapunov_functions, c2_P, c_V, bisection = True, trial_scale = 0.3): 
    print("-" * 50)
    print(f"Verifying reachability using neural Lyapunov functions...")
    n = len(systems)
    start_time = time.time()
    c2_V = np.array([item[1] for item in c_V])
    print("Verified level sets of neural Lyapunov functions for subsystems: \n", c2_V)

    c1_V = []
    for i in range(len(systems)):
        included_level = compute_V_inclusion(systems[i], neural_lyapunov_functions[i], 
                                c2_P[i], c_V[i][1])
        c1_V.append(included_level)

    start_time = time.time()
    target_levels = c1_V
    print("Target level set: \n", target_levels)
    compute_lower_bound(systems, neural_lyapunov_functions, c2_V)

    def target_reached_for_scale(scale):
        print(f"Testing scale factor for reachability: ", scale)
        initial_levels = c2_V * scale

        print("Initial level sets for neural Lyapunov functions:, \n", initial_levels)
        print("Target level set: \n", target_levels)
        print(f"Verifying reachability using neural Lyapunov functions...")

        result = verify_reach(systems, interconnection_matrix, target_levels, initial_levels, 
                            neural_lyap_funcs=neural_lyapunov_functions, MAX_ITERATIONS=20)
        return result

    if bisection is True:  
        low, high = 0, 1
        best_scale = 0
        accuracy = 1e-2

        while high - low > accuracy:
            mid = (low + high) / 2
            result = target_reached_for_scale(mid)
            if result:
                low = mid
                best_scale = mid
            else: 
                high = mid
    else: 
        best_scale = trial_scale
        result = target_reached_for_scale(best_scale)
    
    end_time = time.time()
    exec_time = end_time - start_time

    print(f"Scaling factor for which reachability is verified: ", best_scale)    
    print(f"Level sets verified for reachability: ", c2_V * best_scale)    
    print(f"Time taken for verifying reachability using neural Lyapunov functions: {exec_time} seconds.\n")
    return c2_V * best_scale, exec_time

if __name__ == '__main__':

    N = 300000
    layer = 2
    width = 20
    max_epoch = 20
    n_samples = 3000
    lr = 1e-3
    batch_size = 32

    network_size = 10
    connection_strength = 0.1
    density = 3

    systems, M, mu_s = generate_network(network_size, connection_strength, density)

    c_P, time_c_P = verify_subsys_quadratic_lyap(systems)    
    nets, model_paths, time_train = learn_subsys_neural_lyap(systems, lr = lr, batch_size = batch_size, 
                    N=N, max_epoch=max_epoch, layer=layer, width=width, n_samples=n_samples)
    # c_V, time_c_V = verify_subsys_neural_lyap(systems, nets, c_P)


    # network = f"size_{network_size}_stregth_{connection_strength}_density_{density}"
    # model = f"new_results_N_{N}_layer={layer}_width_{width}_epoch_{max_epoch}_samples_{n_samples}"
    # output_file = job_id + "_" + network + "_" + model + ".csv"

    # write_header = not os.path.exists(output_file)

    print(f"Case network_size={network_size}, connection_strength={connection_strength}, density={density}:")
    print("-" * 100)

    # verified result for 2x30 networks:
    c_V =  [(0.19, 0.79), 
            (0.10, 0.68), 
            (0.12, 0.71), 
            (0.14, 0.72), 
            (0.31, 0.92), 
            (0.30, 0.87), 
            (0.39, 0.91), 
            (0.12, 0.61), 
            (0.14, 0.73), 
            (0.13, 0.67)]

    time_c_V = 0.0

    c1_P, time_c1_P = verify_network_local_stability(systems, M, c_P, bisection=True)
    c2_P, time_c2_P = verify_quadratic_network_reach(systems, M, c1_P, c_P, bisection=True)
    c2_SOS, time_c2_SOS = verify_sos_network_reach(systems, M, c2_P, bisection=True)
    c2_V, time_c2_V = verify_neural_network_reach(systems, M, nets, c2_P, c_V, bisection=True, trial_scale=0.4)
    
    # with open(output_file, 'a', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)

    #     if write_header:
    #         # Write the headers
    #         csvwriter.writerow(['size', 'stength', 'density', 'c_P', 'time_c_P', 'time_train', 'c_V', 'time_c_V', 
    #                             'c1_P', 'time_c1_P', 'c2_P', 'time_c2_P', 'c2_SOS', 'time_c2_SOS',
    #                             'c2_V', 'time_c2_V'])
    #         write_header = False  # So that headers are written only once

    #     # Write the data
    #     csvwriter.writerow([network_size, connection_strength, density, c_P, time_c_P, time_train, c_V, time_c_V, c1_P, 
    #                         time_c1_P, c2_P, time_c2_P, c2_SOS, time_c2_SOS, c2_V, time_c2_V])

    # sos_expr = get_sos_V()
    # sos_V = lambdify_sos_V(sos_expr)

    # # # density = 3
    # c2_P = [1.53, 1.07, 1.17, 1.26, 2.13, 2.13, 2.68, 1.10, 1.26, 1.18]
    # c2_SOS = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    # c2_P_indy = np.array(c2_P) / 0.78125

    # # # # 2x20
    # # # # c2_V = [0.21, 0.18, 0.19, 0.19, 0.24, 0.23, 0.24, 0.16, 0.19, 0.18]

    # # # # 2x30
    # # c2_V = [0.26, 0.22, 0.23, 0.24, 0.30, 0.29, 0.30, 0.20, 0.24, 0.22]
    # # c2_V_indy = [0.79, 0.68, 0.71, 0.72, 0.92, 0.87, 0.91, 0.61, 0.73, 0.67]

    # # # # 3x10

    # c2_V = [0.24, 0.19, 0.19, 0.22, 0.26, 0.27, 0.28, 0.19, 0.21, 0.21]
    # c2_V_indy =  [0.83, 0.66, 0.67, 0.77, 0.90, 0.91, 0.95, 0.66, 0.73, 0.71]

    # # density = 4
    # # c2_P = [1.24, 0.86, 0.95, 1.02, 1.73, 1.73, 2.17, 0.89, 1.02, 0.96]
    # # c2_SOS = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
    # # c2_V = [0.18, 0.14, 0.14, 0.16, 0.19, 0.19, 0.20, 0.14, 0.15, 0.15]

    # plot_all(systems, mu_s, nets, sos_V, c2_SOS=c2_SOS, c2_V=c2_V, c2_V_indy=c2_V_indy, c2_P=c2_P, c2_P_indy=c2_P_indy)

    # for i, subsystem in enumerate(systems):
    #     plot_V(subsystem, nets[i], model_paths[i]+"_"+job_id+"_"+network, sos_V = sos_V[i], c2_SOS = c2_SOS[i], 
    #                             c2_V=c2_V[i], c2_P=c2_P[i])
