import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pytz
import os
from tqdm import tqdm

def generate_data(system, n_samples=100000, overwrite=False, n_nodes=None):
    domain = system.domain
    M = 200
    eps = 1e-7

    T = 10

    def augmented_dynamics(t, z):
        dz_ = [func(*z[:-1]) for func in system.f]
        dz = sum([s**2 for s in z[:-1]])
        return dz_ + [dz]

    def get_train_output(x, z, depth=0):
        if np.linalg.norm(x) <= eps:
            y = np.tanh(20/M*z)
            z_T = z
        elif z >= M:
            y = 1.0
            z_T = z
        else:
            sol = solve_ivp(lambda t, z: augmented_dynamics(t, z), [0, T], [x[0], x[1], z], rtol=1e-6, atol=1e-9)
            x_T = np.array([sol.y[0][-1], sol.y[1][-1]])
            y, z_T = get_train_output(x_T, sol.y[2][-1], depth=depth+1)
        return [y, z_T]

    def generate_train_data(x):
        y, z_T = get_train_output(x, 0)
        return [x, y, z_T]

    t_filename = f'results/{system.name}_data_{n_samples}_samples_M_{M}.npy'
    z_filename = f'results/{system.name}_Z_values_{n_samples}_samples_M_{M}.npy'

    if os.path.exists(t_filename) and overwrite != True:
        print("Data exists. Loading training data...")
        t_data = np.load(t_filename)
        x_train, y_train = t_data[:, :-1], t_data[:, -1]
    else:
        print("Generating new training data...")
        x_train = np.array([np.random.uniform(dim[0], dim[1], n_samples) for dim in domain]).T
        
        y_train = np.zeros(n_samples)
        z_T = np.zeros(len(x_train))

        # Replace parallel function with loop to generate data
        for i in tqdm(range(len(x_train))):
            result = generate_train_data(x_train[i])
            x_train[i] = result[0]
            y_train[i] = result[1]
            z_T[i] = result[2]

        print("Saving training data...")
        t_data = np.column_stack((x_train, y_train))
        np.save(t_filename, t_data)

        # plt.figure()
        # plt.scatter(z_T, np.zeros_like(z_T) + 0.5)
        # plt.yticks([])
        # plt.xlabel('Value')
        # plt.xlim([0, 2000])
        # plt.title('Z Values Clustering')
        # plt.savefig(f'{z_filename}.pdf')
        # plt.close()

    # Rest of your plotting code remains unchanged...
    # plt.figure()    
    # plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='coolwarm', s=1)
    # plt.savefig(f"results/{system.name}_data_{n_samples}_samples_M_{M}.pdf")
    # plt.close()

    return np.column_stack((x_train, y_train))
