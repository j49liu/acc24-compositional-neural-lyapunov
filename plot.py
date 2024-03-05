import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.lines import Line2D
plt.rcParams['pdf.fonttype'] = 42  

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_all(systems, mu_s, nets, sos_V, c2_SOS=None, c2_V=None, c2_V_indy=None, c2_P=None, c2_P_indy=None):
    n = len(systems) // 2
    fig, axes = plt.subplots(n, 2, figsize=(12, 6*n))  
    
    domain = systems[0].domain
    d = len(systems[0].symbolic_vars)

    # Generate contour plot
    x1 = np.linspace(*domain[0], 400)
    x2 = np.linspace(*domain[1], 400)
    x1, x2 = np.meshgrid(x1, x2)

    zero_dims = d - 2
    x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]
    input_tensor = np.vstack([x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]).T
    x_test = torch.tensor(input_tensor, dtype=torch.float32)

    for i, system in enumerate(systems):
        net = nets[i]
        V_net = net(x_test)
        V_test = V_net.detach().cpu().numpy().reshape(x1.shape)

        P = system.P

        def quad_func_scalar(x1, x2):
            x = np.array([x1, x2])
            return x.T @ P @ x

        quad_func = np.vectorize(quad_func_scalar, excluded=[2])

        row = i // 2
        col = i % 2
        ax = axes[row][col]
        
        # levels = [c2_SOS[i]]  
        # cs_sos = ax.contour(x1, x2, sos_V[i](x1, x2), levels=levels, colors='g', linewidths=2, linestyles='-.', label="SOS")

        # levels = [c2_P[i]]
        # cs_P = ax.contour(x1, x2, quad_func(x1, x2), levels=levels, colors='r', linewidths=2, linestyles='--', label="quadratic")

        # levels = [c2_V[i]]
        # cs_V = ax.contour(x1, x2, V_test, colors='b', levels=levels, label="neural")    


        levels = [c2_SOS[i]]  
        cs_sos = ax.contour(x1, x2, sos_V[i](x1, x2), levels=levels, colors='g', linewidths=2, linestyles='-')
        levels = [1.0]  
        cs_sos_indy = ax.contour(x1, x2, sos_V[i](x1, x2), levels=levels, colors='g', linewidths=1, linestyles='-.')


        # levels = [c2_P[i]]
        # cs_P = ax.contour(x1, x2, quad_func(x1, x2), levels=levels, colors='r', linewidths=2, linestyles='-')
        # levels = [c2_P_indy[i]]
        # cs_P_indy = ax.contour(x1, x2, quad_func(x1, x2), levels=levels, colors='r', linewidths=1, linestyles='-.')


        levels = [c2_V[i]]
        cs_V = ax.contour(x1, x2, V_test, levels=levels, colors='b', linewidths=3, linestyles='-')
        levels = [c2_V_indy[i]]
        cs_V_indy = ax.contour(x1, x2, V_test, levels=levels, colors='b', linewidths=1, linestyles='-.')


        sos_line = Line2D([0], [0], color='g', linewidth=2, linestyle='-', label='SOS (network)')
        sos_indy_line = Line2D([0], [0], color='g', linewidth=1, linestyle='-.', label='SOS (individual)')

        # quad_line = Line2D([0], [0], color='r', linewidth=2, linestyle='-', label='quadratic (network)')
        # quad_indy_line = Line2D([0], [0], color='r', linewidth=1, linestyle='-.', label='quadratic (individual)')

        net_line = Line2D([0], [0], color='b', linewidth=3, linestyle='-', label='neural (network)')
        net_indy_line = Line2D([0], [0], color='b', linewidth=1, linestyle='-.', label='neural (individual)')

        # ax.legend(handles=[sos_line, sos_indy_line, quad_line, quad_indy_line, net_line, net_indy_line], loc='best')
        ax.legend(handles=[sos_line, sos_indy_line, net_line, net_indy_line], loc='best')


        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.clabel(cs_V, inline=1, fontsize=10)
        ax.set_title(f'Subsystem {i+1}: Verified Level sets ($\mu_{{{i+1}}}={mu_s[i]}$)')

    plt.tight_layout()
    plt.savefig(f'all_plots_density_3.pdf', format='pdf', dpi=300)

def plot_V(system, net, model_path, sos_V=None, c2_SOS=0.99, c1_V=None, c2_V=None, c1_P=None, c2_P=None):
    domain = system.domain
    d = len(system.symbolic_vars)
    # Generate contour plot
    x1 = np.linspace(*domain[0], 400)
    x2 = np.linspace(*domain[1], 400)

    x1, x2 = np.meshgrid(x1, x2)

    # Set other dimensions to zero
    zero_dims = d - 2
    x_dims = [np.zeros_like(x1) for _ in range(zero_dims)]

    # Stack x1, x2, and zero dimensions to form the input tensor
    input_tensor = np.vstack([x1.ravel(), x2.ravel()] + [x_dim.ravel() for x_dim in x_dims]).T
    x_test = torch.tensor(input_tensor, dtype=torch.float32).to(device)
    V_net = net(x_test)
    V_test = V_net.detach().cpu().numpy().reshape(x1.shape)

    P = system.P
    def quad_func_scalar(x1, x2):
        x = np.array([x1, x2])
        return x.T @ P @ x

    quad_func = np.vectorize(quad_func_scalar, excluded=[2])

    fig = plt.figure(figsize=(12, 6))  # Set figure size

    # Subplot 1: 3D surface plot of the learned function
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(x1, x2, V_test)  # Plot the learned function
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("V(x)")
    ax1.set_title("Learned Lyapunov Function")

    # Subplot 2: Contour plot of target set and level sets
    ax2 = fig.add_subplot(122)
    if sos_V is not None:
        levels = [c2_SOS]  # Level set for the sos_V
        cs0 = ax2.contour(x1, x2, sos_V(x1, x2), levels=levels, colors='g', linewidths=2, linestyles='-.')    
    levels = [c1_P, c2_P]
    cs1 = ax2.contour(x1, x2, quad_func(x1, x2), levels=levels, colors='r', linewidths=2, linestyles='--')
    if c1_V is not None:
        levels = [c1_V]
        cs = ax2.contour(x1, x2, V_test, colors='b', levels=levels)
    if c2_V is not None:
        levels = [c2_V]
        cs = ax2.contour(x1, x2, V_test, colors='b', levels=levels)    

    # ax2.plot(vsol.y[0], vsol.y[1], color='red', linewidth=2, label='Limit cycle (ROA boundary)')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.clabel(cs, inline=1, fontsize=10)
    ax2.set_title('Level sets')

    plt.tight_layout()
    plt.savefig(f'{model_path}.pdf', format='pdf', dpi=300)
    # plt.show()
