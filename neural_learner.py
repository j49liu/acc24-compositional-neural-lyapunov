import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 123  # Choose your seed

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

np.random.seed(seed)

class Net(nn.Module):
    def __init__(self, num_inputs, num_layers, width, activation_function=nn.Tanh()):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, width))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(width, width))
        self.final_layer = nn.Linear(width, 1)  # Added a final layer for output
        self.activation = activation_function

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x


def Zubov_loss(x, net, system, mu=0.1, beta=1.0, c1=0.01, c2=1):
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    def evaluate_dynamics(f, x):
        # Split the tensor into its components
        x_split = torch.split(x, 1, dim=1)
        return [fi(x_split[0].squeeze(), x_split[1].squeeze()) for fi in f]

    f_values = evaluate_dynamics(system.f, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_dot = (V_grad * f_tensor).sum(dim=1)
    norm_sq = (x**2).sum(dim=1)
    
    mask = (beta**2 - norm_sq) > 0

    loss = ( 
            (V_dot + mu * norm_sq * (1-V) * (1+V))**2 
            + torch.where(mask, torch.clamp(torch.tanh(c1 * norm_sq) - V, min=0)**2, torch.zeros_like(norm_sq))
            + torch.where(mask, torch.clamp(torch.tanh(c2 * norm_sq) - V, max=0)**2, torch.zeros_like(norm_sq))
            # + V_zero**2 
           ).mean()
    
    return loss

def Lyapunov_loss(x, net, system, beta=1.0, l1=1.0, l2=1.0, c2=100.0, c1=0.01):
    x.requires_grad = True
    zero_tensor = torch.zeros_like(x[0]).unsqueeze(0).to(device)
    V_zero = net(zero_tensor)
    V = net(x).squeeze()
    f = system.f    
    V_grad = torch.autograd.grad(V.sum(), x, create_graph=True)[0]

    def evaluate_dynamics(f, x):
        # Split the tensor into its components
        x_split = torch.split(x, 1, dim=1)
        return [fi(x_split[0].squeeze(), x_split[1].squeeze()) for fi in f]

    f_values = evaluate_dynamics(system.f, x)
    f_tensor = torch.stack(f_values, dim=1)

    V_dot = (V_grad * f_tensor).sum(dim=1)

    norm_sq = (x**2).sum(dim=1)

    loss = (
            V_zero**2 
            # + l1*(V_dot + 0.1*norm_sq)**2 
            + l1*torch.relu(V_dot + 0.1*norm_sq)**2 
            # + torch.relu(1 - V)*torch.relu(V_dot + 0.1*V)**2 
            # + torch.relu(beta**2 - norm_sq)*torch.relu(1.0 - V)**2
            # + l2*torch.relu(V - c2 * norm_sq)**2
            # + l2*torch.relu(c1 * norm_sq - V)**2
           ).mean()
    
    return loss

def neural_learner(system, data=None, data_only=False, layer=3, width=10, N=100000, 
                    batch_size=32, max_epoch=5, lr=0.01, overwrite=False, activation_function=nn.Tanh()): 

    domain = system.domain
    d = len(system.symbolic_vars)
    net = Net(d, layer, width, activation_function=activation_function).to(device)
    model_name = system.name
    if data is None:
        model_path = f"results/{model_name}_N={N}_epoch={max_epoch}_layer_{layer}_width_{width}_lr_{lr}.pt"
    elif data_only == True:
        num_data_points = data.shape[0]
        model_path = f"results/{model_name}_{num_data_points}_samples_epoch={max_epoch}_layer_{layer}_width_{width}_lr_{lr}.pt"        
    else: 
        num_data_points = data.shape[0]
        model_path = f"results/{model_name}_{num_data_points}_samples_N={N}_epoch={max_epoch}_layer_{layer}_width_{width}_lr_{lr}.pt"

    if not overwrite and os.path.isfile(model_path):
        print("Model exists. Loading model...")
        net = torch.load(model_path, map_location=device)
    else:
        # Optimizer and loss
        print("Training model...")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # Collocation points
        ranges = torch.tensor(domain).to(device)
        x_train = torch.rand((N, d)).to(device) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        
        # "Ground truth" data for data-augmented training
        if data is not None:
            x_data, y_data = data[:, :-1], data[:, -1]
            x_data_tensor = torch.FloatTensor(x_data).to(device)
            y_data_tensor = torch.FloatTensor(y_data).to(device)

        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size

        max_epoch_loss = float('inf')
        average_epoch_loss = float('inf')
        epoch = 0
        
        start_time = time.time()

        while (average_epoch_loss > 1e-5 or max_epoch_loss > 1e-5) and epoch < max_epoch:
            total_loss = 0.0
            losses = []  # Store losses for computing max epoch loss

            # Shuffle the training data for each epoch
            indices = torch.randperm(num_samples)
            x_train_shuffled = x_train[indices]
            
            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", unit="batch", leave=False)
            for i in progress_bar:
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                x_batch = x_train_shuffled[batch_start:batch_end]

                if data is None:
                    data_loss = 0
                else:
                    data_loss = torch.mean((net(x_data_tensor).squeeze() - y_data_tensor)**2)

                if data_only == True: 
                    # print(f"Training model with data only ({num_data_points} data points used)...")
                    loss = data_loss
                else: 
                    loss = Zubov_loss(x_batch, net, system) + data_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                losses.append(loss.item())

                progress_bar.set_postfix(loss=loss.item())

            average_epoch_loss = total_loss / num_batches
            max_epoch_loss = max(losses)  # Compute max epoch loss from the stored losses

            print(f"Epoch {epoch + 1} completed. Average epoch loss: {average_epoch_loss:.5g}, Maximum epoch loss: {max_epoch_loss:.5g}")
            epoch += 1

        elapsed_time = time.time() - start_time
        print(f"Total training time: {elapsed_time:.2f} seconds.")
        print(f"Model trained: {model_path}")
        # Save model
        torch.save(net, model_path)

    return net, model_path 
