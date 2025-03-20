# Implementation of the neuronal STP model, modelling individual models.

#import numpy as np

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ---- Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEBUG below:
# device = torch.device('cpu')
# print("Using device:", device)

TEMP_TEST = True

# ---- Model
class STPModel(nn.Module):
    def __init__(self, N=100, dt=1e-4, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, alpha=1.5, I_b=8, J_EI=1.1, J_IE=2.2):
        super().__init__()

        # Network parameters
        self.N = N
        self.dt = dt

        # Fixed parameters
        self.register_buffer('U', torch.tensor(U))
        self.register_buffer('tau', torch.tensor(tau))
        self.register_buffer('tau_f', torch.tensor(tau_f))
        self.register_buffer('tau_d', torch.tensor(tau_d))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('I_b', torch.tensor(I_b))
        self.register_buffer('J_EI', torch.tensor(J_EI))
        self.register_buffer('J_IE', torch.tensor(J_IE))

        # Trainable parameters
        # raw_J is put through softplus to ensure non-negativity
        self.raw_J = nn.Parameter(torch.randn(N, N) * 0.1)

    def compute_R(self, h):
        """Compute firing rate R(h) = alpha * ln(1 + exp(h / alpha))"""
        return nn.Softplus(beta=1/self.alpha)(h)

    def forward(self, state, I_e):
        """
        Perform update step for network model.
        
        Parameters:
            state: tuple of (h, u, x, h_I)
            I_e: External input of size (batch_size x N)
        Returns:
            next_state: tuple of updated state variables
            u_x: Synaptic efficacies of size (batch_size x N)
        
        """
        # TODO Consider flipping dimensions of state variables? Currently (batch_size, N)
        h, u, x, h_I = state

        R = self.compute_R(h)
        R_I = self.compute_R(h_I)

        # To keep J non-negative
        J_eff = nn.Softplus(beta=1.0)(self.raw_J)
        if TEMP_TEST:
            J_eff = self.raw_J

        # Update synaptic currents (Equation 4)
        # Note: J_ij is the synaptic weight from neuron j to neuron i
        
        synaptic_inputs = torch.matmul(J_eff, (u * x * R).T).T
        dh = (-h + synaptic_inputs - self.J_EI * R_I + self.I_b + I_e)/self.tau
        # Update facilitation variables (Equation 5)
        du = (self.U - u)/self.tau_f + self.U * (1 - u) * R
        # Update depression variables (Equation 6)
        dx = (1 - x)/self.tau_d - u * x * R
        # Update inhibitory population (Equation 7)
        dh_I = (-h_I + self.J_IE * torch.sum(R, dim=1, keepdim=True))/self.tau

        # Check for NaNs in intermediate variables
        if torch.isnan(dh).any() or torch.isnan(du).any() or torch.isnan(dx).any() or torch.isnan(dh_I).any():
            print("NaN detected in intermediate variables")
            print(f"dh: {dh}")
            print(f"du: {du}")
            print(f"dx: {dx}")
            print(f"dh_I: {dh_I}")
            return state, u * x

        # Euler integration step
        h_new = h + dh * self.dt
        u_new = u + du * self.dt
        x_new = x + dx * self.dt
        h_I_new = h_I + dh_I * self.dt

        return (h_new, u_new, x_new, h_I_new)


class STPWrapper(nn.Module):
    """
    Wrapper class for STPModel to handle input and output layers.
    """
    def __init__(self, N=100, dt=1e-4, in_size=1, out_size=1, **stp_kwargs):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.N = N
        self.dt = dt

        self.input_layer = nn.Linear(in_size, N)
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.input_layer.bias, 0.0)
        
        self.stp_model = STPModel(N=N, dt=dt, **stp_kwargs)
        
        self.output_layer = nn.Linear(N, out_size)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, inp):
        # inp has shape (seq_len, batch_size, in_size)
        seq_len = inp.size(0)
        batch_size = inp.size(1)
        # assert(inp.size(2) == self.in_size)
        
        # Initialize STP state variables
        h = torch.zeros(batch_size, self.N, requires_grad=True, device=inp.device)
        u = torch.full((batch_size, self.N), self.stp_model.U.item(), requires_grad=True, device=inp.device)
        x = torch.ones(batch_size, self.N, requires_grad=True, device=inp.device)
        h_I = torch.zeros(batch_size, 1, requires_grad=True, device=inp.device)
        state = (h, u, x, h_I)

        # Run through STP dynamics
        outputs = []
        states = []
        # outputs = torch.zeros(seq_len, batch_size, self.out_size, device=inp.device)
        for t in range(seq_len):
            I_e = self.input_layer(inp[t]) # Transform input to STP dimensions
            state = self.stp_model(state, I_e)
            states.append(state)
            
            # Add to outputs
            output = self.output_layer(self.stp_model.compute_R(state[0]))
            # outputs[t] = output
            outputs.append(output)

        # Stack outputs while retaining gradients for training
        return states, outputs


class PaperSTPWrapper(STPWrapper):
    def __init__(self, N=1000, P=16, f=0.05, J_EE=8, **kwargs):
        J_IE_default = 1.75
        neurons_per_cluster = int(N * f) # Must be the same as in the initialize_eta function
        kwargs['J_IE'] = kwargs.get('J_IE', J_IE_default) / neurons_per_cluster  # * P / N  # Scale J_IE
        J_EE /= neurons_per_cluster

        super().__init__(N=N, in_size=P, out_size=P, **kwargs)

        self.P = P
        # Generate memory patterns and connectivity
        self.eta = initialize_eta(P, N, f, random=True)
        self.J = compute_connection_matrix(self.eta, J_EE)
            

        # No training
        self.stp_model.raw_J.requires_grad_(False)
        self.input_layer.weight.requires_grad_(False)
        self.output_layer.weight.requires_grad_(False)

        # Invert softplus
        self.stp_model.raw_J.data = torch.log(torch.exp(self.J) - 1)
        if TEMP_TEST:
            self.stp_model.raw_J.data = self.J

        # Configure input/output to target clusters
        self._configure_input_output()

    def _configure_input_output(self):
        """Set input/output weights to map to/from clusters"""
        # Note that nn.Linear weights follow (output, input) convention
        P, N = self.eta.shape
        # Input
        self.input_layer.weight = nn.Parameter(self.eta.detach().clone().T)
        # Output
        self.output_layer.weight = nn.Parameter(self.eta.detach().clone())

class ClusterSTPWrapper(STPWrapper):
    """
    Helper class to emulate the per-cluster model with 1 neuron per cluster, 
    and non-zero connections between clusters.
    """
    def __init__(self, P=16, J_EE=8, f=0.05, **kwargs):
        # NOTE this method reuses code from PaperSTPWrapper.
        super().__init__(N=P, in_size=P, out_size=P, **kwargs)

        self.P = P
        # Initialize diagonal eta
        self.eta = torch.eye(P)
        # Baseline weak cross-cluster connections
        self.J = compute_connection_matrix(self.eta, J_EE, J_EE * f) 
        # No training
        self.stp_model.raw_J.requires_grad_(False)
        self.input_layer.weight.requires_grad_(False)
        self.output_layer.weight.requires_grad_(False)
        # Invert softplus
        self.stp_model.raw_J.data = torch.log(torch.exp(self.J) - 1)
        if TEMP_TEST:
            self.stp_model.raw_J.data = self.J
        # Configure input/output to target clusters
        self.input_layer.weight = nn.Parameter(self.eta.detach().clone().T)
        self.output_layer.weight = nn.Parameter(self.eta.detach().clone())


# ---- Initialization functions
def initialize_eta(P=16, N=100, f=0.05, random=True):
    """Create clustered memory patterns (P x N).
    
    random: True for random neurons per pattern, False for contiguous non-connected pattern assignments.
    """
    neurons_per_cluster = int(N * f)
    eta = torch.zeros(P, N)
    for p in range(P):
        if random:
            neurons = torch.randperm(N)[:neurons_per_cluster]
            eta[p, neurons] = 1
        else:
            start = p * neurons_per_cluster
            end = start + neurons_per_cluster
            eta[p, start:end] = 1
    return eta

def compute_connection_matrix(eta, J_EE=8, J_0=0):
    """J_ij = J_EE if i,j share a pattern, else J_0."""
    J = torch.ones(eta.shape[1], eta.shape[1]) * J_0
    J[(eta.T @ eta).bool()] = J_EE
    return J


# ---- Data generation
def generate_xor_data(input_strength):
    X = torch.tensor([[0, 0], [0, input_strength], [input_strength, 0], [input_strength, input_strength]], dtype=torch.float32)
    Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    return X, Y


# ---- Methods
def simulate_paper():
    # Initialize model with paper parameters
    P = 16 # 16
    N = 1000 # 1000
    f = 0.05 # 0.05
    dt = 1e-3
    model = PaperSTPWrapper(
        N=N, P=P, f=f, dt=dt,
        J_EE=8, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, I_b = 8.0
    ).to(device)
    input_strength = 365.0 # Pt. 2.3 of supplemental material
    duration = 2.5
    simulate_paper_with_model(model, input_strength, duration)

def simulate_cluster_stp():
    model = ClusterSTPWrapper(P=16, f=0.05, dt=1e-4, J_EE=8, U=0.3, tau=8e-3, tau_f=1.5, tau_d=0.3, J_IE=1.75, J_EI=1.1, I_b = 8.0).to(device)
    simulate_paper_with_model(model, input_strength=225.0, duration=2.0)

def simulate_paper_with_model(model:STPWrapper, input_strength, duration=2.5):
    """Simulate the paper test with the given model and input strength, for 'duration' seconds."""
    # Stimulation sequence
    seq_len = int(duration / model.dt)
    batch_size = 1
    inputs = torch.zeros(seq_len, batch_size, model.P, device=device)
    for p in range(5):
        start = p * int(100e-3 / model.dt)
        end = start + int(30e-3 / model.dt)
        inputs[start:end, 0, p] = input_strength

    # Plotting inputs
    # plt.figure()
    # for p in range(5):
    #     plt.plot(inputs[:, 0, p].tolist(), label=f'Cluster {p+1}')
    # plt.xlabel('Time (0.1ms steps)')
    # plt.ylabel('Input')
    # plt.legend()
    # plt.show()

    # Run simulation
    states, outputs = model(inputs)
    # (seq_len, state index, batch, neuron)
    ux_traces = {p: [] for p in range(5)}
    state_vars = ('h', 'u', 'x', 'h_I')
    scalar_vars = ('h_I',)
    per_neuron_vars = ('h', 'u', 'x')
    state_traces = {}
    for variable in per_neuron_vars: # Variables with per-neuron values
        state_traces[variable] = {p: [] for p in range(5)}
    for variable in scalar_vars: # Variables with scalar values
        state_traces[variable] = []

    for t in range(seq_len):
        state = states[t]
        for i, var in enumerate(state_vars):
            if var in per_neuron_vars: # If variable has per-neuron values
                var_vals = state[i][0, :]
                for p in range(5):
                    cluster_neurons = model.eta[p].bool()
                    state_traces[var][p].append(var_vals[cluster_neurons].mean().item())
            else: # If variable is scalar
                state_traces[var].append(state[i].item())

        u, x = state[1], state[2]
        u_x = u[0, :] * x[0, :]
        assert u_x.shape == (model.N,), f"u_x shape: {u_x.shape}"
        for p in range(5):
            cluster_neurons = model.eta[p].bool()
            ux_traces[p].append(u_x[cluster_neurons].mean().item())

    # Plot results
    plt.figure()
    for p in range(5):
        plt.plot(ux_traces[p], label=f'Cluster {p+1}')
    plt.xlabel(f'Time ({model.dt * 1e3}ms steps)')
    plt.ylabel('Synaptic Efficacy (u*x)')
    plt.legend()
    plt.show()

    # Plot state variables in a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ['Mean h (Cluster)', 'Mean u (Cluster)', 'Mean x (Cluster)', 'h_I']

    for i, variable in enumerate(state_vars):
        ax = axes[i // 2, i % 2]
        if variable in per_neuron_vars:
            for p in range(5):
                ax.plot(state_traces[variable][p], label=f'Cluster {p+1}')
        else:  # Scalar variable
            ax.plot(state_traces[variable], label=variable)
        ax.set_title(titles[i])
        ax.set_xlabel('Time (0.1ms steps)')
        ax.set_ylabel(variable)
        ax.legend()

    plt.tight_layout()
    plt.show()

def train_xor():
    # Hyperparameters
    N = 100
    dt = 1e-4
    learning_rate = 1#0.001
    seq_len = 100 # 600
    num_epochs = 1000
    input_strength = 225 # TEMP
    input_duration = 50

    # Model and Optimizer
    model = STPWrapper(N=N, in_size=2, out_size=1, dt=dt).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Data
    X, Y = generate_xor_data(input_strength)
    X, Y = X.to(device), Y.to(device)
    assert X.shape == (4, 2), f"X shape: {X.shape}"
    x_input = torch.zeros(seq_len, 4, 2, device=device)
    for t in range(input_duration):
        x_input[t] = X

    # Training
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        states, outputs = model(x_input)
        final_output = outputs[-1].squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(final_output, Y)
        loss.backward()
        # print(model.stp_model.raw_J.grad) # Very small gradients...
        optimizer.step()

        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Plotting
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    simulate_cluster_stp()
    # simulate_paper()
    # train_xor()