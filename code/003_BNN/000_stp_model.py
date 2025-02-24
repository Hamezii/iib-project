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
        return F.softplus(input=h, beta=1/self.alpha)

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
        J_eff = F.softplus(self.raw_J)

        # Update synaptic currents (Equation 4)
        # Note: J_ij is the synaptic weight from neuron j to neuron i
        synaptic_inputs = torch.matmul(J_eff, (u * x * R).T).T
        # print(h.shape)
        # print(synaptic_inputs.shape)
        # print(R_I.shape)
        # print(I_e.shape)
        # print(h)
        # print(synaptic_inputs)
        # print(R_I) # R_I goes to inf
        # print(I_e)
        # input()
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

        return (h_new, u_new, x_new, h_I_new), u_new * x_new

# ---- Initialization functions
def initialize_eta(N, P, f): # Unused currently
    """
    Generates an initial random Eta value for the STP model.

    Each memory is represented by a binary N-dimensional
    vector of zeros and ones.

    Parameters:
    N: Number of neurons.
    P: Number of memory patterns.
    f: Sparsity factor of the memories.

    Returns:
    eta: Binary matrix (P x N) representing memory patterns.
    """
    return (torch.rand(P, N) < f).astype(torch.int8)


def compute_connection_matrix(eta, J_EE): # Unused currently
    """
    Computes the synaptic connection matrix J where:
    J_ij = J_EE if neurons i and j share at least one memory pattern,
    else 0.

    Parameters:
    eta: Binary matrix (P x N) representing memory patterns.
    J_EE: Connection strength for excitatory-to-excitatory synapses.

    Returns:
    J: Synaptic connection matrix (N x N) with values 0 or J_EE.
    """
    # Dot product between all pairs of neurons (N x N)
    overlaps = eta.T @ eta
    # Binary matrix signifying which neurons are connected.
    connected = (overlaps >= 1)
    return connected * J_EE


def generate_xor_data(input_strength):
    X = torch.tensor([[0, 0], [0, input_strength], [input_strength, 0], [input_strength, input_strength]], dtype=torch.float32)
    Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    #X = X.repeat(batch_size, 1)
    #Y = Y.repeat(batch_size, 1)
    return X, Y


if __name__ == "__main__":
    # Hyperparameters
    # N = 2
    # dt = 1e-4
    # learning_rate = 10#3e-1
    # seq_len = 100
    # batch_size = 4
    # input_strength = 225

    # model = STPModel(N=N, dt=dt).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # # Initial state (h, u, x, h_I)
    # state = (torch.zeros(batch_size, N, requires_grad=True).to(device),
    #          torch.full((batch_size, N), model.U.item(), requires_grad=True).to(device),
    #          torch.ones(batch_size, N, requires_grad=True).to(device),
    #          torch.zeros(batch_size, 1, requires_grad=True).to(device))

    # # XOR dataset
    # X, Y = generate_xor_data(input_strength)
    # X, Y = X.to(device), Y.to(device)

    # # Training loop
    # losses = []
    # for epoch in range(1000):
    #     total_loss = 0

    #     # Forward pass
    #     hidden = tuple(s.clone() for s in state)
    #     for t in range(seq_len):
    #         hidden, u_x = model(hidden, X)

    #         # Check for NaNs in the forward pass
    #         if torch.isnan(u_x).any():
    #             print(f"NaN detected in u_x at time step {t}")
    #             break

    #         # Loss
    #         loss = F.binary_cross_entropy_with_logits(u_x[:, 0], Y)
    #         total_loss += loss
    #         loss.backward(retain_graph=True)

    #         # Gradient clipping
    #         # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    #     optimizer.step()
    #     optimizer.zero_grad()
    #     losses.append(total_loss.item() / seq_len)
    #     print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

    #     # Check for NaNs in the loss
    #     if torch.isnan(total_loss):
    #         print(f"NaN detected in total_loss at epoch {epoch}")
    #         break

    # # Plot the loss
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()

    # Hyperparameters
    N = 10#100
    dt = 1e-4
    learning_rate = 1#1e-4
    seq_len = 100 # Number of forward steps
    batch_size = 1#4
    input_strength = 225.0

    with device: # Putting tensors onto cuda device if able
        # Model
        model = STPModel(N=N, dt=dt)

        # Learning
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Initial state (h, u, x, h_I)
        state = (torch.zeros(batch_size, N, requires_grad=True),
                torch.full((batch_size, N), model.U.item(), requires_grad=True),
                torch.ones(batch_size, N, requires_grad=True),
                torch.zeros(batch_size, 1, requires_grad=True))
    
        # Input stimulating the first 5 neurons for 30ms
        I_e = torch.zeros(seq_len, batch_size, N)
        I_e[:30, :, :5] = input_strength

        eta = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        J_EE = 8
        J_initial = compute_connection_matrix(eta, J_EE)
        model.raw_J.data = torch.log(J_initial + 1e-3) F.inverse_softplus(J_initial)

    # Training loop
    losses = []
    for epoch in range(100):
        total_loss = 0

        # Forward pass
        with device:
            hidden = tuple(s.clone() for s in state)

        for t in tqdm(range(seq_len)):
            hidden, u_x = model(hidden, I_e[t])

            # Check for NaNs in the forward pass
            if torch.isnan(u_x).any():
                print(f"NaN detected in u_x at time step {t}")
                break

            # Loss
            # TODO - Dummy loss function, looks for larger efficacy 
            # in stimulated neurons and low in others
            target = torch.zeros_like(u_x)
            target[:, :5] = 0.8
            loss = torch.mean((u_x - target)**2) # mean squared error
            total_loss += loss
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Maybe loss.backward should be outside the loop?
        # total_loss.backward(retain_graph=True)

        optimizer.step()
        optimizer.zero_grad()
        losses.append(total_loss.item() / seq_len)
        print(f"Epoch {epoch}, Loss: {losses[-1]:.4f}")

        # Check for NaNs in the loss
        if torch.isnan(total_loss):
            print(f"NaN detected in total_loss at epoch {epoch}")
            break