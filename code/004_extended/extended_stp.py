# Implementation of the neuronal STP model, modelling individual models.

#import numpy as np

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# ---- Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') # If just want to use cpu


# --- Parametrizations
# Adds preprocessing to network parameters before use.
# Useful since parameter training domain is unconstrained.
# Usage:
#  parametrize.register_parametrization(module, "parameter", ParametrizationReLU(positive=True))
class ParametrizationSoftplus(nn.Module): # NOTE not currently used.
    """Enforce sign of module parameter by using softplus."""
    softplus = nn.Softplus(beta=1.0/0.2) # alpha of 0.2, very close to ReLU
    def __init__(self, positive=True):
        super().__init__()
        self.positive = positive
        if positive:
            self.func = lambda X: self.softplus(X)
        else:
            self.func = lambda X: - self.softplus(X)
    def forward(self, X):
        return self.func(X)
    def right_inverse(self, A):
        # For softplus with beta 1: would be torch.log(torch.exp(A) - 1)
        # For now, setting the parameter will directly set preprocessed value
        return A

class ParametrizationReLU(nn.Module): # Used for constraining non-negative matrix weights
    """Enforce sign of module parameter by using ReLU."""
    relu = nn.ReLU()
    def __init__(self, positive=True):
        super().__init__()
        self.positive = positive
    def forward(self, X):
        if self.positive:
            return self.relu(X)
        else:
            return -self.relu(-X)
    def right_inverse(self, A):
        return A


# ---- Model
class RecurrentLayer(nn.Module):
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
        # TODO parametrize.register_parametrization(self, "J", ParametrizationReLU(positive=True))
        self.J = nn.Parameter(torch.zeros(N, N))
        assert self.J.shape == (N, N), self.J.shape

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

        # Update synaptic currents (Equation 4)
        # Note: J_ij is the synaptic weight from neuron j to neuron i
        # TODO ensure self.J non-negativity with parametrization

        synaptic_inputs = torch.matmul(self.J, (u * x * R).T).T
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
    """Wrapper class for the recurrent neurons to handle input and output layers."""
    def __init__(self, N=1000, dt=1e-4, in_size=1, out_size=1, **stp_kwargs):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.N = N
        self.dt = dt

        self.input_layer = nn.Linear(in_size, N, bias=False)
        self.stp_layer = RecurrentLayer(N=N, dt=dt, **stp_kwargs)
        self.output_layer = nn.Linear(N, out_size, bias=False)

        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize parameters, overridden by child classes."""
        nn.init.normal_(self.input_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.input_layer.bias, 0.0)

        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, inp, step_hook_func=None):
        """Perform forward pass through all time for input inp.
        
        inp: shape (seq_len, batch_size, channels)
        step_hook_func: func w/ params(recurrent_layer_model, inp, outp) to call at each step
        """
        seq_len, batch_size, _ = inp.shape
        # assert(inp.size(2) == self.in_size)
        # Initialize STP state variables
        h = torch.zeros(batch_size, self.N, device=inp.device)
        u = torch.full((batch_size, self.N), self.stp_layer.U.item(), device=inp.device)
        x = torch.ones(batch_size, self.N, device=inp.device)
        h_I = torch.zeros(batch_size, 1, device=inp.device)
        state = (h, u, x, h_I)
        # Run through STP dynamics
        states = []
        outputs = torch.zeros(seq_len, batch_size, self.out_size, device=inp.device)
        # outputs = []
        if step_hook_func is not None:
            hook = self.stp_layer.register_forward_hook(step_hook_func)

        with parametrize.cached(): # Cache paramtrization calls for faster processing
            for t in range(seq_len):
                I_e = self.input_layer(inp[t]) # Transform input to STP dimensions
                state = self.stp_layer(state, I_e)
                states.append(state)

                # Add to outputs
                output = self.output_layer(self.stp_layer.compute_R(state[0]))
                outputs[t, :, :] = output
                # outputs.append(output)

        if step_hook_func is not None:
            hook.remove()
        return states, outputs

class PaperSTPWrapper(STPWrapper):
    def __init__(self, N=1000, P=16, f=0.05, J_EE=8.0, **kwargs):
        J_IE_default = 1.75
        neurons_per_cluster = int(N * f) # Must be the same as in the initialize_eta function

        # Scale connection strengths by neuron counts
        kwargs['J_IE'] = kwargs.get('J_IE', J_IE_default) / neurons_per_cluster
        J_EE /= neurons_per_cluster

        self.P = P
        # Generate memory patterns and connectivity
        self.eta = initialize_eta(P, N, f, random=True)
        self.J = compute_connection_matrix(self.eta, J_EE)

        super().__init__(N=N, in_size=P, out_size=P, **kwargs)

    def initialize_parameters(self):
        self.stp_layer.J.requires_grad = False
        self.input_layer.weight.requires_grad = False
        self.output_layer.weight.requires_grad = False
        with torch.no_grad():
            # Don't know if this method passes gradients
            self.stp_layer.J.copy_(self.J)
            self.input_layer.weight.copy_(self.eta.T)
            self.output_layer.weight.copy_(self.eta)
        # self.stp_layer.J.data = self.J
        # detach() here removes gradient, clone() makes new memory copy
        # self.input_layer.weight = nn.Parameter(self.eta.detach().clone().T)
        # self.output_layer.weight = nn.Parameter(self.eta.detach().clone())

class ExtendedSTPWrapper(STPWrapper):
    def __init__(self, N_a=1000, N_b=1000, P=16, f=0.05, J_EE=8.0, **kwargs):
        self.N_a=N_a # Number of default neurons
        self.N_b=N_b # Nunber of additional computational neurons
        self.N = N_a + N_b # NOTE self.N is the total number of neurons

        # Scale connection strengths by neuron counts
        neurons_per_cluster = int(N_a * f) # Must be the same as in the initialize_eta function
        J_IE_default = 1.75
        kwargs['J_IE'] = kwargs.get('J_IE', J_IE_default) / neurons_per_cluster
        J_EE /= neurons_per_cluster

        self.P = P
        # Generate memory patterns and connectivity
        # NOTE these are both for the unextended model
        self.eta = initialize_eta(P, N_a, f, random=True)
        self.J_orig = compute_connection_matrix(self.eta, J_EE)

        super().__init__(N=self.N, in_size=P, out_size=P, **kwargs)

    def initialize_parameters(self):
        trainable_B = False
        trainable_C = True
        positive_C = True
        C_sparsity = 0.1
        C_std = 1
        trainable_base_neurons = False
        constrain_comp_neurons = True
        # New extended J matrix: with quadrants
        self.J_11 = nn.Parameter(self.J_orig, requires_grad=trainable_base_neurons)
        if trainable_base_neurons:
            parametrize.register_parametrization(self, "J_11", ParametrizationReLU(positive=True))
        # Zero feedback from comp neurons to base neurons:
        self.J_12 = torch.zeros(self.N_a, self.N_b)
        # Computational neuron weights
        self.J_21 = nn.Parameter(torch.zeros(self.N_b, self.N_a))
        self.J_22 = nn.Parameter(torch.zeros(self.N_b, self.N_b))
        if constrain_comp_neurons:
            parametrize.register_parametrization(self, "J_21", ParametrizationReLU(positive=True))
            parametrize.register_parametrization(self, "J_22", ParametrizationReLU(positive=True))
        # Xavier/Glorot init is better suited for sigmoid/tanh activations.
        # Have ReLU-adjacent activation, so using He/kaiman initialization.
        # This initialization considers number of inputs, but there might be orientation mismatch.
        #  - Assumes W.shape = [fan_out, fan_in] i.e. x @ W.T, which I believe is consistent.
        nn.init.kaiming_normal_(self.J_21, nonlinearity="relu")
        nn.init.kaiming_normal_(self.J_21, nonlinearity="relu")
        # NOTE check gradients are correct processed with this method
        self.stp_layer.J = nn.Parameter(torch.vstack((
            torch.hstack((self.J_11, self.J_12)),
            torch.hstack((self.J_21, self.J_22))
        )))

        # Input (N x P)
        self.B_1 = nn.Parameter(self.eta.T, requires_grad=trainable_B)
        if trainable_B:
            parametrize.register_parametrization(self, "B_1", ParametrizationReLU(positive=True))
        # Zero computational neuron input
        self.B_2 = torch.zeros(self.N_b, self.P, requires_grad=False)
        # TODO check requires_grad logic for this parameter, where it depends on previous parameter
        #  - Can't just assign a torch.Tensor since .weight is expecting a nn.Parameter
        self.input_layer.weight = nn.Parameter(torch.vstack((self.B_1, self.B_2)), requires_grad=True)

        # Output (P x N)
        self.C_1 = torch.zeros(self.P, self.N_a, requires_grad=False)
        self.C_2 = nn.Parameter(torch.zeros(self.P, self.N_b), requires_grad=trainable_C)
        if positive_C:
            parametrize.register_parametrization(self, "C_2", ParametrizationReLU(positive=True))
        # Consider using nn.init.sparse_() or nn.init.kaiming_normal()
        nn.init.sparse_(self.C_2, sparsity=C_sparsity, std=C_std)
        self.output_layer.weight = nn.Parameter(torch.hstack((self.C_1, self.C_2)), requires_grad=True)

class ClusterSTPWrapper(STPWrapper):
    """
    Helper class to emulate the per-cluster model with 1 neuron per cluster, 
    and non-zero connections between clusters.
    """
    def __init__(self, P=16, J_EE=8.0, f=0.05, **kwargs):
        # NOTE this method reuses code from PaperSTPWrapper.
        self.P = P
        # Initialize diagonal eta
        self.eta = torch.eye(P)
        # Baseline weak cross-cluster connections
        self.J = compute_connection_matrix(self.eta, J_EE, J_EE * f)

        super().__init__(N=P, in_size=P, out_size=P, **kwargs)

    def initialize_parameters(self):
        # No training
        # self.stp_layer.J.requires_grad_(False)
        # self.input_layer.weight.requires_grad_(False)
        # self.output_layer.weight.requires_grad_(False)
        self.stp_layer.J = nn.Parameter(self.J, requires_grad=False)
            
        # Configure input/output to target clusters
        self.input_layer.weight = nn.Parameter(self.eta.detach().clone().T, requires_grad=False)
        self.output_layer.weight = nn.Parameter(self.eta.detach().clone(), requires_grad=False)

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

def compute_connection_matrix(eta, J_EE=8.0, J_0=0.0):
    """J_ij = J_EE if i,j share a pattern, else J_0."""
    J = torch.full((eta.shape[1], eta.shape[1]), J_0, dtype=torch.float32)
    J[(eta.T @ eta).bool()] = J_EE
    return J
