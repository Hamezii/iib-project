# Implementation of the neuronal STP model, modelling individual models.
# WIP

import numpy as np


class STPModel:
    def __init__(self, U=0.2, tau_f=1500, tau_d=200):
        """
        Initialize the STP model parameters.
        U: Utilization of synaptic efficacy
        tau_f: Time constant of facilitation (ms)
        tau_d: Time constant of depression (ms)
        """
        self.U = U
        self.tau_f = tau_f
        self.tau_d = tau_d
        self.u = 0
        self.x = 1

    def update(self, dt, spike):
        """
        Update the synaptic variables based on the spike input.
        This is for excitatory-to-excitatory synapses. 
        dt: Time step (ms)
        spike: Binary spike input (0 or 1)
        """
        dx = (1-self.x)/self.tau_d - self.u*self.x*spike
        du = (self.U-self.u)/self.tau_f + self.U*(1-self.u)*spike

        self.x += dx*dt
        self.u += du*dt

        return self.u*self.x

def initialize_eta(P, N, f):
    """
    Generates an initial random Eta value for the STP model.
    
    Each memory is represented by a binary N-dimensional
    vector of zeros and ones.

    Parameters:
    P: Number of memory patterns.
    N: Number of excitatory neurons.
    f: Sparsity factor of the memories.
    
    Returns:
    eta: Binary matrix (P x N) representing memory patterns.
    """
    return (np.random.rand(P, N) < f).astype(np.int8)

def compute_connection_matrix(eta, J_EE):
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



# Example usage
if __name__ == "__main__":
    stp = STPModel()
    dt = 1
    spikes = [0, 1, 0, 0, 1, 0, 1, 0, 0, 0]

    for spike in spikes:
        efficacy = stp.update(dt, spike)
        print(f"Spike: {spike}, Synaptic Efficacy: {efficacy}")
